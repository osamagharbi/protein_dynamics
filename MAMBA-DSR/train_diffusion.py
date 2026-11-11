#!/usr/bin/env python3
"""Stage B: direct bidirectional-Mamba regression of DSR latents.

This file keeps the original Stage-B entry point for workflow compatibility,
but there is no diffusion process here: no noisy latent, epsilon target,
DDPM schedule, residual model, or DDIM sampler is constructed.
"""
import argparse
import os
import time

import torch
from pyhocon import ConfigFactory

DSR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DSR")
import sys
sys.path.insert(0, DSR_DIR)
from lib.dataset import RawDataset  # noqa: E402

from model import MambaLatentPredictor, count_params
from protocol import dsr_eval_indices, protocol_times, seed_everything, validate_indices  # noqa: E402

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp")


def load_frame_times(conf_path, case):
    conf_text = open(conf_path).read().replace("CASE_NAME", case)
    conf = ConfigFactory.parse_string(conf_text)
    data_dir = os.path.join(DSR_DIR, conf.get_string("dataset.data_dir"))
    raw = RawDataset(
        data_dir=data_dir,
        extension=conf.get_string("dataset.extension"),
        d_in=conf.get_int("dataset.d_in"),
        with_normal=conf.get_bool("dataset.with_normal"),
    )
    return torch.tensor([raw[i][2] for i in range(len(raw))], dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a-ckpt", type=str, required=True)
    ap.add_argument("--conf", type=str, default=os.path.join(DSR_DIR, "confs", "abeta.conf"))
    ap.add_argument("--case", type=str, default="abeta_150")
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--time-emb-dim", type=int, default=64)
    ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--d-conv", type=int, default=4)
    ap.add_argument("--expand", type=int, default=2)
    ap.add_argument("--bimamba-type", type=str, default="v4",
                    choices=["v2", "v4"])
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--allow-legacy-split", action="store_true",
                    help="Allow a Stage-A checkpoint without protocol metadata.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("MAMBA-DSR requires CUDA for the bundled bidirectional Mamba kernels")
    seed_everything(args.seed)
    device = torch.device("cuda")

    ckpt = torch.load(args.stage_a_ckpt, map_location="cpu", weights_only=False)
    z0_all = ckpt["latent_codes"]["weight"].float().to(device)
    num_latents, latent_dim = z0_all.shape
    train_indices = ckpt.get("train_frame_indices")
    if train_indices is None:
        if not args.allow_legacy_split:
            raise RuntimeError(
                "Stage-A checkpoint has no train_frame_indices. Re-run Stage A with "
                "--protocol dsr_abeta, or pass --allow-legacy-split."
            )
        train_indices = list(range(num_latents))
        protocol_name = "legacy_stage_b_holdout"
    else:
        protocol_name = ckpt.get("protocol", "unknown")
    full_num_frames = int(ckpt.get("full_num_frames", max(train_indices) + 1))
    validate_indices(train_indices, full_num_frames, "train_frame_indices")
    if len(train_indices) != num_latents:
        raise ValueError("Stage-A train-frame metadata does not match the latent table")

    t_all = torch.tensor(protocol_times(full_num_frames, train_indices), device=device)
    train_idx = torch.tensor(train_indices, dtype=torch.long, device=device)
    t_train = t_all[train_idx]
    eval_indices = dsr_eval_indices(full_num_frames, train_indices)
    print(f"[data] {num_latents} training latents, dim={latent_dim}")
    print(f"[split] evaluation frames are held out: {eval_indices}")

    # Normalize targets for stable direct regression and store the transform in
    # the checkpoint so Stage C can invert it exactly.
    latent_mean = z0_all.mean(dim=0, keepdim=True)
    latent_std = z0_all.std(dim=0, keepdim=True).clamp_min(1e-6)
    targets = (z0_all - latent_mean) / latent_std

    model_config = {
        "latent_dim": latent_dim,
        "time_emb_dim": args.time_emb_dim,
        "hidden": args.hidden,
        "depth": args.depth,
        "d_state": args.d_state,
        "d_conv": args.d_conv,
        "expand": args.expand,
        "bimamba_type": args.bimamba_type,
    }
    model = MambaLatentPredictor(**model_config).to(device)
    print(f"[mamba] Bidirectional Mamba {count_params(model):,} params")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = os.path.join(OUT_ROOT, args.run_name or f"{args.case}_mamba_lat{latent_dim}")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(num_latents, device=device)
        losses = []
        for start in range(0, num_latents, args.batch_size):
            idx = perm[start:start + args.batch_size]
            pred = model(t_train[idx])
            loss = torch.nn.functional.mse_loss(pred, targets[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        if epoch % 1000 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"[mamba epoch {epoch}/{args.epochs}] mse={sum(losses) / len(losses):.6f} "
                  f"({time.time() - t0:.1f}s)")
        if epoch % 2000 == 0 or epoch == args.epochs:
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "latent_mean": latent_mean.cpu(),
                "latent_std": latent_std.cpu(),
                "epoch": epoch,
                "stage_a_ckpt": args.stage_a_ckpt,
                "case": args.case,
                "protocol": protocol_name,
                "train_frame_indices": train_indices,
                "eval_frame_indices": eval_indices,
                "train_times": t_train.cpu().tolist(),
                "eval_times": t_all[eval_indices].cpu().tolist(),
                "full_num_frames": full_num_frames,
                "num_train_frames": num_latents,
                "latent_dim": latent_dim,
                "seed": args.seed,
                "architecture": "bidirectional_mamba_direct_latent_regression",
            }, os.path.join(out_dir, "ckpt_latest.pth"))
            print(f"[checkpoint] saved -> {out_dir}/ckpt_latest.pth")

    print(f"[done] total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
