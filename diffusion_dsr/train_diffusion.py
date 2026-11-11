#!/usr/bin/env python3
"""
Stage B: train a small DDPM over the training-frame latents produced by Stage A
(train_autodecoder.py). Under the frozen-DSR protocol, Stage A contains only
frames 0,5,...,145, and Stage B is trained on all of those latents. The
interleaved evaluation-frame latents are never loaded.

Usage:
    conda activate graphmae
    cd diffusion_dsr
    python train_diffusion.py --stage-a-ckpt exp/abeta_dsr_protocol_v2/checkpoints/ckpt_latest.pth \
        --case abeta_150 --run-name abeta_dsr_protocol_v2
"""
import argparse
import os
import sys
import time

import torch
from pyhocon import ConfigFactory

DSR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DSR")
sys.path.insert(0, DSR_DIR)
from lib.dataset import RawDataset  # noqa: E402

from diffusion import GaussianDiffusion
from model import LatentDenoiser, MeanPredictor, count_params
from protocol import dsr_eval_indices, protocol_times, seed_everything, validate_indices  # noqa: E402

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp")


def load_frame_times(conf_path, case):
    """Recompute the (deterministic, sorted-by-file) per-frame physical time
    array the same way Stage A's FrameIndexedDataset ordered them, so frame
    index i here matches latent row i in the Stage A checkpoint exactly."""
    conf_text = open(conf_path).read().replace("CASE_NAME", case)
    conf = ConfigFactory.parse_string(conf_text)
    data_dir = os.path.join(DSR_DIR, conf.get_string("dataset.data_dir"))
    raw = RawDataset(
        data_dir=data_dir,
        extension=conf.get_string("dataset.extension"),
        d_in=conf.get_int("dataset.d_in"),
        with_normal=conf.get_bool("dataset.with_normal"),
    )
    times = torch.tensor([raw[i][2] for i in range(len(raw))], dtype=torch.float32)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a-ckpt", type=str, required=True)
    ap.add_argument("--conf", type=str, default=os.path.join(DSR_DIR, "confs", "abeta.conf"))
    ap.add_argument("--case", type=str, default="abeta_150")
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=256,
                     help="With only ~100-1000 tiny (32-dim) latent vectors total, this "
                          "workload is Python/kernel-launch-overhead-bound, not GPU-compute-"
                          "bound -- default is large enough to fit the whole training set "
                          "in a single batch per epoch (fewer, bigger steps beats more, "
                          "smaller ones here). Raise further if more datasets are added.")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mean-epochs", type=int, default=5000,
                     help="v2: epochs to pre-train the deterministic MeanPredictor mu(t) "
                          "before freezing it and training the diffusion on residuals.")
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--allow-legacy-split", action="store_true",
                     help="Allow loading an old Stage-A checkpoint without protocol metadata. "
                          "This is exploratory only and is not the fair comparison protocol.")
    args = ap.parse_args()

    seed_everything(args.seed)

    ckpt = torch.load(args.stage_a_ckpt, map_location="cpu", weights_only=False)
    z0_all = ckpt["latent_codes"]["weight"].float().cuda()  # (num_train_frames, latent_dim)
    num_latents, latent_dim = z0_all.shape
    stage_a_train_indices = ckpt.get("train_frame_indices")
    if stage_a_train_indices is None:
        if not args.allow_legacy_split:
            raise RuntimeError(
                "Stage-A checkpoint has no train_frame_indices. Re-run Stage A with "
                "--protocol dsr_abeta, or pass --allow-legacy-split for exploratory use."
            )
        stage_a_train_indices = list(range(num_latents))
        protocol_name = "legacy_stage_b_holdout"
    else:
        protocol_name = ckpt.get("protocol", "unknown")
    full_num_frames = int(ckpt.get("full_num_frames", max(stage_a_train_indices) + 1))
    validate_indices(stage_a_train_indices, full_num_frames, "stage_a_train_indices")
    if len(stage_a_train_indices) != num_latents:
        raise ValueError("Stage-A train-frame metadata does not match the latent table")

    print(f"[data] {num_latents} training-frame latents, dim={latent_dim}, loaded from {args.stage_a_ckpt}")
    t_phys_all = torch.tensor(
        protocol_times(full_num_frames, stage_a_train_indices),
        dtype=torch.float32,
        device="cuda",
    )

    train_frame_indices = torch.tensor(stage_a_train_indices, dtype=torch.long, device="cuda")
    t_phys = t_phys_all[train_frame_indices]
    eval_indices = dsr_eval_indices(full_num_frames, stage_a_train_indices)
    print(f"[split] Stage B trains on all {num_latents} Stage-A training latents")
    print(f"[split] evaluation frames are held out from Stage A and Stage B: {eval_indices}")

    z0 = z0_all
    num_frames = z0.shape[0]

    run_name = args.run_name or f"{args.case}_hybrid_lat{latent_dim}"
    out_dir = os.path.join(OUT_ROOT, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Phase 1: deterministic mean predictor mu(t) -> latent -----------
    # Trained on RAW latents (predict the actual latent at each time). On a
    # smooth latent trajectory this is near-perfect regression.
    mean_model = MeanPredictor(latent_dim=latent_dim).cuda()
    print(f"[mean] MeanPredictor {count_params(mean_model):,} params")
    mean_opt = torch.optim.Adam(mean_model.parameters(), lr=1e-3, weight_decay=1e-6)
    t0 = time.time()
    for epoch in range(1, args.mean_epochs + 1):
        mean_model.train()
        pred = mean_model(t_phys)
        mean_loss = torch.nn.functional.mse_loss(pred, z0)
        mean_opt.zero_grad(); mean_loss.backward(); mean_opt.step()
        if epoch % 1000 == 0 or epoch == 1 or epoch == args.mean_epochs:
            print(f"[mean epoch {epoch}/{args.mean_epochs}] mse={mean_loss.item():.6f} "
                  f"({time.time()-t0:.1f}s)")
    mean_model.eval()

    # ---- Compute residuals with the FROZEN mean predictor ---------------
    with torch.no_grad():
        mu = mean_model(t_phys)
        resid = z0 - mu                              # (num_frames, latent_dim)
    r_mean = resid.mean(dim=0, keepdim=True)         # ~0 by construction
    r_std = resid.std(dim=0, keepdim=True).clamp_min(1e-6)
    resid_norm = (resid - r_mean) / r_std
    print(f"[resid] residual std per-dim (mean over dims) = {r_std.mean().item():.4f} "
          f"(small => mean predictor captured most of the signal)")

    # ---- Phase 2: diffusion over the (normalized) residuals -------------
    model = LatentDenoiser(latent_dim=latent_dim).cuda()
    print(f"[diff] LatentDenoiser {count_params(model):,} params")
    ema_model = LatentDenoiser(latent_dim=latent_dim).cuda()
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)

    diffusion = GaussianDiffusion(T=1000, device="cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    t1 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(num_frames, device="cuda")
        losses = []
        for start in range(0, num_frames, args.batch_size):
            idx = perm[start:start + args.batch_size]
            loss = diffusion.training_loss(model, resid_norm[idx], t_phys[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.mul_(args.ema_decay).add_(p, alpha=1 - args.ema_decay)

        if epoch % 1000 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"[diff epoch {epoch}/{args.epochs}] loss={sum(losses)/len(losses):.5f} "
                  f"({time.time()-t1:.1f}s)")

        if epoch % 2000 == 0 or epoch == args.epochs:
            torch.save({
                "mean_state_dict": mean_model.state_dict(),
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "r_mean": r_mean.cpu(),
                "r_std": r_std.cpu(),
                "latent_dim": latent_dim,
                "epoch": epoch,
                "stage_a_ckpt": args.stage_a_ckpt,
                "case": args.case,
                "protocol": protocol_name,
                "train_frame_indices": stage_a_train_indices,
                "eval_frame_indices": eval_indices,
                "train_times": t_phys.cpu().tolist(),
                "eval_times": t_phys_all[eval_indices].cpu().tolist(),
                "full_num_frames": full_num_frames,
                "num_train_frames": num_frames,
                "seed": args.seed,
                "hybrid": True,
            }, os.path.join(out_dir, "ckpt_latest.pth"))
            print(f"[checkpoint] saved -> {out_dir}/ckpt_latest.pth")

    print(f"[done] total time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
