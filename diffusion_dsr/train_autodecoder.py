#!/usr/bin/env python3
"""
Stage A: retrofit DSR to a per-FRAME auto-decoder latent instead of DSR's
original per-SEQUENCE latent.  Under the frozen-DSR comparison protocol, only
the DSR training frames are loaded, so no evaluation-frame geometry leaks into
Stage A.

Reuses DSR's own decoder architecture, sampler, and Eikonal-regularized loss
unchanged (imported from DSR/lib/) -- the only surgical change is what index
gets passed into the latent embedding lookup: frame index instead of
sequence index. DSR/ itself is never modified.

Usage:
    conda activate graphmae
    cd diffusion_dsr
    python train_autodecoder.py --case abeta_150 --protocol dsr_abeta \
        --latent-size 8 --end-iter 800 --run-name abeta_dsr_protocol_v2
"""
import argparse
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
from pyhocon import ConfigFactory
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset

DSR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DSR")
sys.path.insert(0, DSR_DIR)

from lib.dataset import RawDataset  # noqa: E402
from lib.models.network import ImplicitNet, gradient  # noqa: E402
from lib.models.sample import Sampler  # noqa: E402
from lib.utils import get_learning_rate_schedules, get_spec_with_default  # noqa: E402
from protocol import (  # noqa: E402
    DSR_TRAIN_INTERVAL,
    PROTOCOL_NAME,
    dsr_train_indices,
    parse_indices,
    protocol_times,
    seed_everything,
    validate_indices,
)

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp")


class FrameIndexedDataset(Dataset):
    """Wraps DSR's RawDataset to expose the flat frame index (0..N-1) in
    place of the (file_path, sequence_id) fields, so every frame gets its
    own latent instead of every sequence sharing one."""

    def __init__(self, raw_dataset, frame_indices=None, time_values=None):
        self.raw = raw_dataset
        self.frame_indices = list(range(len(raw_dataset))) if frame_indices is None else list(frame_indices)
        self.time_values = time_values

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        raw_idx = self.frame_indices[idx]
        data, local_sigmas, t, _file_path, _sequence_id = self.raw[raw_idx]
        if self.time_values is not None:
            t = self.time_values[raw_idx]
        # ``protocol_times`` returns ordinary Python floats.  PyTorch's
        # default collate therefore promotes them to float64, while the
        # decoder and point tensors are float32.  Keep the time coordinate in
        # the same dtype as the network inputs at this dataset boundary.
        t = torch.as_tensor(t, dtype=torch.float32)
        return data, local_sigmas, t, idx


def build_network(conf, d_in, latent_size):
    net = ImplicitNet(d_in=(d_in + latent_size + 1), **conf.get_config("model.network")).cuda()
    return torch.nn.DataParallel(net)


def add_latent(lat_vecs, points, frame_ids):
    num_points = points.shape[0]
    batch_vecs = lat_vecs(frame_ids.unsqueeze(-1).repeat(1, num_points).view(-1).cuda())
    return torch.cat([batch_vecs, points], 1)


def latent_size_reg(lat_vecs, frame_ids):
    latents = lat_vecs(frame_ids)
    return latents.norm(dim=1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=str, default=os.path.join(DSR_DIR, "confs", "abeta.conf"),
                     help="Reused only for network architecture + loss hyperparameters, "
                          "not for dataset.num_sequences or latent_vector.latent_size.")
    ap.add_argument("--case", type=str, default="abeta_150",
                    help="Full trajectory case containing all frames. For the frozen Aβ "
                         "comparison use abeta_150, whose obj directory points to the 149 GT frames.")
    ap.add_argument("--protocol", type=str, choices=["none", "dsr_abeta"], default="dsr_abeta",
                    help="dsr_abeta trains only on frames 0,5,...,145. Use none only for exploratory runs.")
    ap.add_argument("--train-indices", type=str, default=None,
                    help="Optional comma-separated original frame indices; overrides --protocol.")
    ap.add_argument("--latent-size", type=int, default=8,
                     help="Per-frame latent dim. v2 default 8 (was 32): the trajectory is "
                          "intrinsically a ~1-parameter family in t, so a small latent is far "
                          "easier for Stage B's diffusion to model and the smoothness prior + "
                          "decoder can still fit it (the oracle had a large margin to spare).")
    ap.add_argument("--smooth-lambda", type=float, default=1.0,
                     help="v2: weight of the temporal-smoothness loss on the latent table. "
                          "Penalizes ||z(t_{i+1}) - z(t_i)||^2 so the latent trajectory becomes "
                          "a smooth curve in t instead of scattered noise -- this is what makes "
                          "the space learnable by a t-conditioned generator. Set 0 to reproduce v1.")
    ap.add_argument("--consist-lambda", type=float, default=1.0,
                     help="v3: weight of the decoder latent-consistency (Lipschitz) penalty. "
                          "Decodes a perturbed latent z+eps on the same points and pulls its SDF "
                          "toward the clean decode's -- so small latent errors (like the mean "
                          "predictor's ~0.05) cause small surface errors. Attacks the isolated "
                          "bottleneck: decoder over-sensitivity to latent error. Set 0 to disable.")
    ap.add_argument("--consist-sigma", type=float, default=0.03,
                     help="v3: per-dim std of the latent perturbation for the consistency penalty. "
                          "~matched to (slightly above) the observed mean-predictor latent error so "
                          "the decoder is trained flat over exactly the ball the predictor lands in.")
    ap.add_argument("--end-iter", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=20,
                    help="Print the epoch summary every N epochs. Use 1 to print every epoch; "
                         "use 0 to disable epoch summaries.")
    ap.add_argument("--progress", action="store_true",
                    help="Display a live progress bar for the frames processed in each epoch.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--init-from", type=str, default=None,
                     help="v3.1: path to an existing Stage-A checkpoint to load decoder + latents "
                          "from and FINE-TUNE (instead of training from scratch). Enables cheap "
                          "~80-epoch fine-tunes of a converged decoder with a stronger consistency "
                          "penalty, evaluated at full decoder quality (~10x faster than 3h retrains).")
    ap.add_argument("--ft-lr", type=float, default=1e-5,
                     help="Flat learning rate used when --init-from is set (small, since the "
                          "decoder is already converged and we only want to reshape its latent "
                          "sensitivity without forgetting the fit).")
    ap.add_argument("--consist-surface", action="store_true",
                     help="v3.1: use the SURFACE-TARGETED consistency penalty -- enforce the "
                          "perturbed latent still puts the manifold points at f=0 (|f(mnfld,z+eps)|) "
                          "plus preserves the off-surface field. Directly protects the zero-level-set "
                          "(what IoU/Chamfer actually measure), unlike the mean-squared-SDF form "
                          "which a tiny value error can pass while the surface still shifts.")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed_everything(args.seed)
    run_name = args.run_name or f"{args.case}_perframe_lat{args.latent_size}"
    out_dir = os.path.join(OUT_ROOT, run_name)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    conf_text = open(args.conf).read().replace("CASE_NAME", args.case)
    conf = ConfigFactory.parse_string(conf_text)
    # dataset.data_dir in the conf is relative to DSR/, so resolve it there.
    data_dir = os.path.join(DSR_DIR, conf.get_string("dataset.data_dir"))

    raw_dataset = RawDataset(
        data_dir=data_dir,
        extension=conf.get_string("dataset.extension"),
        d_in=conf.get_int("dataset.d_in"),
        with_normal=conf.get_bool("dataset.with_normal"),
        s=conf.get_float("dataset.s"),
        t=conf.get_float("dataset.t"),
    )
    full_num_frames = len(raw_dataset)
    explicit_indices = parse_indices(args.train_indices)
    if explicit_indices is not None:
        train_frame_indices = explicit_indices
        protocol_name = "explicit_indices"
    elif args.protocol == "dsr_abeta":
        train_frame_indices = dsr_train_indices(full_num_frames, DSR_TRAIN_INTERVAL)
        protocol_name = PROTOCOL_NAME
    else:
        train_frame_indices = list(range(full_num_frames))
        protocol_name = "none"
    validate_indices(train_frame_indices, full_num_frames, "train_frame_indices")
    all_protocol_times = protocol_times(full_num_frames, train_frame_indices)
    dataset = FrameIndexedDataset(raw_dataset, train_frame_indices, all_protocol_times)
    num_frames = len(dataset)
    print(f"[data] full trajectory has {full_num_frames} frames for case={args.case}")
    print(f"[split] Stage A trains on {num_frames} frames: {train_frame_indices}")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=False)

    d_in = conf.get_int("dataset.d_in")
    latent_size = args.latent_size
    grad_lambda = conf.get_float("model.loss.lambda")
    normals_lambda = conf.get_float("model.loss.normals_lambda")
    latent_lambda = conf.get_float("model.loss.latent_lambda")
    global_sigma = conf.get_float("model.sampler.properties.global_sigma")
    with_normals = normals_lambda > 0

    network = build_network(conf, d_in, latent_size)
    print(network)

    lat_vecs = torch.nn.Embedding(num_frames, latent_size).cuda()
    torch.nn.init.normal_(
        lat_vecs.weight.data, 0.0,
        get_spec_with_default(conf["model.latent_vector"], "code_init_std_dev", 1.0) / math.sqrt(latent_size),
    )
    lat_vecs.requires_grad_()

    # v3.1 fine-tune: load a converged decoder + latents to reshape sensitivity.
    if args.init_from:
        init_ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        assert init_ck["latent_size"] == latent_size, "latent-size mismatch with --init-from ckpt"
        network.load_state_dict(init_ck["model_state_dict"])
        lat_vecs.load_state_dict(init_ck["latent_codes"])
        if init_ck.get("train_frame_indices") != train_frame_indices:
            raise ValueError("--init-from checkpoint uses a different train-frame split")
        print(f"[init] loaded decoder + {init_ck['num_frames']} latents from {args.init_from}")

    lr_schedules = get_learning_rate_schedules(conf)
    if args.init_from:
        # flat small LR for fine-tuning (bypass the from-scratch decay schedule)
        optimizer = torch.optim.Adam([
            {"params": network.parameters(), "lr": args.ft_lr},
            {"params": lat_vecs.parameters(), "lr": args.ft_lr},
        ])
    else:
        optimizer = torch.optim.Adam([
            {"params": network.parameters(), "lr": lr_schedules[0].get_learning_rate(0)},
            {"params": lat_vecs.parameters(), "lr": lr_schedules[1].get_learning_rate(0)},
        ])

    sampler_type = conf.get_string("model.sampler.sampler_type")

    loss_log = {"epoch": [], "mnfld_loss": [], "grad_loss": [], "latent_loss": [], "normals_loss": [], "smooth_loss": [], "consist_loss": [], "loss": []}
    t0 = time.time()
    for epoch in range(1, args.end_iter + 1):
        network.train()
        if not args.init_from:  # keep flat ft-lr when fine-tuning
            for i, sched in enumerate(lr_schedules):
                optimizer.param_groups[i]["lr"] = sched.get_learning_rate(epoch)

        epoch_loader = dataloader
        if args.progress:
            epoch_loader = tqdm(
                dataloader,
                total=len(dataloader),
                desc=f"epoch {epoch}/{args.end_iter}",
                unit="frame",
                leave=False,
            )

        for data, local_sigmas, t, frame_id in epoch_loader:
            data = data.squeeze(0).cuda()
            indices = torch.tensor(np.random.choice(data.shape[0], data.shape[0], False))
            cur_data = data[indices]
            mnfld_pnts = cur_data[:, :d_in]
            normals = cur_data[:, d_in:]
            mnfld_sigma = local_sigmas.squeeze(0)[indices].cuda()

            sampler = Sampler.get_sampler(sampler_type)(global_sigma, mnfld_sigma)
            nonmnfld_pnts = sampler.get_points(mnfld_pnts.unsqueeze(0), mnfld_sigma.unsqueeze(0)).squeeze(0)

            # raw (pre-latent) [xyz, t] tensors -- kept so the consistency term
            # can re-prepend a *perturbed* latent to the exact same points.
            mnfld_raw = torch.cat((mnfld_pnts, t.unsqueeze(-1).repeat(1, mnfld_pnts.shape[0]).view(-1, 1).cuda()), dim=1)
            nonmnfld_raw = torch.cat((nonmnfld_pnts, t.unsqueeze(-1).repeat(1, nonmnfld_pnts.shape[0]).view(-1, 1).cuda()), dim=1)

            mnfld_pnts_t = add_latent(lat_vecs, mnfld_raw, frame_id)
            nonmnfld_pnts_t = add_latent(lat_vecs, nonmnfld_raw, frame_id)

            mnfld_pnts.requires_grad_()
            nonmnfld_pnts.requires_grad_()

            mnfld_pred = network(mnfld_pnts_t)
            nonmnfld_pred = network(nonmnfld_pnts_t)

            mnfld_grad = gradient(mnfld_pnts_t, mnfld_pred)
            nonmnfld_grad = gradient(nonmnfld_pnts_t, nonmnfld_pred)

            mnfld_loss = mnfld_pred.abs().mean()
            grad_loss = ((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
            loss = mnfld_loss + grad_lambda * grad_loss

            if with_normals:
                normals_loss = ((mnfld_grad - normals).abs()).norm(2, dim=1).mean()
                loss = loss + normals_lambda * normals_loss
            else:
                normals_loss = torch.zeros(1)

            lat_loss = latent_size_reg(lat_vecs, frame_id.cuda())
            loss = loss + latent_lambda * lat_loss

            # v2: temporal-smoothness prior on the latent trajectory. Rows of
            # the embedding table follow the ordered DSR training frames, so
            # consecutive rows are consecutive observed training times. Penalizing
            # their squared difference turns the latent trajectory from scattered
            # noise into a smooth curve in t.
            # (Single-sequence case: no cross-sequence boundary to worry about.)
            if args.smooth_lambda > 0:
                smooth_loss = ((lat_vecs.weight[1:] - lat_vecs.weight[:-1]) ** 2).sum(-1).mean()
                loss = loss + args.smooth_lambda * smooth_loss
            else:
                smooth_loss = torch.zeros(1)

            # v3: decoder latent-consistency (Lipschitz) penalty. Re-decode the
            # same points with a perturbed latent z+eps and pull its SDF toward
            # the clean decode (detached) -- teaching the decoder to be flat in
            # the latent within radius ~consist_sigma, so the mean predictor's
            # small latent error at inference produces a small surface error.
            if args.consist_lambda > 0:
                z = lat_vecs(frame_id.cuda())                       # (1, latent)
                eps = torch.randn_like(z) * args.consist_sigma
                zp = (z + eps)
                mnfld_p = torch.cat([zp.repeat(mnfld_raw.shape[0], 1), mnfld_raw], dim=1)
                nonmnfld_p = torch.cat([zp.repeat(nonmnfld_raw.shape[0], 1), nonmnfld_raw], dim=1)
                mnfld_pred_p = network(mnfld_p)
                nonmnfld_pred_p = network(nonmnfld_p)
                if args.consist_surface:
                    # surface-targeted: perturbed latent must still put the
                    # manifold points at f=0 (protects the zero-level-set that
                    # IoU/Chamfer measure), + preserve the off-surface field.
                    consist_loss = mnfld_pred_p.abs().mean() + \
                                   ((nonmnfld_pred_p - nonmnfld_pred.detach()) ** 2).mean()
                else:
                    # v3 original: mean-squared SDF match everywhere (too weak --
                    # a tiny value error still shifts the surface).
                    consist_loss = ((mnfld_pred_p - mnfld_pred.detach()) ** 2).mean() + \
                                   ((nonmnfld_pred_p - nonmnfld_pred.detach()) ** 2).mean()
                loss = loss + args.consist_lambda * consist_loss
            else:
                consist_loss = torch.zeros(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.progress:
                epoch_loader.set_postfix(loss=f"{loss.item():.4f}")

        loss_log["epoch"].append(epoch)
        loss_log["mnfld_loss"].append(mnfld_loss.item())
        loss_log["grad_loss"].append(grad_loss.item())
        loss_log["latent_loss"].append(lat_loss.item())
        loss_log["normals_loss"].append(normals_loss.item())
        loss_log["smooth_loss"].append(smooth_loss.item())
        loss_log["consist_loss"].append(consist_loss.item())
        loss_log["loss"].append(loss.item())

        if args.log_every > 0 and (epoch % args.log_every == 0 or epoch == 1):
            elapsed = time.time() - t0
            print(f"[epoch {epoch}/{args.end_iter}] loss={loss.item():.4f} "
                  f"mnfld={mnfld_loss.item():.4f} grad={grad_loss.item():.4f} "
                  f"normals={normals_loss.item():.4f} latent={lat_loss.item():.4f} "
                  f"smooth={smooth_loss.item():.4f} consist={consist_loss.item():.4f} "
                  f"({elapsed:.1f}s elapsed)")

        if epoch % 200 == 0 or epoch == args.end_iter:
            ckpt = {
                "model_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "latent_codes": lat_vecs.state_dict(),
                "epoch": epoch,
                "num_frames": num_frames,
                "full_num_frames": full_num_frames,
                "latent_size": latent_size,
                "case": args.case,
                "protocol": protocol_name,
                "train_frame_indices": train_frame_indices,
                "eval_frame_indices": [i for i in range(full_num_frames) if i not in set(train_frame_indices)],
                "normalization_scale": raw_dataset.s,
                "normalization_translation": raw_dataset.t,
                "protocol_times": all_protocol_times,
                "seed": args.seed,
            }
            torch.save(ckpt, os.path.join(ckpt_dir, "ckpt_latest.pth"))
            pickle.dump(loss_log, open(os.path.join(out_dir, "loss_log.pkl"), "wb"))
            print(f"[checkpoint] saved at epoch {epoch} -> {ckpt_dir}/ckpt_latest.pth")

    print(f"[done] total time {time.time()-t0:.1f}s, {num_frames} per-frame latents in "
          f"{ckpt_dir}/ckpt_latest.pth")


if __name__ == "__main__":
    main()
