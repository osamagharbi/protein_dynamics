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
    cd MAMBA-DSR
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


def collate_frames(batch):
    """Keep variable-length frame point clouds unpadded.

    A normal DataLoader collate function tries to stack the point clouds, but
    molecular surfaces do not necessarily contain the same number of points.
    Returning the samples as a list lets the training loop retain each frame's
    boundary and concatenate only after per-frame sampling is complete.
    """
    return batch


def build_network(conf, d_in, latent_size):
    net = ImplicitNet(d_in=(d_in + latent_size + 1), **conf.get_config("model.network")).cuda()
    # DataParallel adds scatter/replicate/gather overhead and provides no
    # benefit when the launcher exposes one GPU.  Keep it for multi-GPU runs
    # so existing distributed-style usage remains available.
    return net if torch.cuda.device_count() == 1 else torch.nn.DataParallel(net)


def add_latent(lat_vecs, points, frame_ids):
    # ``frame_ids`` has one entry per point.  This supports both the original
    # one-frame path and the accelerated variable-length multi-frame path.
    batch_vecs = lat_vecs(frame_ids.cuda())
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
    ap.add_argument("--full-num-frames", type=int, default=None,
                    help="Total number of frames in the source trajectory when the input "
                         "data directory is a train-only view. The checkpoint keeps original "
                         "frame indices while Stage A loads only the selected meshes.")
    ap.add_argument("--latent-size", type=int, default=8,
                     help="Per-frame latent dim. v2 default 8 (was 32): the trajectory is "
                          "intrinsically a ~1-parameter family in t, so a small latent is far "
                          "easier for Stage B's Mamba predictor to model and the smoothness prior + "
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
    ap.add_argument("--frame-batch-size", type=int, default=1,
                    help="Number of frames optimized together. All selected frames and all "
                         "their points are retained; values >1 change only the optimization "
                         "batching and are useful for acceleration.")
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
    loaded_num_frames = len(raw_dataset)
    explicit_indices = parse_indices(args.train_indices)
    full_num_frames = args.full_num_frames or loaded_num_frames
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
    if loaded_num_frames == full_num_frames:
        # Full trajectory input: select the original rows directly.
        raw_indices = train_frame_indices
        time_values = all_protocol_times
    else:
        # Train-only view: rows are local, while their physical frame identity
        # is carried by train_frame_indices.
        if loaded_num_frames != len(train_frame_indices):
            raise ValueError(
                "a train-only Stage-A view must contain exactly one mesh per "
                f"selected index (loaded={loaded_num_frames}, indices={len(train_frame_indices)})"
            )
        raw_indices = list(range(loaded_num_frames))
        time_values = [all_protocol_times[index] for index in train_frame_indices]
    dataset = FrameIndexedDataset(
        raw_dataset,
        raw_indices,
        time_values,
    )
    num_frames = len(dataset)
    print(f"[data] full trajectory has {full_num_frames} frames for case={args.case}")
    print(f"[split] Stage A trains on {num_frames} frames: {train_frame_indices}")

    if args.frame_batch_size < 1:
        raise ValueError("--frame-batch-size must be at least 1")
    dataloader = DataLoader(
        dataset,
        batch_size=args.frame_batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False,
        collate_fn=collate_frames,
        pin_memory=True,
    )

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

        # Aggregate metrics over the whole epoch instead of reporting only the
        # final frame/batch when frame-batching is enabled.
        epoch_values = {key: [] for key in (
            "mnfld_loss", "grad_loss", "latent_loss", "normals_loss",
            "smooth_loss", "consist_loss", "loss",
        )}

        for batch in epoch_loader:
            # Each item remains independent through sampling.  In particular,
            # off-surface samples from one molecular frame are never generated
            # around points from another frame.
            batch_frame_ids = torch.tensor(
                [int(item[3]) for item in batch], dtype=torch.long, device="cuda"
            )
            mnfld_points_parts = []
            mnfld_normals_parts = []
            mnfld_time_parts = []
            mnfld_id_parts = []
            mnfld_slot_parts = []
            nonmnfld_points_parts = []
            nonmnfld_time_parts = []
            nonmnfld_id_parts = []
            nonmnfld_slot_parts = []

            for slot, (data, local_sigmas, t, frame_id) in enumerate(batch):
                data = data.cuda(non_blocking=True)
                local_sigmas = local_sigmas.cuda(non_blocking=True)
                indices = torch.randperm(data.shape[0], device="cuda")
                cur_data = data[indices]
                mnfld_pnts_i = cur_data[:, :d_in]
                normals_i = cur_data[:, d_in:]
                mnfld_sigma_i = local_sigmas[indices]

                sampler = Sampler.get_sampler(sampler_type)(global_sigma, mnfld_sigma_i)
                nonmnfld_pnts_i = sampler.get_points(
                    mnfld_pnts_i.unsqueeze(0), mnfld_sigma_i.unsqueeze(0)
                ).squeeze(0)

                mnfld_points_parts.append(mnfld_pnts_i)
                mnfld_normals_parts.append(normals_i)
                mnfld_time_parts.append(torch.full(
                    (mnfld_pnts_i.shape[0],), float(t), device="cuda", dtype=torch.float32
                ))
                mnfld_id_parts.append(torch.full(
                    (mnfld_pnts_i.shape[0],), int(frame_id), device="cuda", dtype=torch.long
                ))
                mnfld_slot_parts.append(torch.full(
                    (mnfld_pnts_i.shape[0],), slot, device="cuda", dtype=torch.long
                ))
                nonmnfld_points_parts.append(nonmnfld_pnts_i)
                nonmnfld_time_parts.append(torch.full(
                    (nonmnfld_pnts_i.shape[0],), float(t), device="cuda", dtype=torch.float32
                ))
                nonmnfld_id_parts.append(torch.full(
                    (nonmnfld_pnts_i.shape[0],), int(frame_id), device="cuda", dtype=torch.long
                ))
                nonmnfld_slot_parts.append(torch.full(
                    (nonmnfld_pnts_i.shape[0],), slot, device="cuda", dtype=torch.long
                ))

            mnfld_pnts = torch.cat(mnfld_points_parts, dim=0)
            normals = torch.cat(mnfld_normals_parts, dim=0)
            mnfld_times = torch.cat(mnfld_time_parts, dim=0)
            mnfld_frame_ids = torch.cat(mnfld_id_parts, dim=0)
            mnfld_slots = torch.cat(mnfld_slot_parts, dim=0)
            nonmnfld_pnts = torch.cat(nonmnfld_points_parts, dim=0)
            nonmnfld_times = torch.cat(nonmnfld_time_parts, dim=0)
            nonmnfld_frame_ids = torch.cat(nonmnfld_id_parts, dim=0)
            nonmnfld_slots = torch.cat(nonmnfld_slot_parts, dim=0)

            # raw (pre-latent) [xyz, t] tensors -- kept so the consistency term
            # can re-prepend a perturbed latent to the exact same points.
            mnfld_raw = torch.cat((mnfld_pnts, mnfld_times.unsqueeze(-1)), dim=1)
            nonmnfld_raw = torch.cat((nonmnfld_pnts, nonmnfld_times.unsqueeze(-1)), dim=1)

            mnfld_pnts_t = add_latent(lat_vecs, mnfld_raw, mnfld_frame_ids)
            nonmnfld_pnts_t = add_latent(lat_vecs, nonmnfld_raw, nonmnfld_frame_ids)

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

            lat_loss = latent_size_reg(lat_vecs, batch_frame_ids)
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
                z = lat_vecs(batch_frame_ids)                       # (frames, latent)
                eps = torch.randn_like(z) * args.consist_sigma
                zp = z + eps
                mnfld_p = torch.cat([zp[mnfld_slots], mnfld_raw], dim=1)
                nonmnfld_p = torch.cat([zp[nonmnfld_slots], nonmnfld_raw], dim=1)
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

            epoch_values["mnfld_loss"].append(mnfld_loss.item())
            epoch_values["grad_loss"].append(grad_loss.item())
            epoch_values["latent_loss"].append(lat_loss.item())
            epoch_values["normals_loss"].append(normals_loss.item())
            epoch_values["smooth_loss"].append(smooth_loss.item())
            epoch_values["consist_loss"].append(consist_loss.item())
            epoch_values["loss"].append(loss.item())

            if args.progress:
                epoch_loader.set_postfix(loss=f"{loss.item():.4f}")

        loss_log["epoch"].append(epoch)
        epoch_means = {
            key: float(np.mean(values)) for key, values in epoch_values.items()
        }
        for key, value in epoch_means.items():
            loss_log[key].append(value)

        if args.log_every > 0 and (epoch % args.log_every == 0 or epoch == 1):
            elapsed = time.time() - t0
            print(f"[epoch {epoch}/{args.end_iter}] loss={epoch_means['loss']:.4f} "
                  f"mnfld={epoch_means['mnfld_loss']:.4f} grad={epoch_means['grad_loss']:.4f} "
                  f"normals={epoch_means['normals_loss']:.4f} latent={epoch_means['latent_loss']:.4f} "
                  f"smooth={epoch_means['smooth_loss']:.4f} consist={epoch_means['consist_loss']:.4f} "
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
