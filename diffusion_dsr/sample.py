#!/usr/bin/env python3
"""
Stage C: sample a per-frame latent via the Stage B diffusion model at an
exact trajectory frame, decode it through Stage A's shared DSR-style decoder,
extract a mesh via marching cubes, and evaluate against that exact ground-
truth frame using DSR's own metrics.

Usage:
    conda activate graphmae
    cd diffusion_dsr
    python sample.py --stage-a-ckpt exp/abeta_150_perframe_lat32/checkpoints/ckpt_latest.pth \
                      --stage-b-ckpt exp/abeta_150_diffusion_lat32/ckpt_latest.pth \
                      --frame-index 75 --num-samples 4 --run-name frame75_ensemble
"""
import argparse
import os
import sys

import numpy as np
import torch
import trimesh
from pyhocon import ConfigFactory

DSR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DSR")
sys.path.insert(0, DSR_DIR)
from lib.models.network import ImplicitNet  # noqa: E402
from lib.plots import get_mesh  # noqa: E402
from lib.metrics import compute_iou, compute_trimesh_chamfer, compute_normal_consistency  # noqa: E402

from diffusion import GaussianDiffusion
from model import LatentDenoiser, MeanPredictor
from protocol import ground_truth_path, protocol_times, seed_everything, validate_indices  # noqa: E402

RECON_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reconstructions")


def load_stage_a_decoder(conf_path, case, stage_a_ckpt):
    conf_text = open(conf_path).read().replace("CASE_NAME", case)
    conf = ConfigFactory.parse_string(conf_text)
    ckpt = torch.load(stage_a_ckpt, map_location="cpu", weights_only=False)
    latent_size = ckpt["latent_size"]
    d_in = conf.get_int("dataset.d_in")
    net = ImplicitNet(d_in=(d_in + latent_size + 1), **conf.get_config("model.network")).cuda()
    net = torch.nn.DataParallel(net)
    state_dict = ckpt["model_state_dict"]
    net.load_state_dict(state_dict)
    net.eval()
    return net, latent_size, conf


def load_stage_b_model(stage_b_ckpt, latent_size):
    """v2 hybrid: returns (residual_denoiser, mean_predictor, r_mean, r_std)."""
    ckpt = torch.load(stage_b_ckpt, map_location="cpu", weights_only=False)
    assert ckpt.get("hybrid"), (
        "This sample.py expects a v2 hybrid Stage-B checkpoint (mean + residual). "
        "For a v1 checkpoint, use the v1 sample.py."
    )
    model = LatentDenoiser(latent_dim=latent_size).cuda()
    model.load_state_dict(ckpt["ema_state_dict"])
    model.eval()
    mean_model = MeanPredictor(latent_dim=latent_size).cuda()
    mean_model.load_state_dict(ckpt["mean_state_dict"])
    mean_model.eval()
    r_mean = ckpt["r_mean"].cuda()
    r_std = ckpt["r_std"].cuda()
    return model, mean_model, r_mean, r_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a-ckpt", type=str, required=True)
    ap.add_argument("--stage-b-ckpt", type=str, required=True)
    ap.add_argument("--conf", type=str, default=os.path.join(DSR_DIR, "confs", "abeta.conf"))
    ap.add_argument("--case", type=str, default="abeta_150",
                    help="Full trajectory case used by Stage A/B.")
    ap.add_argument("--gt-case", type=str, default="abeta_30_gt",
                    help="Ground-truth case containing traj_<index>.obj files.")
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--frame-index", type=int,
                        help="Exact original trajectory frame index. Preferred for protocol evaluation.")
    target.add_argument("--t", type=float,
                        help="Legacy direct physical time query; no exact-frame protocol target is implied.")
    ap.add_argument("--num-samples", type=int, default=1, help=">1 for an ensemble/uncertainty check")
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--ddim-steps", type=int, default=75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--skip-eval", action="store_true", help="skip GT comparison (e.g. for out-of-range t)")
    args = ap.parse_args()
    seed_everything(args.seed)

    stage_a_meta = torch.load(args.stage_a_ckpt, map_location="cpu", weights_only=False)
    stage_b_meta = torch.load(args.stage_b_ckpt, map_location="cpu", weights_only=False)
    if args.frame_index is not None:
        if stage_a_meta.get("protocol") != "dsr_abeta_every5_train":
            raise RuntimeError("exact-frame evaluation requires a corrected Stage-A checkpoint")
        if stage_b_meta.get("protocol") != "dsr_abeta_every5_train":
            raise RuntimeError("exact-frame evaluation requires a corrected Stage-B checkpoint")
    train_indices = stage_a_meta.get("train_frame_indices")
    if train_indices is None:
        raise RuntimeError("Stage-A checkpoint has no protocol train-frame metadata")
    all_times = np.asarray(protocol_times(int(stage_a_meta["full_num_frames"]), train_indices), dtype=np.float32)
    if args.frame_index is not None:
        validate_indices([args.frame_index], len(all_times), "frame_index")
        args.t = float(all_times[args.frame_index])
        target_label = f"frame_{args.frame_index:03d}"
        exact_gt_path = ground_truth_path(DSR_DIR, args.gt_case, args.frame_index)
    else:
        target_label = f"t{args.t:.4f}"
        exact_gt_path = None

    decoder, latent_size, conf = load_stage_a_decoder(args.conf, args.case, args.stage_a_ckpt)
    denoiser, mean_model, r_mean, r_std = load_stage_b_model(args.stage_b_ckpt, latent_size)
    diffusion = GaussianDiffusion(T=1000, device="cuda")

    run_name = args.run_name or f"{target_label}_n{args.num_samples}"
    out_dir = os.path.join(RECON_ROOT, f"DiffusionDSR_{args.case}_{run_name}")
    obj_dir = os.path.join(out_dir, "obj")
    os.makedirs(obj_dir, exist_ok=True)

    # v2 hybrid: z = mu(t) [deterministic mean] + residual [diffusion-sampled].
    t_batch = torch.full((args.num_samples,), args.t, device="cuda")
    with torch.no_grad():
        mu = mean_model(t_batch)                                              # (n, latent)
    resid_norm = diffusion.ddim_sample(denoiser, (args.num_samples, latent_size),
                                        t_batch, num_steps=args.ddim_steps)
    resid = resid_norm * r_std + r_mean
    z = mu + resid
    print(f"[sample] drew {args.num_samples} latent(s) at t={args.t}: mu(t) + "
          f"{args.ddim_steps}-step DDIM residual (mean |resid|={resid.abs().mean().item():.4f})")

    scale = conf.get_float("dataset.s")
    translate = conf.get_float("dataset.t")

    mesh_paths = []
    normalized_meshes = []  # kept purely for compute_iou, see note below
    for i in range(args.num_samples):
        latent_i = z[i:i + 1]
        t_i = torch.tensor([args.t], device="cuda")
        shapename = f"sample_{i:02d}"
        try:
            get_mesh(path=obj_dir, epoch=0, decoder=decoder, latent=latent_i, t=t_i,
                      shapename=shapename, resolution=args.resolution, connected=False)
        except AttributeError:
            # get_mesh's SDF grid never crossed zero (no surface to extract) --
            # happens with an undertrained decoder/denoiser; skip, don't crash.
            print(f"[sample {i}] WARNING: no zero-crossing in SDF grid at this "
                  f"(latent, t) -- skipping (likely an undertrained checkpoint)")
            continue
        mesh_path = os.path.join(obj_dir, f"igr_epoch_0_{shapename}.obj")
        # get_mesh writes vertices in normalized [-1,1]^3 grid space. Keep an
        # in-memory copy of that BEFORE rescaling -- compute_iou (DSR's
        # lib/metrics.py) builds its own evaluation grid hardcoded to
        # [-1,1]^3 (see define_grid_3d), so it only gives a meaningful
        # number when fed meshes that are already normalized into that box.
        # Chamfer/NC don't have this problem (plain point-to-point distance,
        # correct in any consistent frame), so for those we still rescale to
        # real physical coordinates exactly as DSR's own runner_igr_t.py
        # reconstruction mode does.
        normalized_meshes.append(trimesh.load(mesh_path, process=False, force="mesh", skip_materials=True))
        rescaled = trimesh.load(mesh_path, process=False, force="mesh", skip_materials=True)
        rescaled.vertices = ((rescaled.vertices + 1.0) * scale) / 2.0 + translate
        rescaled.export(mesh_path, "obj")
        mesh_paths.append(mesh_path)
        print(f"[sample {i}] mesh -> {mesh_path}")

    if not mesh_paths:
        print("[done] no valid meshes were extracted (see warnings above) -- "
              "nothing further to evaluate.")
        return

    if not args.skip_eval:
        if exact_gt_path is None:
            raise ValueError("Protocol evaluation requires --frame-index; --t is legacy-only")
        gt_path = exact_gt_path
        gt_t = all_times[args.frame_index]
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"ground-truth frame does not exist: {gt_path}")
        print(f"[eval] exact ground-truth frame: {gt_path} (t={gt_t:.4f})")
        gt_mesh = trimesh.load(gt_path, process=False, force="mesh", skip_materials=True)
        # Normalized GT copy, using the SAME real<->[-1,1]^3 transform as
        # everything else, purely for the compute_iou call (see comment above).
        gt_mesh_normalized = gt_mesh.copy()
        gt_mesh_normalized.vertices = ((gt_mesh.vertices - translate) / scale) * 2.0 - 1.0

        ious, chamfers, ncs = [], [], []
        for mesh_path, pred_mesh_normalized in zip(mesh_paths, normalized_meshes):
            pred_mesh = trimesh.load(mesh_path, process=False, force="mesh", skip_materials=True)
            iou = compute_iou(gt_mesh_normalized, pred_mesh_normalized)
            chamfer = compute_trimesh_chamfer(pred_mesh, gt_mesh)
            nc = compute_normal_consistency(gt_mesh, pred_mesh)
            ious.append(iou); chamfers.append(chamfer); ncs.append(nc)
            print(f"  {os.path.basename(mesh_path)}: IoU={iou:.4f} Chamfer={chamfer:.6f} NC={nc:.4f}")

        print(f"[eval summary] IoU={np.mean(ious):.4f}+-{np.std(ious):.4f}  "
              f"Chamfer={np.mean(chamfers):.6f}+-{np.std(chamfers):.6f}  "
              f"NC={np.mean(ncs):.4f}+-{np.std(ncs):.4f}")

    if args.num_samples > 1:
        # Ensemble/uncertainty check: pairwise Chamfer distance between the
        # sampled meshes themselves (not vs. GT) -- the "does this do
        # something DSR structurally can't" metric.
        meshes = [trimesh.load(p, process=False, force="mesh", skip_materials=True) for p in mesh_paths]
        pairwise = []
        for i in range(len(meshes)):
            for j in range(i + 1, len(meshes)):
                pairwise.append(compute_trimesh_chamfer(meshes[i], meshes[j]))
        print(f"[ensemble] pairwise Chamfer across {args.num_samples} samples at fixed t: "
              f"mean={np.mean(pairwise):.6f} std={np.std(pairwise):.6f} "
              f"(nonzero => real sample diversity, not mode collapse)")

    print(f"[done] outputs in {out_dir}")


if __name__ == "__main__":
    main()
