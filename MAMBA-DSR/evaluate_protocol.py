#!/usr/bin/env python3
"""Evaluate MAMBA-DSR with an explicit frame list and DSR metrics.

    The evaluator never infers a paper split from the number of files. Pass an
    explicit frame list for a paper comparison. The ``normalized`` metric space
    matches the official DSR evaluator; ``physical`` is retained for older
    MAMBA-vs-Diffusion-DSR reports.

Example:
    python evaluate_protocol.py \
      --stage-a-ckpt exp/abeta_dsr_protocol_v2/checkpoints/ckpt_latest.pth \
      --stage-b-ckpt exp/abeta_dsr_protocol_v2/ckpt_latest.pth \
      --run-name abeta_dsr_protocol_v2_eval --num-samples 1
"""

import argparse
import csv
import json
import os
import shutil
import sys

import numpy as np
import torch
import trimesh
from pyhocon import ConfigFactory

DSR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DSR")
sys.path.insert(0, DSR_DIR)
from lib.metrics import (  # noqa: E402
    compute_iou,
    compute_normal_consistency,
    compute_trimesh_chamfer,
)
from lib.plots import get_mesh  # noqa: E402

from protocol import (  # noqa: E402
    DSR_RESOLUTION,
    ground_truth_path,
    parse_indices,
    protocol_times,
    seed_everything,
    validate_indices,
)
from sample import load_stage_a_decoder, load_stage_b_model  # noqa: E402

RECON_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reconstructions")


def evaluate_frame(decoder, mamba_model, latent_mean, latent_std,
                   conf, frame_index, target_t, gt_path, out_dir, num_samples,
                   resolution, metric_space):
    os.makedirs(out_dir, exist_ok=True)
    obj_dir = os.path.join(out_dir, "obj")
    os.makedirs(obj_dir, exist_ok=True)
    t_batch = torch.full((num_samples,), float(target_t), device="cuda")
    with torch.no_grad():
        z = mamba_model(t_batch) * latent_std + latent_mean

    scale = conf.get_float("dataset.s")
    translate = conf.get_float("dataset.t")
    mesh_paths = []
    normalized_meshes = []

    for sample_index in range(num_samples):
        shapename = f"sample_{sample_index:02d}"
        try:
            get_mesh(
                path=obj_dir,
                epoch=0,
                decoder=decoder,
                latent=z[sample_index:sample_index + 1],
                t=torch.tensor([target_t], device="cuda"),
                shapename=shapename,
                resolution=resolution,
                connected=False,
            )
        except AttributeError:
            print(f"[frame {frame_index}] sample {sample_index}: no zero crossing")
            continue

        mesh_path = os.path.join(obj_dir, f"igr_epoch_0_{shapename}.obj")
        normalized = trimesh.load(mesh_path, process=False, force="mesh", skip_materials=True)
        rescaled = normalized.copy()
        rescaled.vertices = ((rescaled.vertices + 1.0) * scale) / 2.0 + translate
        rescaled.export(mesh_path, "obj")
        normalized_meshes.append(normalized)
        mesh_paths.append(mesh_path)

    result = {
        "frame_index": int(frame_index),
        "time": float(target_t),
        "num_samples_requested": int(num_samples),
        "num_samples_valid": len(mesh_paths),
        "resolution": int(resolution),
        "metrics": [],
    }
    if not mesh_paths:
        return result

    gt_mesh = trimesh.load(gt_path, process=False, force="mesh", skip_materials=True)
    gt_normalized = gt_mesh.copy()
    gt_normalized.vertices = ((gt_mesh.vertices - translate) / scale) * 2.0 - 1.0

    for mesh_path, pred_normalized in zip(mesh_paths, normalized_meshes):
        pred_mesh = trimesh.load(mesh_path, process=False, force="mesh", skip_materials=True)
        if metric_space == "normalized":
            metric_pred = pred_normalized
            metric_gt = gt_normalized
        else:
            metric_pred = pred_mesh
            metric_gt = gt_mesh
        result["metrics"].append({
            "sample": os.path.basename(mesh_path),
            "iou": float(compute_iou(gt_normalized, pred_normalized, N=resolution)),
            "chamfer": float(compute_trimesh_chamfer(metric_pred, metric_gt)),
            "normal_consistency": float(compute_normal_consistency(metric_gt, metric_pred)),
        })

    for key in ("iou", "chamfer", "normal_consistency"):
        values = [metric[key] for metric in result["metrics"]]
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_std"] = float(np.std(values))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a-ckpt", required=True)
    ap.add_argument("--stage-b-ckpt", required=True)
    ap.add_argument("--conf", default=os.path.join(DSR_DIR, "confs", "abeta.conf"))
    ap.add_argument("--case", default="abeta_150", help="Full 149-frame trajectory case.")
    ap.add_argument("--gt-case", default="abeta_30_gt")
    ap.add_argument("--frame-indices", default=None,
                    help="Optional comma-separated target indices; defaults to checkpoint eval_frame_indices.")
    ap.add_argument("--include-training-frames", action="store_true",
                    help="Allow requested frames to overlap the training list. Required for "
                         "the DSR paper's training-frame reconstruction table.")
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--resolution", type=int, default=DSR_RESOLUTION)
    ap.add_argument("--metric-space", choices=["normalized", "physical"], default="normalized",
                    help="Official DSR evaluation normalizes both meshes before all metrics. "
                         "Use physical only for legacy MAMBA-vs-Diffusion reports.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="abeta_dsr_protocol_eval")
    ap.add_argument("--keep-reconstructions", action="store_true",
                    help="Keep per-frame predicted OBJ files. By default they are removed "
                         "after metrics are computed to avoid filling the disk.")
    args = ap.parse_args()
    seed_everything(args.seed)

    stage_a_ckpt = torch.load(args.stage_a_ckpt, map_location="cpu", weights_only=False)
    stage_b_ckpt = torch.load(args.stage_b_ckpt, map_location="cpu", weights_only=False)
    if not stage_a_ckpt.get("train_frame_indices"):
        raise RuntimeError("Stage-A checkpoint has no train_frame_indices metadata")
    if not stage_b_ckpt.get("train_frame_indices"):
        raise RuntimeError("Stage-B checkpoint has no train_frame_indices metadata")
    if stage_a_ckpt["train_frame_indices"] != stage_b_ckpt["train_frame_indices"]:
        raise RuntimeError("Stage-A and Stage-B use different training-frame lists")

    decoder, latent_size, conf = load_stage_a_decoder(args.conf, args.case, args.stage_a_ckpt)
    mamba_model, latent_mean, latent_std = load_stage_b_model(args.stage_b_ckpt, latent_size)
    all_times = np.asarray(
        protocol_times(int(stage_b_ckpt["full_num_frames"]), stage_b_ckpt["train_frame_indices"]),
        dtype=np.float32,
    )

    frame_indices = parse_indices(args.frame_indices)
    if frame_indices is None:
        if args.include_training_frames:
            frame_indices = list(range(len(all_times)))
        else:
            frame_indices = list(stage_b_ckpt["eval_frame_indices"])
    validate_indices(frame_indices, len(all_times), "frame_indices")
    train_indices = set(stage_b_ckpt["train_frame_indices"])
    overlap = sorted(set(frame_indices) & train_indices)
    if overlap and not args.include_training_frames:
        raise ValueError(f"evaluation indices overlap the training split: {overlap}")
    if overlap:
        print(f"[warning] including {len(overlap)} training frames; results are diagnostic, not held-out")

    root = os.path.join(RECON_ROOT, f"MAMBA-DSR_{args.case}_{args.run_name}")
    os.makedirs(root, exist_ok=True)
    # Per-frame result cache: a walltime-killed evaluation resumes where it stopped.
    # Entries are reused only for the same Stage-B checkpoint and evaluation settings.
    cache_dir = os.path.join(root, "frame_results")
    os.makedirs(cache_dir, exist_ok=True)
    b_stat = os.stat(args.stage_b_ckpt)
    fingerprint = {"stage_b_ckpt": os.path.abspath(args.stage_b_ckpt), "stage_b_mtime": b_stat.st_mtime,
                   "stage_b_size": b_stat.st_size, "resolution": args.resolution,
                   "metric_space": args.metric_space, "num_samples": args.num_samples, "seed": args.seed}
    results = []
    for frame_index in frame_indices:
        cache_path = os.path.join(cache_dir, f"frame_{frame_index:05d}.json")
        if os.path.isfile(cache_path):
            cached = json.load(open(cache_path))
            if cached.get("fingerprint") == fingerprint:
                results.append(cached["result"])
                continue
        gt_path = ground_truth_path(DSR_DIR, args.gt_case, frame_index)
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"missing exact ground-truth frame: {gt_path}")
        print(f"[eval] frame {frame_index}/{len(all_times) - 1}, t={all_times[frame_index]:.6f}", flush=True)
        result = evaluate_frame(
            decoder, mamba_model, latent_mean, latent_std,
            conf, frame_index, all_times[frame_index], gt_path,
            os.path.join(root, f"frame_{frame_index:03d}"), args.num_samples,
            args.resolution, args.metric_space,
        )
        results.append(result)
        with open(cache_path + ".tmp", "w") as handle:
            json.dump({"fingerprint": fingerprint, "result": result}, handle)
        os.replace(cache_path + ".tmp", cache_path)
        if not args.keep_reconstructions:
            shutil.rmtree(os.path.join(root, f"frame_{frame_index:03d}"), ignore_errors=True)
        if "iou_mean" in result:
            print(f"  IoU={result['iou_mean']:.4f}+-{result['iou_std']:.4f} "
                  f"Chamfer={result['chamfer_mean']:.6f}+-{result['chamfer_std']:.6f} "
                  f"NC={result['normal_consistency_mean']:.4f}+-{result['normal_consistency_std']:.4f}")

    csv_path = os.path.join(root, "metrics.csv")
    with open(csv_path, "w", newline="") as handle:
        fields = ["frame_index", "time", "num_samples_requested", "num_samples_valid",
                  "resolution", "metric_space", "iou_mean", "iou_std", "chamfer_mean", "chamfer_std",
                  "normal_consistency_mean", "normal_consistency_std"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {field: result.get(field, "") for field in fields}
            row["metric_space"] = args.metric_space
            writer.writerow(row)

    metadata = {
        "protocol": stage_b_ckpt["protocol"],
        "case": args.case,
        "gt_case": args.gt_case,
        "train_frame_indices": stage_b_ckpt["train_frame_indices"],
        "eval_frame_indices": frame_indices,
        "resolution": args.resolution,
        "metric_space": args.metric_space,
        "num_samples": args.num_samples,
        "include_training_frames": args.include_training_frames,
        "architecture": "bidirectional_mamba_direct_latent_regression",
        "stage_a_ckpt": args.stage_a_ckpt,
        "stage_b_ckpt": args.stage_b_ckpt,
        "results": results,
    }
    with open(os.path.join(root, "metrics.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"[done] metrics -> {csv_path}")


if __name__ == "__main__":
    main()
