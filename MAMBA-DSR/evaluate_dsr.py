#!/usr/bin/env python3
"""Evaluate an official DSR checkpoint with the paper metric procedure.

This is deliberately separate from ``evaluate_protocol.py`` because an
official DSR checkpoint has one sequence latent and no Stage-B checkpoint.
Both evaluators use the same frame list, normalized mesh coordinates, DSR IoU,
DSR Chamfer implementation, and normal-consistency implementation.
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
from lib.metrics import compute_iou, compute_normal_consistency, compute_trimesh_chamfer  # noqa: E402
from lib.models.network import ImplicitNet  # noqa: E402
from lib.plots import get_mesh  # noqa: E402

from protocol import ground_truth_path, parse_indices, protocol_times, seed_everything, validate_indices  # noqa: E402

RECON_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reconstructions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--conf", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--gt-case", required=True)
    ap.add_argument("--frame-indices", required=True)
    ap.add_argument("--full-num-frames", type=int, required=True)
    ap.add_argument("--train-indices", required=True,
                    help="The frames used to normalize time in the official DSR run.")
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="dsr_paper_train_eval")
    ap.add_argument("--keep-reconstructions", action="store_true",
                    help="Keep per-frame predicted OBJ files. By default they are removed "
                         "after metrics are computed to avoid filling the disk.")
    args = ap.parse_args()
    seed_everything(args.seed)

    conf = ConfigFactory.parse_string(open(args.conf).read().replace("CASE_NAME", args.case))
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    latent_state = ckpt["latent_codes"]
    latent_size = int(latent_state["weight"].shape[-1])
    d_in = conf.get_int("dataset.d_in")
    decoder = ImplicitNet(
        d_in=d_in + latent_size + 1,
        **conf.get_config("model.network"),
    ).cuda()
    decoder = torch.nn.DataParallel(decoder)
    decoder.load_state_dict(ckpt["model_state_dict"])
    decoder.eval()
    latent = latent_state["weight"][0:1].float().cuda()

    train_indices = parse_indices(args.train_indices)
    frame_indices = parse_indices(args.frame_indices)
    validate_indices(train_indices, args.full_num_frames, "train_indices")
    validate_indices(frame_indices, args.full_num_frames, "frame_indices")
    times = np.asarray(protocol_times(args.full_num_frames, train_indices), dtype=np.float32)
    scale = conf.get_float("dataset.s")
    translate = conf.get_float("dataset.t")
    root = os.path.join(RECON_ROOT, f"DSR_{args.case}_{args.run_name}")
    os.makedirs(root, exist_ok=True)
    rows = []

    for frame_index in frame_indices:
        gt_path = ground_truth_path(DSR_DIR, args.gt_case, frame_index)
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(gt_path)
        out_dir = os.path.join(root, f"frame_{frame_index:05d}")
        obj_dir = os.path.join(out_dir, "obj")
        os.makedirs(obj_dir, exist_ok=True)
        shapename = "sample_00"
        get_mesh(
            path=obj_dir, epoch=0, decoder=decoder, latent=latent,
            t=torch.tensor([times[frame_index]], device="cuda"),
            shapename=shapename, resolution=args.resolution, connected=False,
        )
        pred_path = os.path.join(obj_dir, "igr_epoch_0_sample_00.obj")
        pred = trimesh.load(pred_path, process=False, force="mesh", skip_materials=True)
        gt = trimesh.load(gt_path, process=False, force="mesh", skip_materials=True)
        gt.vertices = ((gt.vertices - translate) / scale) * 2.0 - 1.0
        metrics = {
            "frame_index": frame_index,
            "time": float(times[frame_index]),
            "resolution": args.resolution,
            "iou": float(compute_iou(gt, pred, N=args.resolution)),
            "chamfer": float(compute_trimesh_chamfer(pred, gt)),
            "normal_consistency": float(compute_normal_consistency(gt, pred)),
        }
        rows.append(metrics)
        if not args.keep_reconstructions:
            shutil.rmtree(out_dir, ignore_errors=True)
        print(
            f"[eval] frame {frame_index}: IoU={metrics['iou']:.4f} "
            f"Chamfer={metrics['chamfer']:.6f} NC={metrics['normal_consistency']:.4f}",
            flush=True,
        )

    csv_path = os.path.join(root, "metrics.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "method": "DSR",
        "case": args.case,
        "gt_case": args.gt_case,
        "train_indices": train_indices,
        "frame_indices": frame_indices,
        "full_num_frames": args.full_num_frames,
        "resolution": args.resolution,
        "metric_space": "normalized",
        "results": rows,
        "mean": {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("iou", "chamfer", "normal_consistency")
        },
        "std": {
            key: float(np.std([row[key] for row in rows]))
            for key in ("iou", "chamfer", "normal_consistency")
        },
    }
    with open(os.path.join(root, "metrics.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[done] metrics -> {csv_path}")


if __name__ == "__main__":
    main()
