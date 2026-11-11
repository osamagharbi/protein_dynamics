#!/usr/bin/env python3
"""Compare corrected-protocol Diffusion-DSR and the existing DSR meshes."""

import argparse
import csv
import json
import os
import glob
import gc

import numpy as np
import trimesh

from lib.metrics import compute_iou, compute_normal_consistency, compute_trimesh_chamfer
from protocol import dsr_eval_indices, dsr_train_indices


def load_mesh(path):
    return trimesh.load(path, process=False, force="mesh", skip_materials=True)


def summary(rows, prefix):
    return {
        metric: {
            "mean": float(np.mean([row[prefix + metric] for row in rows])),
            "std": float(np.std([row[prefix + metric] for row in rows])),
            "median": float(np.median([row[prefix + metric] for row in rows])),
        }
        for metric in ("iou", "chamfer", "normal_consistency")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion-csv", required=True)
    parser.add_argument("--dsr-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=300000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.diffusion_csv, newline="") as handle:
        diffusion_rows = {int(row["frame_index"]): row for row in csv.DictReader(handle)}

    eval_indices = dsr_eval_indices(149, dsr_train_indices(149))
    rows = []
    for frame_index in eval_indices:
        gt_path = os.path.join(args.gt_dir, f"traj_{frame_index}.obj")
        # Match by frame index. The DSR reconstruction script encodes the
        # normalized time in the filename, and Python's formatting can differ
        # by one last decimal place (e.g. 0.006896 vs 0.006897).
        dsr_matches = glob.glob(
            os.path.join(args.dsr_dir, f"igr_epoch_*_frame_{frame_index:05d}_t_*_seq_00.obj")
        )
        if len(dsr_matches) != 1:
            raise FileNotFoundError(
                f"expected one DSR mesh for frame {frame_index}, found {dsr_matches}"
            )
        dsr_path = dsr_matches[0]
        if frame_index not in diffusion_rows:
            raise ValueError(f"Missing Diffusion-DSR row for frame {frame_index}")

        np.random.seed(frame_index)
        gt = load_mesh(gt_path)
        dsr = load_mesh(dsr_path)
        rows.append({
            "frame_index": frame_index,
            "dsr_iou": compute_iou(gt, dsr, N=128),
            "dsr_chamfer": compute_trimesh_chamfer(dsr, gt, num_mesh_samples=args.num_samples),
            "dsr_normal_consistency": compute_normal_consistency(gt, dsr),
            "diffusion_iou": float(diffusion_rows[frame_index]["iou_mean"]),
            "diffusion_chamfer": float(diffusion_rows[frame_index]["chamfer_mean"]),
            "diffusion_normal_consistency": float(diffusion_rows[frame_index]["normal_consistency_mean"]),
        })
        del gt, dsr
        gc.collect()

    output_csv = os.path.join(args.output_dir, "protocol_comparison.csv")
    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "num_eval_frames": len(rows),
        "train_indices": dsr_train_indices(149),
        "eval_indices": eval_indices,
        "resolution": 128,
        "chamfer_samples": args.num_samples,
        "dsr": summary(rows, "dsr_"),
        "diffusion_dsr": summary(rows, "diffusion_"),
    }
    with open(os.path.join(args.output_dir, "protocol_comparison.json"), "w") as handle:
        json.dump(report, handle, indent=2)

    with open(os.path.join(args.output_dir, "PROTOCOL_COMPARISON.md"), "w") as handle:
        handle.write("# Corrected-protocol DSR comparison\n\n")
        handle.write(f"Evaluation frames: **{len(rows)}**  \nResolution: **128**  \n")
        handle.write(f"Chamfer samples per mesh: **{args.num_samples:,}**\n\n")
        handle.write("| Model | IoU (mean ± std) | Chamfer (mean ± std) | NC (mean ± std) |\n")
        handle.write("|---|---:|---:|---:|\n")
        for label, key in (("DSR", "dsr"), ("Diffusion-DSR", "diffusion_dsr")):
            values = report[key]
            handle.write(
                f"| {label} | {values['iou']['mean']:.4f} ± {values['iou']['std']:.4f} | "
                f"{values['chamfer']['mean']:.4f} ± {values['chamfer']['std']:.4f} | "
                f"{values['normal_consistency']['mean']:.4f} ± {values['normal_consistency']['std']:.4f} |\n"
            )
        handle.write("\nHigher is better for IoU and NC; lower is better for Chamfer.\n")

    print(f"Wrote {output_csv}")
    print(json.dumps(report["dsr"], indent=2))
    print(json.dumps(report["diffusion_dsr"], indent=2))


if __name__ == "__main__":
    main()
