#!/usr/bin/env python3
"""Train and evaluate the official DSR baseline on the paper manifest.

This is the control experiment for MAMBA-DSR. It uses the original
``DSR/runner_igr_t.py`` and only the selected train-view symlinks, then uses
the shared normalized evaluator so DSR and MAMBA-DSR metrics are identical.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

from paper_protocol import paper_cases
from run_multicase import (
    CONFIG_DIR,
    DSR_DIR,
    ROOT,
    ensure_gt_link,
    make_train_view,
    normalization,
    obj_indices,
)


def make_dsr_config(case, spec, train_view, full_count, train_count, scale, translate, epochs):
    base = (DSR_DIR / "confs" / "abeta_res128.conf").read_text()
    base = base.replace(
        "base_exp_dir = ./exp/CASE_NAME/igrt_debug_sphere",
        f"base_exp_dir = ./exp/dsr_paper/{case}",
    )
    base = re.sub(r"data_dir\s*=.*", f"data_dir = {train_view}", base, count=1)
    base = re.sub(r"^\s*s\s*=.*$", f"    s = {scale:.12f}", base, count=1, flags=re.MULTILINE)
    base = re.sub(r"^\s*t\s*=.*$", f"    t = {translate:.12f}", base, count=1, flags=re.MULTILINE)
    base = re.sub(r"end_iter\s*=.*", f"end_iter = {epochs}", base, count=1)
    base = re.sub(r"frame_per_sequence_train\s*=.*", f"frame_per_sequence_train = {train_count}", base, count=1)
    base = re.sub(r"train_frame_interval\s*=.*", f"train_frame_interval = {spec.train_step}", base, count=1)
    base = re.sub(r"reconstruct_frame_per_seq\s*=.*", f"reconstruct_frame_per_seq = [{full_count}]", base, count=1)
    base = re.sub(r"latent_size\s*=.*", "latent_size = 192", base, count=1)
    path = CONFIG_DIR / f"dsr_{case}.conf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(base)
    return path


def latest_checkpoint(case):
    root = DSR_DIR / "exp" / "dsr_paper" / case
    candidates = list(root.glob("*/checkpoints/ckpt_latest.pth"))
    if not candidates:
        raise FileNotFoundError(f"no DSR checkpoint found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--allow-truncated-local-data", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate the manifest and prepare symlink views/configs without training.")
    args = ap.parse_args()

    cases = paper_cases(ROOT.parent, DSR_DIR)
    selected = args.only or list(cases)
    python = sys.executable
    for case in selected:
        spec = cases[case]
        available_count = len(obj_indices(spec.full_dir))
        required = spec.required_frames()
        if required and available_count < required and not args.allow_truncated_local_data:
            raise RuntimeError(
                f"{case}: paper requires {required} frames but local data has {available_count}; "
                "obtain the missing source trajectory or explicitly allow truncation"
            )
        full_count = min(available_count, required) if required else available_count
        train_indices = spec.train_indices(full_count)
        scale, translate = normalization(spec.full_dir, train_indices)
        train_view = make_train_view(f"dsr_{case}", spec.full_dir, train_indices)
        ensure_gt_link(spec)
        config = make_dsr_config(
            case, spec, train_view, full_count, len(train_indices),
            scale, translate, args.epochs,
        )
        print(
            f"[{case}] available={available_count} protocol_frames={full_count} "
            f"train={len(train_indices)} ({spec.note})",
            flush=True,
        )
        if args.dry_run:
            continue
        subprocess.run(
            [python, "runner_igr_t.py", "--conf", str(config), "--mode", "train",
             "--case", case, "--gpu", args.gpu],
            cwd=DSR_DIR,
            check=True,
        )
        ckpt = latest_checkpoint(case)
        index_text = ",".join(str(index) for index in train_indices)
        subprocess.run(
            [
                python, str(ROOT / "evaluate_dsr.py"), "--ckpt", str(ckpt),
                "--conf", str(config), "--case", case, "--gt-case", spec.gt_case,
                "--frame-indices", index_text, "--train-indices", index_text,
                "--full-num-frames", str(full_count), "--resolution", str(args.resolution),
                "--seed", str(args.seed), "--run-name", f"{case}_paper_train_eval",
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
