#!/usr/bin/env python3
"""Train and evaluate the official DSR baseline on the paper manifest.

This is the control experiment for MAMBA-DSR. It uses the original
``DSR/runner_igr_t.py`` and only the selected train-view symlinks, then uses
the shared normalized evaluator so DSR and MAMBA-DSR metrics are identical.

Seeds: upstream DSR never sets a random seed, so training goes through
dsr_seeded.py, which seeds Python/NumPy/PyTorch with --seed and then runs the
unmodified runner_igr_t.py. --tag (e.g. _s1) keeps a second seed's checkpoints,
config and evaluation apart from the first (exp/dsr_paper/<case><tag>/).
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
    cached_normalization,
    ckpt_epoch,
    ensure_gt_link,
    make_train_view,
    obj_indices,
)


def make_dsr_config(case, spec, train_view, full_count, train_count, scale, translate, epochs, tag=""):
    base = (DSR_DIR / "confs" / "abeta_res128.conf").read_text()
    base = base.replace(
        "base_exp_dir = ./exp/CASE_NAME/igrt_debug_sphere",
        f"base_exp_dir = ./exp/dsr_paper/{case}{tag}",
    )
    base = re.sub(r"data_dir\s*=.*", f"data_dir = {train_view}", base, count=1)
    base = re.sub(r"^\s*s\s*=.*$", f"    s = {scale:.12f}", base, count=1, flags=re.MULTILINE)
    base = re.sub(r"^\s*t\s*=.*$", f"    t = {translate:.12f}", base, count=1, flags=re.MULTILINE)
    base = re.sub(r"end_iter\s*=.*", f"end_iter = {epochs}", base, count=1)
    base = re.sub(r"frame_per_sequence_train\s*=.*", f"frame_per_sequence_train = {train_count}", base, count=1)
    base = re.sub(r"train_frame_interval\s*=.*", f"train_frame_interval = {spec.train_step}", base, count=1)
    base = re.sub(r"reconstruct_frame_per_seq\s*=.*", f"reconstruct_frame_per_seq = [{full_count}]", base, count=1)
    base = re.sub(r"latent_size\s*=.*", "latent_size = 192", base, count=1)
    path = CONFIG_DIR / f"dsr_{case}{tag}.conf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(base)
    return path


def drop_empty_runs(case, tag=""):
    """DSR --is_continue resumes the newest timestamp dir; one killed before its
    first checkpoint would make that load fail, so remove such dirs."""
    root = DSR_DIR / "exp" / "dsr_paper" / f"{case}{tag}"
    if root.is_dir():
        for run_dir in root.iterdir():
            if run_dir.is_dir() and not (run_dir / "checkpoints" / "ckpt_latest.pth").is_file():
                import shutil
                print(f"[{case}] removing checkpoint-less DSR run {run_dir}", flush=True)
                shutil.rmtree(run_dir)


def latest_checkpoint(case, tag=""):
    root = DSR_DIR / "exp" / "dsr_paper" / f"{case}{tag}"
    candidates = list(root.glob("*/checkpoints/ckpt_latest.pth"))
    if not candidates:
        raise FileNotFoundError(f"no DSR checkpoint found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0,
                    help="Seeds DSR training (via dsr_seeded.py) and the evaluator.")
    ap.add_argument("--tag", default="", help="Run-name suffix for extra seeds, e.g. _s1.")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--allow-truncated-local-data", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate the manifest and prepare symlink views/configs without training.")
    ap.add_argument("--resume", action="store_true",
                    help="Continue DSR training from its latest checkpoint (DSR --is_continue) "
                         "and skip finished training/evaluation.")
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
        scale, translate = cached_normalization(spec, train_indices)
        train_view = make_train_view(f"dsr_{case}", spec.full_dir, train_indices)
        ensure_gt_link(spec)
        config = make_dsr_config(
            case, spec, train_view, full_count, len(train_indices),
            scale, translate, args.epochs, args.tag,
        )
        print(
            f"[{case}{args.tag}] available={available_count} protocol_frames={full_count} "
            f"train={len(train_indices)} ({spec.note})",
            flush=True,
        )
        if args.dry_run:
            continue
        try:
            trained = ckpt_epoch(latest_checkpoint(case, args.tag)) == args.epochs
        except FileNotFoundError:
            trained = False
        if not (args.resume and trained):
            if args.resume:
                drop_empty_runs(case, args.tag)
            subprocess.run(
                [python, str(ROOT / "dsr_seeded.py"), str(args.seed),
                 "runner_igr_t.py", "--conf", str(config), "--mode", "train",
                 "--case", case, "--gpu", args.gpu, *(["--is_continue"] if args.resume else [])],
                cwd=DSR_DIR,
                check=True,
            )
        else:
            print(f"[{case}] DSR training complete ({args.epochs} epochs) -- skipped", flush=True)
        ckpt = latest_checkpoint(case, args.tag)
        run_name = f"{case}_paper_train_eval{args.tag}"
        metrics = ROOT.parent / "reconstructions" / f"DSR_{case}_{run_name}" / "metrics.json"
        if args.resume and metrics.is_file() and metrics.stat().st_mtime > ckpt.stat().st_mtime:
            print(f"[{case}] DSR evaluation complete -- skipped", flush=True)
            continue
        index_text = ",".join(str(index) for index in train_indices)
        subprocess.run(
            [
                python, str(ROOT / "evaluate_dsr.py"), "--ckpt", str(ckpt),
                "--conf", str(config), "--case", case, "--gt-case", spec.gt_case,
                "--frame-indices", index_text, "--train-indices", index_text,
                "--full-num-frames", str(full_count), "--resolution", str(args.resolution),
                "--seed", str(args.seed), "--run-name", run_name,
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
