#!/usr/bin/env python3
"""Run the paper-defined MAMBA-DSR comparison protocol sequentially.

The old launcher used every fifth frame across each complete local trajectory
and pointed Stage A at the complete ``gt`` directory. That made large runs
needlessly read 6--11 GB of OBJ files and did not reproduce the DSR paper's
sampling rules. This launcher creates a train-only *symlink view*, so no mesh
is copied and Stage A reads only the selected training frames.

The default evaluation is the DSR paper's reconstruction-across-time task:
the selected training frames are reconstructed with the trained model and
metrics are computed in normalized coordinates. Held-out interpolation must
be requested separately with an explicit frame list.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

from paper_protocol import PaperCase, paper_cases


ROOT = Path(__file__).resolve().parent
DSR_DIR = ROOT.parent / "DSR"
STAGE_DIR = DSR_DIR / "datasets" / "mamba_paper_views"
CONFIG_DIR = ROOT / "exp" / "paper_configs"
CASES = paper_cases(ROOT.parent, DSR_DIR)


def obj_indices(path: Path) -> list[int]:
    indices = []
    for item in path.glob("traj_*.obj"):
        match = re.fullmatch(r"traj_(\d+)\.obj", item.name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def normalization(source_dir: Path, indices: list[int]):
    """Match DSR's normalization using only the selected training meshes."""
    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    for index in indices:
        path = source_dir / f"traj_{index}.obj"
        with path.open() as handle:
            for line in handle:
                if not line.startswith("v "):
                    continue
                fields = line.split()
                xyz = [float(fields[1]), float(fields[2]), float(fields[3])]
                for axis in range(3):
                    mins[axis] = min(mins[axis], xyz[axis])
                    maxs[axis] = max(maxs[axis], xyz[axis])
    if any(value == float("inf") for value in mins):
        raise RuntimeError(f"no OBJ vertices found in {source_dir}")
    return max(maxs[i] - mins[i] for i in range(3)) * 1.1, min(mins)


def cached_normalization(spec: PaperCase, train_indices: list[int]):
    """Reuse jeanzay/data_prep meta.json (same rule, same frames) instead of re-reading OBJs."""
    meta_path = spec.full_dir.parent / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        norm = meta.get("dsr_normalization") or {}
        if (norm.get("s") is not None and norm.get("computed_on") == "paper_train_frames"
                and meta.get("paper_train_indices") == train_indices):
            return float(norm["s"]), float(norm["t"])
    return normalization(spec.full_dir, train_indices)


def ckpt_epoch(path: Path):
    if not path.is_file():
        return None
    import torch
    return int(torch.load(path, map_location="cpu", weights_only=False).get("epoch", 0))


def link_file(link: Path, target: Path):
    link.parent.mkdir(parents=True, exist_ok=True)
    target = target.resolve()
    if link.is_symlink() and link.resolve() == target:
        return
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target)


def make_train_view(case: str, source_dir: Path, train_indices: list[int]) -> Path:
    view_dir = STAGE_DIR / case / "obj"
    view_dir.mkdir(parents=True, exist_ok=True)
    expected = {f"traj_{index}.obj" for index in train_indices}
    # Remove only managed stale links from this case's view.
    for item in view_dir.glob("traj_*.obj"):
        if item.name not in expected:
            item.unlink()
    for index in train_indices:
        source = source_dir / f"traj_{index}.obj"
        if not source.is_file():
            raise FileNotFoundError(f"missing selected training mesh: {source}")
        link_file(view_dir / source.name, source)
    return view_dir.parent


def ensure_gt_link(spec: PaperCase):
    if spec.gt_case == "abeta_30_gt":
        return
    link = DSR_DIR / "datasets" / spec.gt_case
    target = spec.full_dir.resolve()
    if link.is_symlink() and link.resolve() == target:
        return
    if link.exists() or link.is_symlink():
        raise RuntimeError(f"refusing to replace existing ground-truth path: {link}")
    link.symlink_to(target, target_is_directory=True)


def make_config(case: str, spec: PaperCase, train_view: Path,
                full_count: int, train_indices: list[int], scale: float, translate: float) -> Path:
    base = (DSR_DIR / "confs" / "abeta_res128.conf").read_text()
    base = base.replace(
        "base_exp_dir = ./exp/CASE_NAME/igrt_debug_sphere",
        "base_exp_dir = ./exp/mamba_paper/CASE_NAME",
    )
    base = re.sub(r"data_dir\s*=.*", f"data_dir = {train_view}", base, count=1)
    base = re.sub(r"^\s*s\s*=.*$", f"    s = {scale:.12f}", base, count=1, flags=re.MULTILINE)
    base = re.sub(r"^\s*t\s*=.*$", f"    t = {translate:.12f}", base, count=1, flags=re.MULTILINE)
    base = re.sub(r"frame_per_sequence_train\s*=.*", f"frame_per_sequence_train = {len(train_indices)}", base, count=1)
    base = re.sub(r"train_frame_interval\s*=.*", f"train_frame_interval = {spec.train_step}", base, count=1)
    base = re.sub(r"reconstruct_frame_per_seq\s*=.*", f"reconstruct_frame_per_seq = [{full_count}]", base, count=1)
    config_path = CONFIG_DIR / f"{case}.conf"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(base)
    return config_path


def run(command):
    print("[run] " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--stage-a-epochs", type=int, default=4000,
                        help="Official DSR config uses 4000 Stage-A iterations.")
    parser.add_argument("--stage-a-frame-batch-size", type=int, default=1,
                        help="MAMBA-DSR Stage-A frames per decoder update. All selected "
                             "frames/points remain in the dataset; values >1 accelerate "
                             "training by changing only optimization batching.")
    parser.add_argument("--stage-b-epochs", type=int, default=5000)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", nargs="*", choices=list(CASES))
    parser.add_argument("--allow-truncated-local-data", action="store_true",
                        help="Permit a local trajectory shorter than the paper's documented crop. "
                             "The result is then labeled non-exact.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the manifest and prepare symlink views/configs without training.")
    parser.add_argument("--list", action="store_true", help="List the known cases and exit.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip finished stages and continue Stage A from its checkpoint.")
    parser.add_argument("--stop-after-minutes", type=float, default=0.0,
                        help="Stage-A time budget for walltime-limited jobs (0 = none). If Stage A "
                             "is not finished the launcher exits with code 3 (resubmit).")
    parser.add_argument("--tag", default="",
                        help="Suffix for run names, e.g. _s1 for a second seed.")
    args = parser.parse_args()

    if args.list:
        for name, spec in CASES.items():
            print(f"{name:24s} {spec.full_dir}  ({spec.note})")
        return

    selected = args.only or list(CASES)
    summary_rows = []
    python = sys.executable

    for case in selected:
        spec = CASES[case]
        if not spec.full_dir.is_dir():
            raise FileNotFoundError(f"{case}: missing source directory {spec.full_dir}")
        available_indices = obj_indices(spec.full_dir)
        available_count = len(available_indices)
        required = spec.required_frames()
        if required and available_count < required and not args.allow_truncated_local_data:
            raise RuntimeError(
                f"{case}: paper requires at least {required} source frames, but local data has "
                f"{available_count}. Obtain the paper's source trajectory or pass "
                "--allow-truncated-local-data for a clearly non-exact run."
            )
        full_count = min(available_count, required) if required else available_count
        train_indices = spec.train_indices(full_count)
        scale, translate = cached_normalization(spec, train_indices)
        train_view = make_train_view(case, spec.full_dir, train_indices)
        ensure_gt_link(spec)
        config = make_config(case, spec, train_view, full_count, train_indices, scale, translate)
        print(
            f"[{case}] available={available_count} protocol_frames={full_count} "
            f"train={len(train_indices)} ({spec.note}) scale={scale:.6f} translate={translate:.6f}",
            flush=True,
        )

        if args.dry_run:
            continue

        stage_a_name = f"{case}_paper_mamba_stageA{args.tag}"
        stage_b_name = f"{case}_paper_mamba_stageB{args.tag}"
        eval_name = f"{case}_paper_mamba_train_eval{args.tag}"
        stage_a = ROOT / "exp" / stage_a_name / "checkpoints" / "ckpt_latest.pth"
        stage_b = ROOT / "exp" / stage_b_name / "ckpt_latest.pth"
        metrics_path = ROOT.parent / "reconstructions" / f"MAMBA-DSR_{case}_{eval_name}" / "metrics.json"
        index_text = ",".join(str(index) for index in train_indices)

        if not (args.resume and ckpt_epoch(stage_a) == args.stage_a_epochs):
            run([
                python, "train_autodecoder.py", "--conf", str(config), "--case", case,
                "--protocol", "none", "--train-indices", index_text,
                "--full-num-frames", str(full_count), "--latent-size", "8",
                "--smooth-lambda", "1.0", "--end-iter", str(args.stage_a_epochs),
                "--frame-batch-size", str(args.stage_a_frame_batch_size),
                "--seed", str(args.seed), "--gpu", args.gpu, "--run-name", stage_a_name,
                *(["--resume"] if args.resume else []),
                *(["--stop-after-minutes", str(args.stop_after_minutes)] if args.stop_after_minutes else []),
            ])
            done = ckpt_epoch(stage_a)
            if done != args.stage_a_epochs:
                print(f"[{case}] Stage A at epoch {done}/{args.stage_a_epochs} -> resubmit to continue",
                      flush=True)
                sys.exit(3)
        else:
            print(f"[{case}] Stage A complete ({args.stage_a_epochs} epochs) -- skipped", flush=True)

        stage_b_fresh = stage_b.is_file() and stage_b.stat().st_mtime > stage_a.stat().st_mtime
        if not (args.resume and stage_b_fresh and ckpt_epoch(stage_b) == args.stage_b_epochs):
            run([
                python, "train_mamba.py", "--conf", str(config), "--stage-a-ckpt", str(stage_a),
                "--case", case, "--epochs", str(args.stage_b_epochs), "--seed", str(args.seed),
                "--run-name", stage_b_name,
            ])
        else:
            print(f"[{case}] Stage B complete -- skipped", flush=True)

        if not (args.resume and metrics_path.is_file()
                and metrics_path.stat().st_mtime > stage_b.stat().st_mtime):
            run([
                python, "evaluate_protocol.py", "--conf", str(config), "--stage-a-ckpt", str(stage_a),
                "--stage-b-ckpt", str(stage_b), "--case", case, "--gt-case", spec.gt_case,
                "--frame-indices", index_text, "--include-training-frames",
                "--resolution", str(args.resolution), "--metric-space", "normalized",
                "--num-samples", "1", "--seed", str(args.seed), "--run-name", eval_name,
            ])
        else:
            print(f"[{case}] evaluation complete -- skipped", flush=True)

        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text())
            valid = [row for row in metrics["results"] if "iou_mean" in row]
            if valid:
                summary_rows.append({
                    "case": case,
                    "seed": args.seed,
                    "tag": args.tag,
                    "protocol": "dsr_paper_reconstruction_train_frames",
                    "train_rule": spec.note,
                    "available_frames": available_count,
                    "protocol_frames": full_count,
                    "train_frames": len(train_indices),
                    "metric_space": "normalized",
                    "iou_mean": sum(row["iou_mean"] for row in valid) / len(valid),
                    "chamfer_mean": sum(row["chamfer_mean"] for row in valid) / len(valid),
                    "normal_consistency_mean": sum(row["normal_consistency_mean"] for row in valid) / len(valid),
                    "stage_a": str(stage_a),
                    "stage_b": str(stage_b),
                })

    # One file per case/run so concurrent Slurm jobs never overwrite each other.
    for row in summary_rows:
        summary_path = ROOT / "exp" / "paper_summary" / f"{row['case']}{args.tag}.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)
        print(f"[summary] {row['case']}: IoU {row['iou_mean']:.4f}  Chamfer {row['chamfer_mean']:.6f}  "
              f"NC {row['normal_consistency_mean']:.4f} -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
