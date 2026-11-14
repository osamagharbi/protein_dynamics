#!/usr/bin/env python3
"""Held-out evaluation of trained MAMBA-DSR runs: frames the model never saw.

Uses the checkpoints written by run_multicase.py (no retraining) and the same
evaluator/metrics as the training-frame table (evaluate_protocol.py, DSR
metrics, normalized coordinates, resolution 128).

Splits (all frame indices are ORIGINAL trajectory indices):
  interp       midpoint between every pair of consecutive training frames
               (AdK/NhaA/YiiP every 10th -> frames 5, 15, ...; DIMS every 2nd -> odd frames)
  future_step  next k = 1..20 frames of the thinned sequence after the last training
               frame (last + k*step), windows 1-10 and 11-20 as in DSR Table 2
  future_raw   next k = 1..20 raw trajectory frames after the last training frame
The DSR paper does not say which of the two "future" definitions Table 2 uses, so both
are reported. Frames without a ground-truth mesh in the dataset are skipped and counted.

Reference baselines (no model, CPU only), same metrics:
  copy-nearest  interp: the ground truth of the nearest training frame
  copy-last     future: the ground truth of the last training frame
A model that does not beat these has learned nothing beyond the training frames.

    python eval_heldout.py --only adk_dims ifabp_water            # all splits
    python eval_heldout.py --only adk_equilibrium --max-per-split 3   # quick test
Output: exp/heldout_summary/<case><tag>.csv (one row per split/window x {model, baseline}).
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from run_multicase import CASES, DSR_DIR, ROOT, ensure_gt_link, obj_indices

RECON = ROOT.parent / "reconstructions"
HORIZON = 20


def frame_splits(train, available, max_per_split=0):
    """{split: [(frame, window, reference_training_frame)]} limited to frames with ground truth."""
    train = sorted(train)
    step = Counter(b - a for a, b in zip(train, train[1:])).most_common(1)[0][0]
    last = train[-1]
    splits = {
        "interp": [(a + (b - a) // 2, "mid", a) for a, b in zip(train, train[1:]) if b - a > 1],
        "future_step": [(last + k * step, "1-10" if k <= 10 else "11-20", last) for k in range(1, HORIZON + 1)],
        "future_raw": [(last + k, "1-10" if k <= 10 else "11-20", last) for k in range(1, HORIZON + 1)],
    }
    missing = {}
    for name, rows in splits.items():
        kept = [r for r in rows if r[0] in available]
        missing[name] = len(rows) - len(kept)
        if max_per_split and len(kept) > max_per_split:   # evenly spaced subset, for quick tests
            pick = np.linspace(0, len(kept) - 1, max_per_split).round().astype(int)
            kept = [kept[i] for i in sorted(set(pick))]
        splits[name] = kept
    return splits, missing, step


def load_norm(path, scale, translate):
    import trimesh
    mesh = trimesh.load(path, process=False, force="mesh", skip_materials=True)
    mesh.vertices = ((mesh.vertices - translate) / scale) * 2.0 - 1.0
    return mesh


def baseline_metrics(gt_dir, frame, ref, scale, translate, resolution, cache_dir):
    """Metrics of 'predict the reference frame's ground truth' against the frame's ground truth."""
    key = {"ref": ref, "resolution": resolution, "s": scale, "t": translate}
    cache = cache_dir / f"frame_{frame:05d}.json"
    if cache.is_file():
        c = json.loads(cache.read_text())
        if c.get("key") == key:
            return c["metrics"]
    sys.path.insert(0, str(DSR_DIR))
    from lib.metrics import compute_iou, compute_normal_consistency, compute_trimesh_chamfer
    gt = load_norm(gt_dir / f"traj_{frame}.obj", scale, translate)
    pr = load_norm(gt_dir / f"traj_{ref}.obj", scale, translate)
    m = {"iou_mean": float(compute_iou(gt, pr, N=resolution)),
         "chamfer_mean": float(compute_trimesh_chamfer(pr, gt)),
         "normal_consistency_mean": float(compute_normal_consistency(gt, pr))}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"key": key, "metrics": m}))
    return m


def summarize(rows):
    out = {"n": len(rows)}
    ok = [r for r in rows if r and "iou_mean" in r]
    out["n_failed"] = len(rows) - len(ok)          # no zero crossing -> no mesh
    for key, short in (("iou_mean", "iou"), ("chamfer_mean", "chamfer"), ("normal_consistency_mean", "nc")):
        vals = np.array([r[key] for r in ok], dtype=float)
        good = vals[np.isfinite(vals)]
        out[f"{short}_mean"] = float(good.mean()) if good.size else math.nan
        out[f"{short}_std"] = float(good.std()) if good.size else math.nan
        out[f"{short}_n_nan"] = int(vals.size - good.size)
    return out


def report():
    files = sorted((ROOT / "exp" / "heldout_summary").glob("*.csv"))
    if not files:
        print("no held-out results yet (exp/heldout_summary/ is empty)")
        return
    print(f"{'run':28s} {'split':12s} {'window':6s} {'method':13s} {'n':>4s}  {'IoU':>7s}  {'CD':>9s}  {'NC':>7s}")
    for f in files:
        for r in csv.DictReader(f.open()):
            print(f"{f.stem:28s} {r['split']:12s} {r['window']:6s} {r['method']:13s} {int(r['n']):4d}  "
                  f"{float(r['iou_mean']):7.4f}  {float(r['chamfer_mean']):9.6f}  {float(r['nc_mean']):7.4f}"
                  + (f"  ({r['n_failed']} no mesh)" if r['n_failed'] != '0' else ''))
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(CASES))
    ap.add_argument("--report", action="store_true",
                    help="Print every exp/heldout_summary/*.csv as one table and exit.")
    ap.add_argument("--tag", default="", help="Run-name suffix used at training time, e.g. _s1.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--splits", nargs="+", default=["interp", "future_step", "future_raw"])
    ap.add_argument("--max-per-split", type=int, default=0, help="Quick test: frames per split (0 = all).")
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()
    if not args.only:
        ap.error("--only is required (or --report)")

    import torch
    failed = []
    for case in args.only:
        spec = CASES[case]
        stage_a = ROOT / "exp" / f"{case}_paper_mamba_stageA{args.tag}" / "checkpoints" / "ckpt_latest.pth"
        stage_b = ROOT / "exp" / f"{case}_paper_mamba_stageB{args.tag}" / "ckpt_latest.pth"
        config = ROOT / "exp" / "paper_configs" / f"{case}.conf"
        if not (stage_a.is_file() and stage_b.is_file() and config.is_file()):
            print(f"[{case}] missing trained checkpoints/config -- train it first; skipped", flush=True)
            failed.append(case)
            continue
        ensure_gt_link(spec)
        ck = torch.load(stage_b, map_location="cpu", weights_only=False)
        train = list(ck["train_frame_indices"])
        available = set(obj_indices(spec.full_dir))
        splits, missing, step = frame_splits(train, available, args.max_per_split)
        splits = {k: v for k, v in splits.items() if k in args.splits}
        frames = sorted({f for rows in splits.values() for f, _, _ in rows})
        print(f"[{case}] train {len(train)} frames {train[0]}..{train[-1]} step {step}; "
              f"ground truth for {len(available)} frames; held-out to evaluate: "
              + ", ".join(f"{k} {len(v)} (no GT {missing[k]})" for k, v in splits.items()), flush=True)
        if not frames:
            print(f"[{case}] no held-out frame has ground truth -- skipped", flush=True)
            continue

        run_name = f"{case}_paper_mamba_heldout{args.tag}"
        cmd = [sys.executable, "evaluate_protocol.py", "--conf", str(config),
               "--stage-a-ckpt", str(stage_a), "--stage-b-ckpt", str(stage_b),
               "--case", case, "--gt-case", spec.gt_case,
               "--frame-indices", ",".join(map(str, frames)), "--num-frames", str(max(frames) + 1),
               "--resolution", str(args.resolution), "--metric-space", "normalized",
               "--num-samples", "1", "--seed", str(args.seed), "--run-name", run_name]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
        if subprocess.run(cmd, cwd=ROOT, env=env).returncode != 0:
            print(f"[{case}] evaluation FAILED", flush=True)
            failed.append(case)
            continue
        root = RECON / f"MAMBA-DSR_{case}_{run_name}"
        model = {r["frame_index"]: r for r in json.loads((root / "metrics.json").read_text())["results"]}

        from pyhocon import ConfigFactory
        conf = ConfigFactory.parse_file(str(config))
        scale, translate = conf.get_float("dataset.s"), conf.get_float("dataset.t")
        gt_dir = DSR_DIR / "datasets" / spec.gt_case

        out_rows = []
        for split, rows in splits.items():
            for window in sorted({w for _, w, _ in rows}, key=lambda w: (len(w), w)):
                sel = [(f, ref) for f, w, ref in rows if w == window]
                entries = [("MAMBA-DSR", [model.get(f) for f, _ in sel])]
                if not args.no_baselines:
                    base = "copy-nearest" if split == "interp" else "copy-last"
                    entries.append((base, [baseline_metrics(gt_dir, f, ref, scale, translate,
                                                            args.resolution, root / "baseline_results")
                                           for f, ref in sel]))
                for method, metrics in entries:
                    out_rows.append({"case": case, "tag": args.tag, "split": split, "window": window,
                                     "method": method, "train_step": step,
                                     "frames": " ".join(str(f) for f, _ in sel), **summarize(metrics)})

        summary = ROOT / "exp" / "heldout_summary" / f"{case}{args.tag}.csv"
        summary.parent.mkdir(parents=True, exist_ok=True)
        with summary.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=out_rows[0].keys())
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"\n[{case}] held-out results -> {summary}")
        print(f"  {'split':12s} {'window':6s} {'method':13s} {'n':>4s}  {'IoU':>7s}  {'CD':>9s}  {'NC':>7s}")
        for r in out_rows:
            print(f"  {r['split']:12s} {r['window']:6s} {r['method']:13s} {r['n']:4d}  {r['iou_mean']:7.4f}  "
                  f"{r['chamfer_mean']:9.6f}  {r['nc_mean']:7.4f}"
                  + (f"  ({r['n_failed']} no mesh)" if r["n_failed"] else "")
                  + (f"  ({r['nc_n_nan']} NC nan)" if r["nc_n_nan"] else ""), flush=True)
    if failed:
        print(f"[failed] {' '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
