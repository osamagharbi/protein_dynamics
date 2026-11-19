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

Methods:
  --method mamba (default)   MAMBA-DSR (Stage A + Stage B checkpoints from run_multicase.py)
  --method dsr               official DSR baseline (checkpoint from run_dsr_paper.py), same frames
  --mode hold_t              diagnostic, MAMBA-DSR future frames only: decoder time held at the
                             last training time (t = 1), Mamba latent still follows t
  --mode hold_z              diagnostic: Mamba latent held at t = 1, decoder time still follows t
  --mode hold_all            diagnostic: decoder time AND Mamba latent held at t = 1
                             (= the model's own reconstruction of the last training frame)
  --mode lerp                ablation, interpolation frames only: no Stage B -- the latent is the
                             linear interpolation of the Stage-A codes of the two neighbouring
                             training frames ("code-lerp"). If it matches MAMBA-DSR, the Mamba
                             time model adds nothing over the per-frame codes.
  --tag _s1                  run trained with SEED=1 (MAMBA-DSR or DSR)

    python eval_heldout.py --only adk_dims ifabp_water            # all splits
    python eval_heldout.py --only adk_dims --method dsr
    python eval_heldout.py --only adk_dims --mode hold_t
    python eval_heldout.py --only adk_dims --mode lerp
    python eval_heldout.py --only adk_equilibrium --max-per-split 3   # quick test
    python eval_heldout.py --report      # every result + seed mean/sd + paired per-frame tests
Output: exp/heldout_summary/<run>.csv (one row per split/window x {method, baseline}).
"""

import argparse
import csv
import functools
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from run_multicase import CASES, CONFIG_DIR, DSR_DIR, ROOT, ckpt_epoch, ensure_gt_link, obj_indices

RECON = ROOT.parent / "reconstructions"
HORIZON = 20
MODES = {   # mode: (evaluate_protocol.py options, splits it applies to (prefix), method label)
    "extrapolate": ({}, "", "MAMBA-DSR"),
    "hold_t": ({"clamp_decoder_time": 1.0}, "future", "MAMBA-DSR [hold_t]"),
    "hold_z": ({"clamp_latent_time": 1.0}, "future", "MAMBA-DSR [hold_z]"),
    "hold_all": ({"clamp_decoder_time": 1.0, "clamp_latent_time": 1.0}, "future", "MAMBA-DSR [hold_all]"),
    "lerp": ({"latent_source": "lerp"}, "interp", "code-lerp"),
}
LABEL_SUFFIX = {label: ("" if m == "extrapolate" else f"_{m}") for m, (_, _, label) in MODES.items()}
METHOD_ORDER = ["MAMBA-DSR", "MAMBA-DSR [hold_t]", "MAMBA-DSR [hold_z]", "MAMBA-DSR [hold_all]",
                "code-lerp", "DSR", "copy-nearest", "copy-last"]
BASELINES = ("copy-nearest", "copy-last")
SPLIT_ORDER = ["interp", "future_step", "future_raw"]


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
    rows, seen = [], set()
    for f in files:
        for r in csv.DictReader(f.open()):
            key = (r["case"], r["tag"], r["split"], r["window"], r["method"])
            if key not in seen:            # baselines appear in every file of a case
                seen.add(key)
                rows.append(r)
    order = lambda r: (r["case"], r["tag"], SPLIT_ORDER.index(r["split"]), (len(r["window"]), r["window"]),
                       METHOD_ORDER.index(r["method"]) if r["method"] in METHOD_ORDER else 99)
    rows.sort(key=order)
    print(f"{'run':24s} {'split':12s} {'window':6s} {'method':21s} {'n':>4s}  {'IoU':>7s}  {'CD':>9s}  {'NC':>7s}")
    last = None
    for r in rows:
        run = r["case"] + r["tag"]
        if last and run != last:
            print()
        last = run
        print(f"{run:24s} {r['split']:12s} {r['window']:6s} {r['method']:21s} {int(r['n']):4d}  "
              f"{float(r['iou_mean']):7.4f}  {float(r['chamfer_mean']):9.6f}  {float(r['nc_mean']):7.4f}"
              + (f"  ({r['n_failed']} no mesh)" if r['n_failed'] != '0' else ''))
    seed_tables(rows)
    paired_tables(rows)


@functools.lru_cache(maxsize=None)
def per_frame_iou(case, tag, method):
    """{frame: IoU} of one held-out result, from the evaluator outputs (NaN = no mesh)."""
    try:
        if method in LABEL_SUFFIX:
            d = RECON / f"MAMBA-DSR_{case}_{case}_paper_mamba_heldout{tag}{LABEL_SUFFIX[method]}"
            res = json.loads((d / "metrics.json").read_text())["results"]
            return {int(r["frame_index"]): float(r.get("iou_mean", math.nan)) for r in res}
        if method == "DSR":
            d = RECON / f"DSR_{case}_{case}_paper_heldout{tag}"
            res = json.loads((d / "metrics.json").read_text())["results"]
            return {int(r["frame_index"]): float(r.get("iou", math.nan)) for r in res}
        return {int(f.stem[len("frame_"):]): float(json.loads(f.read_text())["metrics"]["iou_mean"])
                for f in (RECON / "heldout_baselines" / case).glob("frame_*.json")}
    except (OSError, KeyError, ValueError):
        return {}


def paired_stats(a, b):
    """mean(a-b), #a>b, n, two-sided Wilcoxon signed-rank p (NaN if n < 5 or scipy missing)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    p = math.nan
    if d.size >= 5 and np.any(d != 0):
        try:
            from scipy.stats import wilcoxon
            p = float(wilcoxon(d).pvalue)
        except Exception:
            pass
    return float(d.mean()), int((d > 0).sum()), int(d.size), p


def fmt_p(p):
    return "    -  " if not np.isfinite(p) else (f"{p:7.1e}" if p < 1e-3 else f"{p:7.4f}")


def seed_tables(rows):
    """Mean +- sd over seeds (tags '', _s1, _s2 ...): training frames and held-out splits."""
    def msd(v):
        v = np.asarray(v, float)
        return f"{v.mean():.4f} +- {v.std(ddof=1) if v.size > 1 else 0:.4f} (n={v.size})"

    train = {}                                   # (method, case) -> [IoU per seed]
    for f in (ROOT / "exp" / "paper_summary").glob("*.csv"):
        m = re.fullmatch(r"(.+?)(_s\d+)?", f.stem)
        if m and not f.stem.endswith("_smoke"):
            r = next(csv.DictReader(f.open()), None)
            if r:
                train.setdefault(("MAMBA-DSR", m.group(1)), []).append(float(r["iou_mean"]))
    for f in RECON.glob("DSR_*_paper_train_eval*/metrics.json"):
        m = re.fullmatch(r"DSR_(.+)_\1_paper_train_eval(_s\d+)?", f.parent.name)
        if m:
            train.setdefault(("DSR", m.group(1)), []).append(json.loads(f.read_text())["mean"]["iou"])
    if train:
        print("\ntraining frames, IoU mean +- sd over seeds")
        print(f"  {'case':24s} {'MAMBA-DSR':>26s} {'DSR':>26s} {'diff':>8s}")
        for case in sorted({c for _, c in train}):
            a, b = train.get(("MAMBA-DSR", case), []), train.get(("DSR", case), [])
            diff = f"{np.mean(a) - np.mean(b):+8.4f}" if a and b else ""
            print(f"  {case:24s} {msd(a) if a else '-':>26s} {msd(b) if b else '-':>26s} {diff}")

    held = {}
    for r in rows:
        if r["method"] not in BASELINES:
            held.setdefault((r["case"], r["split"], r["window"], r["method"]), []).append(float(r["iou_mean"]))
    multi = {k: v for k, v in held.items() if len(v) > 1}
    if multi:
        print("\nheld-out, IoU mean +- sd over seeds (only results with >= 2 seeds)")
        for (case, split, window, method), v in sorted(
                multi.items(), key=lambda kv: (kv[0][0], SPLIT_ORDER.index(kv[0][1]), kv[0][2],
                                               METHOD_ORDER.index(kv[0][3]) if kv[0][3] in METHOD_ORDER else 99)):
            print(f"  {case:24s} {split:12s} {window:6s} {method:21s} {msd(v)}")


def paired_tables(rows):
    """Per-frame paired comparisons on the same frames (Wilcoxon signed-rank)."""
    groups = {}
    for r in rows:
        groups.setdefault((r["case"], r["tag"], r["split"], r["window"]), {})[r["method"]] = r
    out = []
    print("\npaired per-frame IoU comparisons (same frames; wins = frames where A > B; "
          "Wilcoxon signed-rank, two-sided; n >= 5 only)")
    print(f"  {'run':24s} {'split':12s} {'window':6s} {'A':10s} {'B':13s} {'dIoU':>8s} {'wins':>8s} {'p':>7s}")
    for (case, tag, split, window), methods in sorted(
            groups.items(), key=lambda kv: (kv[0][0], kv[0][1], SPLIT_ORDER.index(kv[0][2]), kv[0][3])):
        base = "copy-nearest" if split == "interp" else "copy-last"
        frames = [int(f) for f in methods[next(iter(methods))]["frames"].split()]
        for a, b in (("MAMBA-DSR", "DSR"), ("MAMBA-DSR", "code-lerp"), ("code-lerp", "DSR"),
                     ("MAMBA-DSR", base), ("DSR", base)):
            if a not in methods or b not in methods:
                continue
            fa, fb = per_frame_iou(case, tag, a), per_frame_iou(case, tag, b)
            pairs = [(fa[f], fb[f]) for f in frames
                     if f in fa and f in fb and np.isfinite(fa[f]) and np.isfinite(fb[f])]
            if len(pairs) < 5:
                continue
            d, wins, n, p = paired_stats(*zip(*pairs))
            out.append({"case": case, "tag": tag, "split": split, "window": window, "a": a, "b": b, "d": d})
            print(f"  {case + tag:24s} {split:12s} {window:6s} {a:10s} {b:13s} {d:+8.4f} {wins:4d}/{n:<3d} {fmt_p(p)}")

    # Across datasets: one mean difference per dataset (seed 0), Wilcoxon over datasets.
    print("\nacross datasets (seed 0, one mean dIoU per dataset; Wilcoxon over datasets)")
    for split, window in (("interp", "mid"), ("future_raw", "1-10"), ("future_step", "1-10")):
        for a, b in (("MAMBA-DSR", "DSR"), ("MAMBA-DSR", "code-lerp"), ("code-lerp", "DSR"),
                     ("MAMBA-DSR", "copy-nearest" if split == "interp" else "copy-last"),
                     ("DSR", "copy-nearest" if split == "interp" else "copy-last")):
            ds = [o["d"] for o in out if o["tag"] == "" and o["split"] == split and o["window"] == window
                  and o["a"] == a and o["b"] == b]
            if len(ds) >= 2:
                d, wins, n, p = paired_stats(ds, np.zeros(len(ds)))
                print(f"  {split:12s} {window:6s} {a:10s} - {b:13s} mean {d:+.4f}   A better on {wins}/{n} "
                      f"datasets   p {fmt_p(p).strip()}")


def case_frames(spec):
    """Protocol frame count and training list, computed exactly as the training launchers do."""
    available = obj_indices(spec.full_dir)
    required = spec.required_frames()
    full = min(len(available), required) if required else len(available)
    return set(available), spec.train_indices(full)


def run_mamba(case, spec, args, frames, run_name, extra):
    stage_a = ROOT / "exp" / f"{case}_paper_mamba_stageA{args.tag}" / "checkpoints" / "ckpt_latest.pth"
    stage_b = ROOT / "exp" / f"{case}_paper_mamba_stageB{args.tag}" / "ckpt_latest.pth"
    config = CONFIG_DIR / f"{case}.conf"
    if not (stage_a.is_file() and stage_b.is_file() and config.is_file()):
        print(f"[{case}] missing MAMBA-DSR checkpoints/config -- train it first; skipped", flush=True)
        return None, None
    cmd = [sys.executable, "evaluate_protocol.py", "--conf", str(config),
           "--stage-a-ckpt", str(stage_a), "--stage-b-ckpt", str(stage_b),
           "--case", case, "--gt-case", spec.gt_case,
           "--frame-indices", ",".join(map(str, frames)), "--num-frames", str(max(frames) + 1),
           "--resolution", str(args.resolution), "--metric-space", "normalized",
           "--num-samples", "1", "--seed", str(args.seed), "--run-name", run_name]
    for k, v in extra.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    if subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)).returncode:
        return None, config
    root = RECON / f"MAMBA-DSR_{case}_{run_name}"
    return {r["frame_index"]: r for r in json.loads((root / "metrics.json").read_text())["results"]}, config


def run_dsr(case, spec, args, frames, train, run_name):
    cks = sorted((DSR_DIR / "exp" / "dsr_paper" / f"{case}{args.tag}").glob("*/checkpoints/ckpt_latest.pth"),
                 key=lambda q: q.stat().st_mtime)
    config = CONFIG_DIR / f"dsr_{case}{args.tag}.conf"
    if not cks or not config.is_file() or ckpt_epoch(cks[-1]) != args.dsr_epochs:
        print(f"[{case}] DSR{args.tag} not trained to {args.dsr_epochs} epochs yet -- skipped", flush=True)
        return None, None
    cmd = [sys.executable, "evaluate_dsr.py", "--ckpt", str(cks[-1]), "--conf", str(config),
           "--case", case, "--gt-case", spec.gt_case,
           "--frame-indices", ",".join(map(str, frames)), "--train-indices", ",".join(map(str, train)),
           "--full-num-frames", str(max(frames) + 1), "--resolution", str(args.resolution),
           "--seed", str(args.seed), "--run-name", run_name]
    if subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)).returncode:
        return None, config
    rows = json.loads((RECON / f"DSR_{case}_{run_name}" / "metrics.json").read_text())["results"]
    return {r["frame_index"]: {"iou_mean": r["iou"], "chamfer_mean": r["chamfer"],
                               "normal_consistency_mean": r["normal_consistency"]} for r in rows}, config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(CASES))
    ap.add_argument("--report", action="store_true",
                    help="Print every exp/heldout_summary/*.csv as one table and exit.")
    ap.add_argument("--method", choices=["mamba", "dsr"], default="mamba")
    ap.add_argument("--mode", "--future-mode", dest="mode", choices=list(MODES), default="extrapolate",
                    help="MAMBA-DSR diagnostic / ablation (see above).")
    ap.add_argument("--tag", default="", help="Run-name suffix used at training time, e.g. _s1.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--dsr-epochs", type=int, default=4000, help="DSR must be trained this far.")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--splits", nargs="+", default=SPLIT_ORDER)
    ap.add_argument("--max-per-split", type=int, default=0, help="Quick test: frames per split (0 = all).")
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()
    if not args.only:
        ap.error("--only is required (or --report)")
    if args.method == "dsr" and args.mode != "extrapolate":
        ap.error("--mode applies to --method mamba only")
    mode = args.mode
    mode_opts, mode_splits, mode_label = MODES[mode]
    # hold_* change only t > 1 (future frames); lerp is defined between training frames only.
    args.splits = [s for s in args.splits if s.startswith(mode_splits)]

    failed = []
    for case in args.only:
        spec = CASES[case]
        ensure_gt_link(spec)
        available, train = case_frames(spec)
        splits, missing, step = frame_splits(train, available, args.max_per_split)
        splits = {k: v for k, v in splits.items() if k in args.splits}
        frames = sorted({f for rows in splits.values() for f, _, _ in rows})
        print(f"[{case}] train {len(train)} frames {train[0]}..{train[-1]} step {step}; "
              f"ground truth for {len(available)} frames; held-out to evaluate: "
              + ", ".join(f"{k} {len(v)} (no GT {missing[k]})" for k, v in splits.items()), flush=True)
        if not frames:
            print(f"[{case}] no held-out frame has ground truth -- skipped", flush=True)
            continue

        if args.method == "dsr":
            label, out_name = "DSR", f"dsr_{case}{args.tag}"
            model, config = run_dsr(case, spec, args, frames, train, f"{case}_paper_heldout{args.tag}")
        else:
            label, suffix = mode_label, LABEL_SUFFIX[mode_label]
            out_name = f"{case}{args.tag}{suffix}"
            model, config = run_mamba(case, spec, args, frames,
                                      f"{case}_paper_mamba_heldout{args.tag}{suffix}", mode_opts)
        if model is None:
            failed.append(case)
            continue

        from pyhocon import ConfigFactory
        conf = ConfigFactory.parse_file(str(config))
        scale, translate = conf.get_float("dataset.s"), conf.get_float("dataset.t")
        gt_dir = DSR_DIR / "datasets" / spec.gt_case
        base_cache = RECON / "heldout_baselines" / case

        out_rows = []
        for split, rows in splits.items():
            for window in sorted({w for _, w, _ in rows}, key=lambda w: (len(w), w)):
                sel = [(f, ref) for f, w, ref in rows if w == window]
                entries = [(label, [model.get(f) for f, _ in sel])]
                if not args.no_baselines:
                    base = "copy-nearest" if split == "interp" else "copy-last"
                    entries.append((base, [baseline_metrics(gt_dir, f, ref, scale, translate,
                                                            args.resolution, base_cache)
                                           for f, ref in sel]))
                for method, metrics in entries:
                    out_rows.append({"case": case, "tag": args.tag, "split": split, "window": window,
                                     "method": method, "train_step": step,
                                     "frames": " ".join(str(f) for f, _ in sel), **summarize(metrics)})

        summary = ROOT / "exp" / "heldout_summary" / f"{out_name}.csv"
        summary.parent.mkdir(parents=True, exist_ok=True)
        with summary.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=out_rows[0].keys())
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"\n[{case}] held-out results -> {summary}")
        print(f"  {'split':12s} {'window':6s} {'method':21s} {'n':>4s}  {'IoU':>7s}  {'CD':>9s}  {'NC':>7s}")
        for r in out_rows:
            print(f"  {r['split']:12s} {r['window']:6s} {r['method']:21s} {r['n']:4d}  {r['iou_mean']:7.4f}  "
                  f"{r['chamfer_mean']:9.6f}  {r['nc_mean']:7.4f}"
                  + (f"  ({r['n_failed']} no mesh)" if r["n_failed"] else "")
                  + (f"  ({r['nc_n_nan']} NC nan)" if r["nc_n_nan"] else ""), flush=True)
    if failed:
        print(f"[failed] {' '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
