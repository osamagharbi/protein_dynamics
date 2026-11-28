"""Evaluation of MAMBA-Flow with exactly the MAMBA-DSR evaluator's conventions: DSR's
128^3 grid over [-1.2, 1.2]^3 (normalized coordinates), skimage marching cubes as in
DSR's get_mesh, OBJ export/reload, then DSR's IoU / Chamfer / normal consistency against
the ground-truth mesh normalized with the same s, t. Metrics run in a process pool.

Training frames  decode the Stage-1 latents of the training frames ("MAMBA-Flow", the
                 counterpart of the training-frame table). -> exp/flow_summary/<case><tag>.csv
Held-out frames  (splits of eval_heldout.py) four predictions per frame:
  MAMBA-Flow          mean signed-distance field of S Stage-2 samples (ensemble)
  MAMBA-Flow [sample] one Stage-2 sample (realism of a single draw)
  MAMBA-Flow [ref]    the latent baseline alone: context latents linearly interpolated
                      (interp) / last context latent (future) -- what Stage 2 adds is
                      MAMBA-Flow minus this
  MAMBA-Flow [AE]     oracle: the held-out ground truth encoded and decoded (ceiling of
                      Stage 1; uses the frame's own surface, so it is NOT a prediction)
  MAMBA-Flow [ctx-mean]  mean latent of the context frames the prediction sees (no Stage 2)
  MAMBA-Flow [case-mean] mean latent of ALL training frames of the protein ("average shape",
                      no Stage 2): if MAMBA-Flow ~ this far ahead, Stage 2 learned mean
                      reversion rather than dynamics
"""
import csv
import json
import math
import multiprocessing as mp
import os
import shutil
import sys

import numpy as np
import torch

from flow import metrics_worker
from flow.common import write_json
from flow.data import case_protocol, load_frame_model_coords
from flow.dynamics import LatentBank, load_dynamics, make_example, sample_targets
from flow.train_shape import load_shape
from run_multicase import DSR_DIR, ROOT

RECON = ROOT.parent / "reconstructions"
FLOW_SUMMARY = ROOT / "exp" / "flow_summary"
LABELS = {"ens": "MAMBA-Flow", "sample": "MAMBA-Flow [sample]", "ref": "MAMBA-Flow [ref]",
          "ae": "MAMBA-Flow [AE]", "ctxmean": "MAMBA-Flow [ctx-mean]",
          "casemean": "MAMBA-Flow [case-mean]"}


# ----------------------------------------------------------------------------- metrics pool
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class MetricPool:
    """Metric jobs in spawned single-threaded worker processes (one CPU core each)."""

    def __init__(self, workers):
        self.pool = None
        if workers > 0:
            saved = {k: os.environ.get(k) for k in THREAD_VARS}
            os.environ.update({k: "1" for k in THREAD_VARS})     # inherited at worker start-up
            try:
                self.pool = mp.get_context("spawn").Pool(workers, initializer=metrics_worker.init_worker)
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        self.pending, self.max_pending = [], max(2, 2 * workers)

    def submit(self, job, callback):
        if self.pool is None:
            callback(metrics_worker.metrics_job(job))
            return
        self.pending.append((self.pool.apply_async(metrics_worker.metrics_job, (job,)), callback))
        while len(self.pending) > self.max_pending:
            self._drain_one()

    def _drain_one(self):
        res, cb = self.pending.pop(0)
        cb(res.get())

    def drain(self):
        while self.pending:
            self._drain_one()

    def close(self):
        self.drain()
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


# ----------------------------------------------------------------------------- meshing
class Mesher:
    """Decoder evaluation on DSR's uniform grid + marching cubes as lib.plots.get_mesh."""

    def __init__(self, shape_model, resolution=128, chunk=65536):
        sys.path.insert(0, str(DSR_DIR))
        from lib.plots import get_grid_uniform
        self.model, self.res, self.chunk = shape_model, resolution, chunk
        self.grid = get_grid_uniform(resolution)
        self.points = self.grid["grid_points"].cuda()
        for blk in self.model.query_cross:              # inference: no double backward needed
            blk.attn.twice = False

    @torch.no_grad()
    def sdf(self, z, anchors, center):
        """z (S, K, d) latents -> (S, res^3) signed distances on the grid (float32, CPU)."""
        h = self.model.tokens(z.cuda().float(), anchors.cuda().expand(len(z), -1, -1))
        c = torch.as_tensor(center, device="cuda", dtype=torch.float32)
        per_call = max(1024, self.chunk // len(z))
        out = [self.model.query(h, (p - c)[None].expand(len(z), -1, -1)).cpu()
               for p in torch.split(self.points, per_call)]
        return torch.cat(out, dim=1).numpy()

    def marching_cubes(self, values):
        """Flat grid values -> (verts, faces, normals, values) in normalized coords, or None."""
        from skimage import measure
        if values.min() > 0 or values.max() < 0:
            return None
        x, y, z = self.grid["xyz"]
        vol = values.astype(np.float64).reshape(y.shape[0], x.shape[0], z.shape[0]).transpose([1, 0, 2])
        dx = x[2] - x[1]
        verts, faces, normals, vals = measure.marching_cubes(volume=vol, level=0.0, spacing=(dx, dx, dx))
        return verts + np.array([x[0], y[0], z[0]]), faces, normals, vals


def _fingerprint(run_dir, extra):
    fp = dict(extra)
    for part in ("shape", "dynamics"):
        p = run_dir / part / "model_ema.pth"
        if p.is_file():
            st = p.stat()
            fp[part] = [str(p), st.st_mtime, st.st_size]
    return fp


def _read_cache(path, fp):
    if path.is_file():
        c = json.loads(path.read_text())
        if c.get("fingerprint") == fp:
            return c["metrics"]
    return None


def _finite_mean(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return float(np.mean(vals)) if vals else math.nan


# ----------------------------------------------------------------------------- training frames
def eval_train(cfg, run_dir, run_name, cases, tag, seed, max_frames=0):
    shape, _ = load_shape(run_dir)
    mesher = Mesher(shape, cfg["resolution"])
    pool = MetricPool(cfg["metric_workers"])
    fp = _fingerprint(run_dir, {"resolution": cfg["resolution"], "kind": "train"})
    fp.pop("dynamics", None)                      # training-frame reconstructions use Stage 1 only
    for case in cases:
        lat = torch.load(run_dir / "latents" / f"{case}.pt", map_location="cpu", weights_only=False)
        spec, _, train, _ = case_protocol(case)
        frames = list(lat["frames"])
        if max_frames and len(frames) > max_frames:
            frames = [frames[i] for i in np.linspace(0, len(frames) - 1, max_frames).round().astype(int)]
        root = RECON / f"MAMBA-Flow_{case}_{run_name}_train"
        cache = root / "frame_results"
        cache.mkdir(parents=True, exist_ok=True)
        results = {}
        todo = []
        for f in frames:
            got = _read_cache(cache / f"frame_{f:05d}.json", fp)
            if got is not None:
                results[f] = got
            else:
                todo.append(f)
        print(f"[eval-train] {case}: {len(frames)} training frames ({len(frames) - len(todo)} cached)", flush=True)
        row_of = {f: i for i, f in enumerate(lat["frames"])}

        def done(f):
            def cb(m):
                results[f] = m["ens"]
                write_json(cache / f"frame_{f:05d}.json", {"fingerprint": fp, "metrics": m["ens"]})
            return cb

        for f in todo:
            vol = mesher.sdf(lat["z"][row_of[f]][None], lat["anchors"], lat["center"])[0]
            pool.submit((str(spec.full_dir / f"traj_{f}.obj"), lat["s"], lat["t"], cfg["resolution"],
                         {"ens": mesher.marching_cubes(vol)}), done(f))
        pool.drain()
        rows = [{"frame_index": f, **(results[f] or {})} for f in frames]
        write_json(root / "metrics.json", {"case": case, "run": run_name, "kind": "train_frames",
                                           "resolution": cfg["resolution"], "results": rows})
        ok = [r for r in rows if "iou_mean" in r]
        summary = {"case": case, "seed": seed, "tag": tag, "method": LABELS["ens"],
                   "protocol": "dsr_paper_reconstruction_train_frames", "train_frames": len(train),
                   "evaluated_frames": len(frames), "metric_space": "normalized",
                   "iou_mean": _finite_mean([r["iou_mean"] for r in ok]),
                   "chamfer_mean": _finite_mean([r["chamfer_mean"] for r in ok]),
                   "normal_consistency_mean": _finite_mean([r["normal_consistency_mean"] for r in ok]),
                   "nc_frames_nan": sum(not math.isfinite(r["normal_consistency_mean"]) for r in ok),
                   "n_failed": len(rows) - len(ok), "run": run_name}
        path = FLOW_SUMMARY / f"{case}{tag}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=summary.keys())
            w.writeheader()
            w.writerow(summary)
        print(f"[eval-train] {case}: IoU {summary['iou_mean']:.4f}  CD {summary['chamfer_mean']:.6f}  "
              f"NC {summary['normal_consistency_mean']:.4f}  ({summary['n_failed']} no mesh) -> {path}",
              flush=True)
    pool.close()


# ----------------------------------------------------------------------------- held-out frames
def _interp_start(n, pos_a, c):
    return int(np.clip(pos_a + 1 - c // 2, 0, n - c))


def _interp_example(off, n, pos_a, frac, c):
    start = _interp_start(n, pos_a, c)
    ctx_tau = np.arange(c) - (c - 1) / 2
    return make_example(off, start + np.arange(c), ctx_tau, [pos_a - start + frac - (c - 1) / 2])


def eval_heldout(cfg, run_dir, run_name, case, seed, max_per_split=0):
    """Returns {label: {(split, frame): metrics dict or None}} and writes metrics.json."""
    from eval_heldout import frame_splits
    spec, available, train, step = case_protocol(case)
    splits, missing, _ = frame_splits(train, available, max_per_split)
    shape, _ = load_shape(run_dir)
    dyn, dck = load_dynamics(run_dir)
    if case not in dck["cases"]:
        raise SystemExit(f"[heldout] {case} was not in the training set of {run_name}: {dck['cases']}")
    bank = LatentBank(run_dir, dck["cases"], stats=dck["stats"])
    ci = bank.cases.index(case)
    off, n, frames = bank.off[ci], bank.n[ci], bank.frames[ci]
    if frames != train:
        raise SystemExit(f"[heldout] {case}: latent frames differ from the protocol training list")
    pos = {f: j for j, f in enumerate(frames)}
    lat = torch.load(run_dir / "latents" / f"{case}.pt", map_location="cpu", weights_only=False)
    anchors, center, s, t = lat["anchors"], lat["center"], lat["s"], lat["t"]
    c_test = min(cfg["ctx_test"], n)
    root = RECON / f"MAMBA-Flow_{case}_{run_name}_heldout"
    cache = root / "frame_results"
    cache.mkdir(parents=True, exist_ok=True)
    fp = _fingerprint(run_dir, {"resolution": cfg["resolution"], "samples": cfg["n_samples"],
                                "flow_steps": cfg["flow_steps"], "ctx_test": c_test, "seed": seed})
    fp_ae = _fingerprint(run_dir, {"resolution": cfg["resolution"], "kind": "ae",
                                   "draws": cfg["encode_draws"], "n_enc": cfg["n_enc"]})
    fp_ae.pop("dynamics", None)
    fp_base = _fingerprint(run_dir, {"resolution": cfg["resolution"], "kind": "latent-means",
                                     "ctx_test": c_test})
    fp_base.pop("dynamics", None)

    # sequences: interp = one target per sequence (context centred on it); future = all
    # targets of the split jointly after the last C training frames.
    seqs = []                                  # (split, [frames], example)
    ctx_start = {}                             # (split, frame) -> first context position
    for split, rows in splits.items():
        if not rows:
            continue
        if split == "interp":
            for f, _, a in rows:
                b = frames[pos[a] + 1]
                seqs.append((split, [f], _interp_example(off, n, pos[a], (f - a) / (b - a), c_test)))
                ctx_start[(split, f)] = _interp_start(n, pos[a], c_test)
        else:
            tgt = [f for f, _, _ in rows]
            ex = make_example(off, np.arange(n - c_test, n), np.arange(c_test) - (c_test - 1),
                              (np.asarray(tgt) - frames[-1]) / step)
            seqs.append((split, tgt, ex))
            ctx_start.update({(split, f): n - c_test for f in tgt})
    print(f"[heldout] {case}: train {len(train)} frames step {step}; "
          + ", ".join(f"{k} {len(v)} (no GT {missing[k]})" for k, v in splits.items())
          + f"; {cfg['n_samples']} samples x {cfg['flow_steps']} flow steps, context {c_test}", flush=True)

    results = {}
    ae_results = {}
    base_results = {}
    pool = MetricPool(cfg["metric_workers"])
    mesher = Mesher(shape, cfg["resolution"])
    gt_path = lambda f: str(spec.full_dir / f"traj_{f}.obj")

    def store(key):
        def cb(m):
            split, f = key
            if split == "ae":
                ae_results[f] = m["ae"]
                write_json(cache / f"ae_{f:05d}.json", {"fingerprint": fp_ae, "metrics": m})
            elif split.startswith("base:"):
                base_results[(split[5:], f)] = m
                write_json(cache / f"base_{split[5:]}_{f:05d}.json", {"fingerprint": fp_base, "metrics": m})
            else:
                results[key] = m
                write_json(cache / f"{split}_{f:05d}.json", {"fingerprint": fp, "metrics": m})
        return cb

    # 1) Stage-2 predictions (ensemble / single sample / latent baseline)
    todo = []
    for split, tgts, ex in seqs:
        got = [_read_cache(cache / f"{split}_{f:05d}.json", fp) for f in tgts]
        for f, g in zip(tgts, got):
            if g is not None:
                results[(split, f)] = g
        if any(g is None for g in got):
            todo.append((split, tgts, ex))
    groups = {}
    for item in todo:                          # same layout -> one batched sampling call
        groups.setdefault((len(item[2][0]), int(item[2][2].sum())), []).append(item)
    for items in groups.values():
        for i in range(0, len(items), cfg["seq_batch"]):
            chunk = items[i:i + cfg["seq_batch"]]
            samples, ref = sample_targets(dyn, bank, [it[2] for it in chunk], [ci] * len(chunk),
                                          [step] * len(chunk), dck["res_scale"], cfg["n_samples"],
                                          cfg["flow_steps"], seed=seed + i)
            for bi, (split, tgts, _) in enumerate(chunk):
                for mi, f in enumerate(tgts):
                    if (split, f) in results:
                        continue
                    vols = mesher.sdf(samples[:, bi, mi], anchors, center)
                    vref = mesher.sdf(ref[bi, mi][None], anchors, center)[0]
                    preds = {"ens": mesher.marching_cubes(vols.mean(0)),
                             "sample": mesher.marching_cubes(vols[0]),
                             "ref": mesher.marching_cubes(vref)}
                    pool.submit((gt_path(f), s, t, cfg["resolution"], preds), store((split, f)))
            print(f"[heldout] {case}: sampled {min(i + cfg['seq_batch'], len(items))}/{len(items)} "
                  f"sequences of this layout", flush=True)

    # 2) autoencoder oracle, once per frame
    all_frames = sorted({f for _, tgts, _ in seqs for f in tgts})
    for f in all_frames:
        got = _read_cache(cache / f"ae_{f:05d}.json", fp_ae)
        if got is not None:
            ae_results[f] = got["ae"]
            continue
        acc = 0
        for d in range(cfg["encode_draws"]):
            p, nrm = load_frame_model_coords(case, f, cfg["n_enc"], seed=1000 * d)
            with torch.no_grad():
                acc = acc + shape.encode(torch.from_numpy(p)[None].cuda(), torch.from_numpy(nrm)[None].cuda(),
                                         anchors[None].cuda())[0].float()
        vol = mesher.sdf((acc / cfg["encode_draws"]).cpu(), anchors, center)[0]
        pool.submit((gt_path(f), s, t, cfg["resolution"], {"ae": mesher.marching_cubes(vol)}), store(("ae", f)))

    # 3) latent-mean baselines (Stage 1 only): context mean and whole-trajectory mean
    z_all = lat["z"].float()
    case_mesh, ctx_mesh = None, {}
    for split, tgts, _ in seqs:
        for f in tgts:
            got = _read_cache(cache / f"base_{split}_{f:05d}.json", fp_base)
            if got is not None:
                base_results[(split, f)] = got
                continue
            if case_mesh is None:
                case_mesh = mesher.marching_cubes(mesher.sdf(z_all.mean(0)[None], anchors, center)[0])
            st = ctx_start[(split, f)]
            if st not in ctx_mesh:
                ctx_mesh[st] = mesher.marching_cubes(
                    mesher.sdf(z_all[st:st + c_test].mean(0)[None], anchors, center)[0])
            pool.submit((gt_path(f), s, t, cfg["resolution"], {"ctxmean": ctx_mesh[st], "casemean": case_mesh}),
                        store((f"base:{split}", f)))
    pool.close()

    out = {label: {} for label in LABELS.values()}
    rows = []
    for split, tgts, _ in seqs:
        for f in tgts:
            m = dict(results[(split, f)], ae=ae_results.get(f), **base_results.get((split, f), {}))
            for key, label in LABELS.items():
                out[label][(split, f)] = m.get(key) or {}
                rows.append({"split": split, "frame_index": f, "method": label, **(m.get(key) or {})})
    write_json(root / "metrics.json", {"case": case, "run": run_name, "kind": "heldout",
                                       "fingerprint": fp, "results": rows})
    shutil.rmtree(root / "tmp", ignore_errors=True)
    return out


def load_heldout(case, run_name):
    """{label: {(split, frame): metrics}} from a finished held-out evaluation (for --report)."""
    path = RECON / f"MAMBA-Flow_{case}_{run_name}_heldout" / "metrics.json"
    out = {}
    for r in json.loads(path.read_text())["results"]:
        out.setdefault(r["method"], {})[(r["split"], int(r["frame_index"]))] = r
    return out
