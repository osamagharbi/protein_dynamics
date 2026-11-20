"""Data for MAMBA-Flow.

Frames, normalization and splits are exactly those of the MAMBA-DSR / DSR runs:
  * training frames = the paper-protocol training list of each case (paper_protocol.py),
  * coordinates     = DSR's normalization ((x - t) / s * 2 - 1, s and t from the training
                      frames, run_multicase.cached_normalization), so meshes are scored in
                      the same space and on the same grid as every other method,
  * points/normals  = DSR's loader (trimesh.load(obj), duplicate vertices merged, vertex
                      normals), local sigma = distance to the 50th neighbour (DSR sampler).
The model additionally works in a per-case centred frame: model = normalized - center,
center = bounding-box centre of the training frames (no per-frame superposition: the
evaluation needs predictions in each frame's original pose, which is unknown for unseen
frames).

Cache: exp/flow/cache/<case>/frames.npz (+ meta.json, anchors_K<k>.npy), built once.
"""
import json
import multiprocessing as mp
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from run_multicase import CASES, ROOT, cached_normalization, ensure_gt_link, obj_indices

FLOW_EXP = ROOT / "exp" / "flow"
CACHE_DIR = FLOW_EXP / "cache"
CACHE_VERSION = 1
GRID_BOUND = 1.2          # DSR's marching-cubes grid spans [-1.2, 1.2]^3 in normalized coords


def case_protocol(case):
    """(spec, available GT frame set, training frame list, training step) as the launchers use."""
    spec = CASES[case]
    available = obj_indices(spec.full_dir)
    required = spec.required_frames()
    full = min(len(available), required) if required else len(available)
    train = sorted(spec.train_indices(full))
    step = Counter(b - a for a, b in zip(train, train[1:])).most_common(1)[0][0]
    return spec, set(available), train, step


def prepared_cases():
    """Every case whose data was prepared by jeanzay/data_prep (meta.json), as submit.sh does."""
    root = ROOT.parent / "datasets" / "DSR_processed"
    return [c for c in CASES if (root / c / "meta.json").is_file()]


def load_dsr_points(path):
    """DSR's RawDataset loader: merged vertices + vertex normals (float32)."""
    import trimesh
    mesh = trimesh.load(str(path), "obj")
    return np.asarray(mesh.vertices, np.float32), np.asarray(mesh.vertex_normals, np.float32)


def normalize(v, s, t):
    return ((v - t) / s) * 2.0 - 1.0


def local_sigma(points):
    """Distance to the 50th nearest neighbour, as DSR's unpack_point_samples."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    k = min(51, len(points))
    return np.concatenate([tree.query(p, k)[0][:, -1] for p in np.array_split(points, 20) if len(p)])


def _load_one(job):
    path, s, t = job
    v, n = load_dsr_points(path)
    v = normalize(v, s, t).astype(np.float32)
    return v, n.astype(np.float16), local_sigma(v).astype(np.float16)


def fps(points, k, seed=0):
    """Farthest point sampling (numpy)."""
    rng = np.random.default_rng(seed)
    idx = np.empty(k, np.int64)
    idx[0] = rng.integers(len(points))
    d = np.full(len(points), np.inf)
    for i in range(1, k):
        d = np.minimum(d, ((points - points[idx[i - 1]]) ** 2).sum(1))
        idx[i] = int(d.argmax())
    return points[idx]


def build_cache(case, workers=None, force=False):
    """Load the training frames of one case into exp/flow/cache/<case>/ (idempotent)."""
    spec, available, train, step = case_protocol(case)
    s, t = cached_normalization(spec, train)
    out = CACHE_DIR / case
    meta_path = out / "meta.json"
    want = {"version": CACHE_VERSION, "case": case, "train_frames": train, "s": float(s), "t": float(t)}
    if not force and meta_path.is_file() and (out / "frames.npz").is_file():
        meta = json.loads(meta_path.read_text())
        if all(meta.get(k) == v for k, v in want.items()):
            return meta
    missing = [i for i in train if i not in available]
    if missing:
        raise FileNotFoundError(f"{case}: training frames without a mesh: {missing[:10]}")
    ensure_gt_link(spec)
    workers = workers or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    jobs = [(str(spec.full_dir / f"traj_{i}.obj"), float(s), float(t)) for i in train]
    print(f"[cache] {case}: {len(train)} training frames (step {step}) with {workers} workers", flush=True)
    with mp.get_context("spawn").Pool(max(1, min(workers, len(jobs)))) as pool:
        loaded = pool.map(_load_one, jobs, chunksize=1)
    allv = np.concatenate([v for v, _, _ in loaded])
    center = ((allv.min(0) + allv.max(0)) / 2).astype(np.float32)
    counts = np.array([len(v) for v, _, _ in loaded], np.int64)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "frames.tmp.npz"
    np.savez(tmp, pts=(allv - center).astype(np.float16),
             nrm=np.concatenate([n for _, n, _ in loaded]),
             sig=np.concatenate([g for _, _, g in loaded]),
             offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
             frames=np.asarray(train, np.int64))
    os.replace(tmp, out / "frames.npz")
    meta = dict(want, step=int(step), center=center.tolist(), n_points=int(counts.sum()),
                points_per_frame=[int(counts.min()), int(np.median(counts)), int(counts.max())],
                gt_case=spec.gt_case, full_dir=str(spec.full_dir),
                extent_model=[(allv - center).min(0).tolist(), (allv - center).max(0).tolist()])
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"[cache] {case}: {counts.sum():,} points, per frame min/median/max "
          f"{meta['points_per_frame']}, centre {np.round(center, 3).tolist()}", flush=True)
    return meta


def case_anchors(case, k, seed=0):
    """K anchors: farthest point sampling over points pooled from all training frames
    (covers the volume swept by the trajectory). Fixed per case and K."""
    path = CACHE_DIR / case / f"anchors_K{k}.npy"
    if path.is_file():
        return np.load(path)
    z = np.load(CACHE_DIR / case / "frames.npz")
    pts, off = z["pts"].astype(np.float32), z["offsets"]
    rng = np.random.default_rng(seed)
    nf = len(off) - 1
    pick = np.unique(np.linspace(0, nf - 1, min(nf, 60)).round().astype(int))
    pool = np.concatenate([pts[off[f]:off[f + 1]][rng.choice(off[f + 1] - off[f],
                                                             min(1500, off[f + 1] - off[f]),
                                                             replace=False)] for f in pick])
    anchors = fps(pool, k, seed).astype(np.float32)
    np.save(path, anchors)
    return anchors


class FrameBank:
    """All cached training frames of the selected cases, resident on one device."""

    def __init__(self, cases, k, device="cuda", case_alpha=0.5):
        self.cases, self.device = list(cases), device
        pts, nrm, sig, off, fcase, fraw = [], [], [], [], [], []
        self.meta, anchors, centers = {}, [], []
        base = 0
        for ci, case in enumerate(self.cases):
            meta = json.loads((CACHE_DIR / case / "meta.json").read_text())
            z = np.load(CACHE_DIR / case / "frames.npz")
            pts.append(z["pts"]); nrm.append(z["nrm"]); sig.append(z["sig"])
            off.append(z["offsets"][:-1] + base)
            base += int(z["offsets"][-1])
            fcase.append(np.full(len(z["frames"]), ci)); fraw.append(z["frames"])
            anchors.append(case_anchors(case, k))
            centers.append(meta["center"])
            self.meta[case] = meta
        cat = lambda xs, dt: torch.as_tensor(np.concatenate(xs)).to(device=device, dtype=dt)
        self.pts, self.nrm = cat(pts, torch.float16), cat(nrm, torch.float16)
        self.sig = cat(sig, torch.float16)
        self.off = cat(off, torch.long)
        self.cnt = torch.diff(torch.cat([self.off, torch.tensor([base], device=device)]))
        self.fcase, self.fraw = cat(fcase, torch.long), cat(fraw, torch.long)
        self.anchors = torch.as_tensor(np.stack(anchors), device=device)          # (C, K, 3)
        self.centers = torch.as_tensor(np.array(centers, np.float32), device=device)
        n_case = torch.bincount(self.fcase, minlength=len(self.cases)).float()
        self.frame_w = (n_case ** (case_alpha - 1.0))[self.fcase]  # case prob ~ n_frames^alpha
        self.case_slices = {c: (int((self.fcase < i).sum()), int((self.fcase <= i).sum()))
                            for i, c in enumerate(self.cases)}

    def __len__(self):
        return len(self.off)

    def _idx(self, fids, n, gen=None):
        r = torch.rand(len(fids), n, device=self.device, generator=gen)
        return self.off[fids, None] + (r * self.cnt[fids, None]).long().clamp_max(self.cnt[fids, None] - 1)

    def frame_points(self, fids, n, gen=None):
        """n random surface points + normals of each frame (with replacement), float32."""
        i = self._idx(fids, n, gen)
        return self.pts[i].float(), self.nrm[i].float()

    def training_batch(self, b, n_enc, q_surf, q_local, q_global, gen=None):
        fids = torch.multinomial(self.frame_w, b, replacement=True, generator=gen)
        pts, nrm = self.frame_points(fids, n_enc, gen)
        xs, ns = self.frame_points(fids, q_surf, gen)
        i = self._idx(fids, q_local, gen)
        xl = self.pts[i].float() + torch.randn(b, q_local, 3, device=self.device, generator=gen) \
            * self.sig[i].float().unsqueeze(-1)
        box = (torch.rand(b, q_global, 3, device=self.device, generator=gen) * 2 - 1) * GRID_BOUND
        xg = box - self.centers[self.fcase[fids]][:, None]
        return dict(fids=fids, pts=pts, nrm=nrm, anchors=self.anchors[self.fcase[fids]],
                    xs=xs, ns=ns, xo=torch.cat([xl, xg], dim=1))


def load_frame_model_coords(case, frame, n, seed=0):
    """Any frame of a case (e.g. a held-out ground-truth frame for the autoencoder oracle):
    n points + normals in the model frame of that case, deterministic subset."""
    meta = json.loads((CACHE_DIR / case / "meta.json").read_text())
    v, nrm = load_dsr_points(Path(meta["full_dir"]) / f"traj_{frame}.obj")
    v = normalize(v, meta["s"], meta["t"]) - np.asarray(meta["center"], np.float32)
    idx = np.random.default_rng(seed + frame).integers(0, len(v), n)
    return v[idx].astype(np.float32), nrm[idx].astype(np.float32)
