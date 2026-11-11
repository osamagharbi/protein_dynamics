#!/usr/bin/env python3
"""prep.py -- download and convert MD trajectories into DSR / MAMBA-DSR surface data.

Modes
-----
  selftest  PyMOL + MDAnalysis end-to-end check on a synthetic peptide (no data needed)
  download  fetch ONE raw source into RAW_ROOT, then index its datasets (frame counts,
            atom selection, XTC offset cache).                   [needs internet]
  plan      split the selected datasets into frame chunks -> one tasks TSV per source
  convert   convert ONE chunk (row --task-id of a tasks TSV) with --workers PyMOL processes
  finalize  validate every frame, compute DSR normalisation, write meta.json and a
            symlinked paper-protocol training view
  status    one-line summary per processed dataset

Output (MAMBA-DSR/paper_protocol.py reads exactly this layout):
  PROC_ROOT/<dataset>/gt/traj_<i>.obj          one protein-only molecular surface per frame
  PROC_ROOT/<dataset>/source.json              written by download (index step)
  PROC_ROOT/<dataset>/meta.json                written by finalize (provenance + QC)
  PROC_ROOT/<dataset>/views/paper_train/obj/   symlinks to the DSR-paper training frames

Surfaces are produced exactly like DSR/data_process.py (PyMOL show_as('surface') ->
OBJ), except that only ``Dataset.selection`` atoms (default: protein) are kept.
Every step is idempotent: finished frames/downloads are skipped on re-run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import registry as R  # noqa: E402

USER_AGENT = "dsr-data-prep/1.0 (research data download)"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def atlas_kwargs(args):
    return dict(atlas_replica=args.atlas_replica, atlas_stride=args.atlas_stride)


# ==========================================================================
# Downloads
# ==========================================================================
def md5sum(path: Path, chunk=1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch_url(url: str, dest: Path, md5: str | None = None, headers: dict | None = None,
              retries: int = 6, timeout: int = 120) -> Path:
    """Resumable download (HTTP Range on a .part file), md5-verified, atomic rename."""
    if dest.exists() and (md5 is None or md5sum(dest) == md5):
        log(f"  have {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                mode = "ab" if (have and r.status == 206) else "wb"
                total = r.headers.get("Content-Length")
                log(f"  GET {dest.name} ({'resume@' + str(have) if mode == 'ab' else 'fresh'}, "
                    f"{int(total) / 1e9:.2f} GB left)" if total else f"  GET {dest.name}")
                with open(part, mode) as f:
                    shutil.copyfileobj(r, f, length=1 << 22)
            if md5 is not None and md5sum(part) != md5:
                part.unlink()
                raise IOError(f"md5 mismatch for {dest.name}")
            os.replace(part, dest)
            return dest
        except (urllib.error.URLError, IOError, TimeoutError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
                raise RuntimeError(f"HTTP {e.code} for {url}") from e
            log(f"  attempt {attempt}/{retries} failed for {dest.name}: {e}")
            if attempt < retries:
                time.sleep(min(300, 10 * 2 ** attempt))
    raise RuntimeError(f"giving up on {url}")


def check_internet(url="https://pypi.org"):
    try:
        urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=20)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[download] no internet access from {os.uname().nodename} ({e}).\n"
                 "  Jean-Zay compute nodes are offline: run downloads on the prepost partition "
                 "or on a login node (DL_TARGET=login).")


def safe_members(tar: tarfile.TarFile, prefix: str):
    for m in tar.getmembers():
        p = Path(m.name)
        if p.is_absolute() or ".." in p.parts or not m.name.startswith(prefix):
            raise RuntimeError(f"unsafe tar member {m.name!r}")
        if not (m.isfile() or m.isdir()):
            continue
        yield m


def dl_mdanalysisdata(src, raw_root: Path):
    from MDAnalysisData import datasets as mdad
    (raw_root / "MDAnalysisData").mkdir(parents=True, exist_ok=True)  # parallel fetchers race on makedirs
    getattr(mdad, src["fetch"])(data_home=str(raw_root / "MDAnalysisData"))


def dl_dataverse(src, raw_root: Path):
    api = f"{src['server']}/api/datasets/:persistentId/?persistentId={src['doi']}"
    with open(fetch_url(api, raw_root / src["subdir"] / "_dataverse_listing.json")) as r:  # retried
        files = json.load(r)["data"]["latestVersion"]["files"]
    for f in files:
        df = f["dataFile"]
        md5 = df.get("md5") or (df.get("checksum") or {}).get("value")
        fetch_url(f"{src['server']}/api/access/datafile/{df['id']}",
                  raw_root / src["subdir"] / df["filename"], md5=md5)


def dl_dryad(src, raw_root: Path):
    d = raw_root / src["subdir"]
    archive = d / src["archive"]
    if not (archive.exists() and archive.stat().st_size == src["size"] and md5sum(archive) == src["md5"]):
        token = os.environ.get("DRYAD_TOKEN")
        if not token:
            sys.exit(f"[dryad] {archive} missing and DRYAD_TOKEN unset. Dryad rejects anonymous "
                     "scripted downloads. Either export DRYAD_TOKEN (Dryad API account), or upload "
                     "your local copy:\n  rsync -avP ~/projects/protein_dynamics/datasets/DSR/DRY_AD_MD/"
                     f"trajectories.tar.gz <jz>:{d}/")
        fetch_url(f"https://datadryad.org/api/v2/files/{src['file_id']}/download", archive,
                  md5=src["md5"], headers={"Authorization": f"Bearer {token}"})
    marker = d / ".extracted"
    if not marker.exists():
        log(f"  extracting {archive.name}")
        with tarfile.open(archive) as t:
            t.extractall(d, members=list(safe_members(t, "trajectories")))
        marker.write_text(time.strftime("%F %T"))


def atlas_ids(args, raw_root: Path) -> list:
    if args.atlas_ids:
        p = Path(args.atlas_ids)
        ids = p.read_text().split() if p.is_file() else args.atlas_ids.replace(",", " ").split()
        return ids
    if args.atlas_limit <= 0:
        return []
    lst = raw_root / "ATLAS" / "ATLAS_pdb.txt"
    if not lst.exists():
        fetch_url(R.SOURCES["atlas"]["list_url"], lst)
    return lst.read_text().split()[: args.atlas_limit]


def dl_atlas_one(pid: str, raw_root: Path, replicas: list, keep_zip: bool):
    src = R.SOURCES["atlas"]
    d = raw_root / src["subdir"] / pid
    want = [f"{pid}.pdb"] + [f"{pid}_prod_R{r}_fit.xtc" for r in replicas]
    if all((d / w).exists() for w in want):
        return pid, "have"
    z = fetch_url(src["zip_url"].format(id=pid), d / f"{pid}_protein.zip")
    with zipfile.ZipFile(z) as zf:
        names = set(zf.namelist())
        for w in want:          # extract only expected flat names (no path traversal)
            if w not in names:
                raise RuntimeError(f"{pid}: {w} not in zip ({sorted(names)})")
            with zf.open(w) as fin, open(d / (w + ".tmp"), "wb") as fout:
                shutil.copyfileobj(fin, fout, length=1 << 22)
            os.replace(d / (w + ".tmp"), d / w)
    if not keep_zip:
        z.unlink()
    return pid, "ok"


def dl_atlas(args, raw_root: Path):
    ids = atlas_ids(args, raw_root)
    if not ids:
        sys.exit("[atlas] no ATLAS ids selected (set ATLAS_LIMIT or ATLAS_IDS)")
    replicas = sorted({args.atlas_replica})
    failed = []
    with ThreadPoolExecutor(max_workers=args.atlas_parallel) as ex:
        futs = {ex.submit(dl_atlas_one, i, raw_root, replicas, args.keep_zip): i for i in ids}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                pid, st = fut.result()
                log(f"  [{n}/{len(ids)}] {pid}: {st}")
            except Exception as e:  # noqa: BLE001
                failed.append(futs[fut])
                log(f"  [{n}/{len(ids)}] {futs[fut]}: FAILED {e}")
    if failed:
        log(f"[atlas] {len(failed)} failed: {' '.join(failed)}  (re-run to retry)")


def resolve_files(ds: R.Dataset, raw_root: Path):
    tops = [raw_root / t for t in ds.topology if (raw_root / t).exists()]
    traj = raw_root / ds.trajectory
    if not tops or not traj.exists():
        missing = [str(raw_root / t) for t in ds.topology] if not tops else [str(traj)]
        raise FileNotFoundError(f"{ds.name}: missing {missing}")
    return tops[0], traj


def output_frames(ds: R.Dataset, n_traj: int) -> list:
    """[(out_idx, orig_idx)] for the dataset's crop/stride, clamped to the trajectory."""
    n_avail = max(0, math.ceil((n_traj - ds.start) / ds.stride))
    n = n_avail if ds.n_frames is None else min(ds.n_frames, n_avail)
    return [(k, ds.start + k * ds.stride) for k in range(n)]


def index_dataset(ds: R.Dataset, raw_root: Path, proc_root: Path) -> dict:
    """Open once: count frames, check the selection, build/cache XTC offsets."""
    import warnings
    warnings.filterwarnings("ignore")
    import MDAnalysis as mda
    top, traj = resolve_files(ds, raw_root)
    u = mda.Universe(str(top), str(traj))
    sel = u.select_atoms(ds.selection)
    if sel.n_atoms == 0:
        raise RuntimeError(f"{ds.name}: selection {ds.selection!r} matched 0 atoms")
    frames = output_frames(ds, len(u.trajectory))
    info = {
        "dataset": ds.name, "source": ds.source, "topology": str(top), "trajectory": str(traj),
        "trajectory_bytes": traj.stat().st_size, "n_traj_frames": len(u.trajectory),
        "n_atoms_total": int(u.atoms.n_atoms), "selection": ds.selection,
        "n_atoms_selected": int(sel.n_atoms), "start": ds.start, "stride": ds.stride,
        "n_output_frames": len(frames), "dt_ps": float(getattr(u.trajectory, "dt", float("nan"))),
        "note": ds.note, "indexed_at": time.strftime("%F %T"),
    }
    write_json(proc_root / ds.name / "source.json", info)
    log(f"  indexed {ds.name}: {info['n_traj_frames']} frames, {sel.n_atoms}/{u.atoms.n_atoms} "
        f"atoms selected -> {len(frames)} output frames")
    return info


def cmd_download(args):
    raw_root, proc_root = Path(args.raw_root), Path(args.proc_root)
    src = R.SOURCES[args.source]
    log(f"[download] source={args.source} kind={src['kind']} raw_root={raw_root}")
    if src["kind"] != "manual":
        check_internet()
    t0 = time.time()
    {"mdanalysisdata": lambda: dl_mdanalysisdata(src, raw_root),
     "dataverse": lambda: dl_dataverse(src, raw_root),
     "dryad": lambda: dl_dryad(src, raw_root),
     "atlas": lambda: dl_atlas(args, raw_root),
     "manual": lambda: None}[src["kind"]]()
    log(f"[download] fetched in {time.time() - t0:.0f}s; indexing datasets")
    ids = atlas_ids(args, raw_root) if args.source == "atlas" else ()
    dsets = [d for d in R.select(args.datasets, ids, **atlas_kwargs(args)) if d.source == args.source]
    bad = []
    for ds in dsets:
        try:
            index_dataset(ds, raw_root, proc_root)
        except Exception as e:  # noqa: BLE001
            bad.append(ds.name)
            log(f"  INDEX FAILED {ds.name}: {e}")
    write_json(raw_root / ".status" / f"{args.source}.json",
               {"source": args.source, "finished": time.strftime("%F %T"),
                "datasets": [d.name for d in dsets], "index_failed": bad})
    if bad:
        sys.exit(f"[download] index failed for {bad}")


# ==========================================================================
# Planning
# ==========================================================================
def known_frames(ds: R.Dataset, raw_root: Path, proc_root: Path):
    """Output-frame count for planning: registry crop, else index (if raw is staged)."""
    sj = proc_root / ds.name / "source.json"
    if sj.exists():
        return json.loads(sj.read_text())["n_output_frames"]
    if ds.n_frames is not None:
        return ds.n_frames          # upper bound; convert clamps to the real length
    try:
        return index_dataset(ds, raw_root, proc_root)["n_output_frames"]
    except FileNotFoundError:
        return None


def cmd_plan(args):
    raw_root, proc_root, plan_dir = Path(args.raw_root), Path(args.proc_root), Path(args.plan_dir)
    plan_dir.mkdir(parents=True, exist_ok=True)
    ids = atlas_ids(args, raw_root)
    dsets = R.select(args.datasets, ids, **atlas_kwargs(args))
    by_src, skipped = {}, []
    for ds in dsets:
        n = known_frames(ds, raw_root, proc_root)
        if not n:
            skipped.append(ds.name)
            continue
        rows = by_src.setdefault(ds.source, [])
        for c, a in enumerate(range(0, n, args.chunk_size)):
            rows.append((ds.name, c, a, min(a + args.chunk_size, n)))
    with open(plan_dir / "sources.tsv", "w", newline="") as fs:
        w = csv.writer(fs, delimiter="\t", lineterminator="\n")
        w.writerow(["source", "kind", "datasets", "tasks_tsv", "n_tasks", "n_frames"])
        for src, rows in by_src.items():
            tsv = plan_dir / f"tasks_{src}.tsv"
            with open(tsv, "w", newline="") as ft:
                wt = csv.writer(ft, delimiter="\t", lineterminator="\n")
                wt.writerow(["dataset", "chunk", "out_start", "out_stop"])
                wt.writerows(rows)
            names = sorted({r[0] for r in rows})
            w.writerow([src, R.SOURCES[src]["kind"], ",".join(names), str(tsv), len(rows),
                        sum(r[3] - r[2] for r in rows)])
    if ids:
        (plan_dir / "atlas_ids.txt").write_text("\n".join(ids) + "\n")
    tot_t = sum(len(r) for r in by_src.values())
    tot_f = sum(r[3] - r[2] for rows in by_src.values() for r in rows)
    print(f"PLAN datasets={sum(len({r[0] for r in v}) for v in by_src.values())} "
          f"sources={len(by_src)} tasks={tot_t} frames<={tot_f} chunk={args.chunk_size}")
    for name in skipped:
        print(f"SKIP {name}: raw data not staged (manual source) -- see README")


# ==========================================================================
# Conversion
# ==========================================================================
def _convert_worker(job):
    name, raw_root, proc_root, frames, tmp_root, akw = job
    import warnings
    warnings.filterwarnings("ignore")
    import MDAnalysis as mda
    try:                                     # headless PyMOL, one instance per process
        import pymol
        pymol.finish_launching(["pymol", "-qc"])
    except Exception:  # noqa: BLE001
        pass
    from pymol import cmd

    ds = R.get_dataset(name, **akw)
    top, traj = resolve_files(ds, Path(raw_root))
    u = mda.Universe(str(top), str(traj))
    sel = u.select_atoms(ds.selection)
    gt = Path(proc_root) / name / "gt"
    tmpd = gt / ".tmp"
    tmpd.mkdir(parents=True, exist_ok=True)
    pdb = Path(tmp_root) / f"{name}_{os.getpid()}.pdb"
    done = skipped = 0
    for out_idx, orig in frames:
        dest = gt / f"traj_{out_idx}.obj"
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue
        u.trajectory[orig]
        sel.write(str(pdb))
        cmd.delete("all")
        cmd.load(str(pdb), "frame")
        cmd.show_as("surface", "frame")
        tmp = tmpd / f"traj_{out_idx}.{os.getpid()}.obj"
        cmd.save(str(tmp), "frame")
        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError(f"{name}: PyMOL wrote an empty surface for frame {orig}")
        os.replace(tmp, dest)
        done += 1
    pdb.unlink(missing_ok=True)
    return done, skipped


def cmd_convert(args):
    raw_root, proc_root = Path(args.raw_root), Path(args.proc_root)
    with open(args.tasks) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not 0 <= args.task_id < len(rows):
        sys.exit(f"--task-id {args.task_id} out of range [0,{len(rows)})")
    row = rows[args.task_id]
    name, a, b = row["dataset"], int(row["out_start"]), int(row["out_stop"])
    ds = R.get_dataset(name, **atlas_kwargs(args))
    sj = proc_root / name / "source.json"
    if sj.exists():
        n_traj = json.loads(sj.read_text())["n_traj_frames"]
    else:
        n_traj = index_dataset(ds, raw_root, proc_root)["n_traj_frames"]
    frames = [f for f in output_frames(ds, n_traj) if a <= f[0] < b]
    if not frames:
        log(f"[convert] {name} chunk {row['chunk']}: beyond trajectory end, nothing to do")
        return
    w = max(1, min(args.workers, len(frames)))
    parts = [frames[i::w] for i in range(w)]     # interleaved: balances size drift over time
    tmp_root = os.environ.get("JOBSCRATCH") or os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir()
    log(f"[convert] {name} out[{a},{b}) = {len(frames)} frames, {w} PyMOL workers, tmp={tmp_root}")
    t0 = time.time()
    jobs = [(name, str(raw_root), str(proc_root), p, tmp_root, atlas_kwargs(args)) for p in parts]
    with mp.get_context("spawn").Pool(w) as pool:
        res = pool.map(_convert_worker, jobs)
    done, skipped = sum(r[0] for r in res), sum(r[1] for r in res)
    dt = time.time() - t0
    log(f"[convert] {name} chunk {row['chunk']}: {done} converted, {skipped} already present, "
        f"{dt:.0f}s ({dt / max(done, 1):.2f}s/frame wall)")


# ==========================================================================
# Finalize / QC
# ==========================================================================
def _obj_stats(path: str):
    """vertex count + bbox of a PyMOL OBJ, without trimesh (fast, low memory)."""
    import numpy as np
    n, lo, hi = 0, np.full(3, np.inf), np.full(3, -np.inf)
    buf = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                buf.append(line[2:])
                if len(buf) >= 200000:
                    a = np.loadtxt(buf, dtype=np.float64, usecols=(0, 1, 2), ndmin=2)
                    lo, hi, n = np.minimum(lo, a.min(0)), np.maximum(hi, a.max(0)), n + len(a)
                    buf = []
    if buf:
        a = np.loadtxt(buf, dtype=np.float64, usecols=(0, 1, 2), ndmin=2)
        lo, hi, n = np.minimum(lo, a.min(0)), np.maximum(hi, a.max(0)), n + len(a)
    return n, lo.tolist(), hi.tolist(), os.path.getsize(path)


def finalize_dataset(ds: R.Dataset, proc_root: Path, workers: int) -> dict:
    import numpy as np
    root = proc_root / ds.name
    sj = root / "source.json"
    if not sj.exists():
        return {"dataset": ds.name, "status": "not_indexed"}
    src = json.loads(sj.read_text())
    frames = output_frames(ds, src["n_traj_frames"])
    gt = root / "gt"
    paths = [gt / f"traj_{k}.obj" for k, _ in frames]
    missing = [k for (k, _), p in zip(frames, paths) if not (p.exists() and p.stat().st_size > 0)]
    present = [(k, str(p)) for (k, _), p in zip(frames, paths) if k not in set(missing)]
    with mp.get_context("spawn").Pool(max(1, workers)) as pool:
        stats = dict(zip([k for k, _ in present], pool.map(_obj_stats, [p for _, p in present])))
    empty = [k for k, s in stats.items() if s[0] == 0]
    ext = {k: max(np.subtract(s[2], s[1])) for k, s in stats.items() if s[0] > 0}
    med = float(np.median(list(ext.values()))) if ext else float("nan")
    outliers = sorted(k for k, e in ext.items() if e > 1.5 * med)   # broken PBC / unfolding

    train = []
    if ds.train_step:
        stop = min(ds.train_stop or len(frames), len(frames))
        train = [k for k in range(0, stop, ds.train_step)]
    norm_frames = [k for k in (train or list(ext)) if k in ext]
    if norm_frames:   # identical rule to MAMBA-DSR/run_multicase.normalization()
        lo = np.min([stats[k][1] for k in norm_frames], axis=0)
        hi = np.max([stats[k][2] for k in norm_frames], axis=0)
        scale, translate = float(max(hi - lo) * 1.1), float(lo.min())
    else:
        scale = translate = None

    view = root / "views" / "paper_train" / "obj"
    if train and not missing:
        if view.parent.exists():
            shutil.rmtree(view.parent)
        view.mkdir(parents=True)
        for k in train:
            (view / f"traj_{k}.obj").symlink_to(Path("../../../gt") / f"traj_{k}.obj")
    shutil.rmtree(gt / ".tmp", ignore_errors=True)

    status = "complete" if not missing and not empty else "incomplete"
    meta = {
        **src, "status": status, "n_expected": len(frames), "n_present": len(present),
        "missing_frames": missing, "empty_frames": empty, "bbox_outlier_frames": outliers,
        "median_bbox_extent_A": med,
        "frame_map": {"out_to_orig": "orig = start + out * stride", "start": ds.start, "stride": ds.stride},
        "paper_train_rule": {"stop": ds.train_stop, "step": ds.train_step},
        "paper_train_indices": train,
        "dsr_normalization": {"s": scale, "t": translate,
                              "computed_on": "paper_train_frames" if train else "all_frames"},
        "train_view": str(view.parent) if train and not missing else None,
        "mean_vertices": float(np.mean([s[0] for s in stats.values()])) if stats else 0.0,
        "total_bytes": int(sum(s[3] for s in stats.values())),
        "finalized_at": time.strftime("%F %T"),
    }
    write_json(root / "meta.json", meta)
    return meta


def cmd_finalize(args):
    proc_root = Path(args.proc_root)
    ids = atlas_ids(args, Path(args.raw_root)) if "atlas" in args.datasets else ()
    rc = 0
    for ds in R.select(args.datasets, ids, **atlas_kwargs(args)):
        m = finalize_dataset(ds, proc_root, args.workers)
        if m.get("status") == "not_indexed":
            log(f"[finalize] {ds.name}: not downloaded/indexed -- skipped")
            continue
        flag = "" if m["status"] == "complete" else "  <-- INCOMPLETE"
        log(f"[finalize] {ds.name}: {m['n_present']}/{m['n_expected']} frames, "
            f"{len(m['empty_frames'])} empty, {len(m['bbox_outlier_frames'])} bbox outliers, "
            f"{m['total_bytes'] / 1e9:.2f} GB, s={m['dsr_normalization']['s']}{flag}")
        rc |= m["status"] != "complete"
    sys.exit(rc)


def cmd_status(args):
    proc_root = Path(args.proc_root)
    print(f"{'dataset':28s} {'indexed':>8s} {'gt_obj':>7s} {'expected':>8s} {'final':>11s} {'GB':>6s}")
    for d in sorted(p for p in proc_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        sj, mj = d / "source.json", d / "meta.json"
        exp = json.loads(sj.read_text())["n_output_frames"] if sj.exists() else "-"
        n = len(list((d / "gt").glob("traj_*.obj"))) if (d / "gt").exists() else 0
        m = json.loads(mj.read_text()) if mj.exists() else {}
        print(f"{d.name:28s} {'yes' if sj.exists() else 'no':>8s} {n:>7d} {exp!s:>8s} "
              f"{m.get('status', '-'):>11s} {m.get('total_bytes', 0) / 1e9:>6.2f}")


def cmd_selftest(args):
    """Synthetic end-to-end check of the exact convert path (PyMOL -> MDAnalysis -> PyMOL OBJ)."""
    import warnings
    warnings.filterwarnings("ignore")
    import MDAnalysis as mda
    import pymol
    pymol.finish_launching(["pymol", "-qc"])
    from pymol import cmd
    d = Path(tempfile.mkdtemp(prefix="dsr_selftest_"))
    cmd.fab("ACDEFGHIKLMNPQRSTVWY", "pep", ss=1)
    cmd.save(str(d / "pep.pdb"), "pep")
    u = mda.Universe(str(d / "pep.pdb"))
    sel = u.select_atoms("protein")
    sel.write(str(d / "sel.pdb"))
    cmd.delete("all")
    cmd.load(str(d / "sel.pdb"), "frame")
    cmd.show_as("surface", "frame")
    cmd.save(str(d / "traj_0.obj"), "frame")
    n, lo, hi, size = _obj_stats(str(d / "traj_0.obj"))
    shutil.rmtree(d)
    ok = n > 1000
    print(f"[selftest] protein atoms={sel.n_atoms} surface vertices={n} bytes={size} "
          f"-> {'PASSED' if ok else 'FAILED'}")
    import MDAnalysisData  # noqa: F401  (download dependency present)
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["selftest", "download", "plan", "convert", "finalize", "status"])
    ap.add_argument("--raw-root", default=os.environ.get("RAW_ROOT", "datasets/raw"))
    ap.add_argument("--proc-root", default=os.environ.get("PROC_ROOT", "datasets/DSR_processed"))
    ap.add_argument("--datasets", default=os.environ.get("DATASETS", "paper"),
                    help=f"groups {R.GROUPS} and/or dataset names")
    ap.add_argument("--source", choices=list(R.SOURCES))
    ap.add_argument("--plan-dir")
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--tasks")
    ap.add_argument("--task-id", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--atlas-limit", type=int, default=int(os.environ.get("ATLAS_LIMIT", 0)))
    ap.add_argument("--atlas-ids", default=os.environ.get("ATLAS_IDS"),
                    help="file with ATLAS pdb_chain ids, or comma list (overrides --atlas-limit)")
    ap.add_argument("--atlas-replica", type=int, default=int(os.environ.get("ATLAS_REPLICA", 1)))
    ap.add_argument("--atlas-stride", type=int, default=int(os.environ.get("ATLAS_STRIDE", 100)))
    ap.add_argument("--atlas-parallel", type=int, default=int(os.environ.get("ATLAS_PARALLEL", 4)))
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()
    if args.mode == "download" and not args.source:
        ap.error("download needs --source")
    if args.mode == "plan" and not args.plan_dir:
        ap.error("plan needs --plan-dir")
    if args.mode == "convert" and (not args.tasks or args.task_id < 0):
        ap.error("convert needs --tasks and --task-id")
    {"selftest": cmd_selftest, "download": cmd_download, "plan": cmd_plan, "convert": cmd_convert,
     "finalize": cmd_finalize, "status": cmd_status}[args.mode](args)


if __name__ == "__main__":
    main()
