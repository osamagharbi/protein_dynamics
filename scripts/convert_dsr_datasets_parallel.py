#!/usr/bin/env python3
"""Convert locally available MD trajectories into DSR surface OBJ datasets.

Each worker is a separate process because PyMOL is not thread-safe.  The
default worker count is intentionally conservative for the 16-CPU/15-GB RAM
WSL machine used by this project.  Missing topologies and compressed archives
are reported and skipped rather than guessed or overwritten.
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "datasets" / "DSR"
DEFAULT_OUTPUT = PROJECT / "datasets" / "DSR_processed"


def local_jobs():
    """Return jobs with a local trajectory and topology already available."""
    return [
        {
            "name": "abeta_wt",
            "trajectory": RAW / "Abeta_500ns/abeta_1_40_mono_phystempLong.dcd",
            "topology": RAW / "Abeta_500ns/abeta_1_40_mono.pdb",
        },
        {
            "name": "abeta_e22g",
            "trajectory": RAW / "Abeta_500ns/abeta_E22G_1_40_mono_phystempLong.dcd",
            "topology": RAW / "Abeta_500ns/abeta_E22G_1_40_mono.pdb",
        },
        {
            "name": "adk_equilibrium",
            "trajectory": RAW / "MDAnalysisData/adk_equilibrium/1ake_007-nowater-core-dt240ps.dcd",
            "topology": RAW / "MDAnalysisData/adk_equilibrium/adk4AKE.psf",
        },
        {
            "name": "adk_dims",
            "trajectory": RAW / "MDAnalysisData/adk_transitions_DIMS/DIMS.tar.gz",
            "topology": None,
            "status": "archive_requires_extraction",
        },
        {
            "name": "adk_froda",
            "trajectory": RAW / "MDAnalysisData/adk_transitions_FRODA/FRODA.tar.gz",
            "topology": None,
            "status": "archive_requires_extraction",
        },
        {
            "name": "ifabp_water",
            "trajectory": RAW / "MDAnalysisData/ifabp_water/rmsfit_ifabp_water_1.dcd",
            "topology": RAW / "MDAnalysisData/ifabp_water/ifabp_water.psf",
        },
        {
            "name": "nhaa_equilibrium",
            "trajectory": RAW / "MDAnalysisData/nhaa_equilibrium/NhaA_non_water.xtc",
            "topology": RAW / "MDAnalysisData/nhaa_equilibrium/NhaA_non_water.gro",
        },
        {
            "name": "yiip_equilibrium",
            "trajectory": RAW / "MDAnalysisData/yiip_equilibrium/YiiP_system_90ns_center.xtc",
            "topology": RAW / "MDAnalysisData/yiip_equilibrium/YiiP_system.pdb",
        },
    ]


def expand_jobs():
    jobs = local_jobs()
    gpcr_dir = RAW / "GPCRmd"
    for trajectory in sorted(gpcr_dir.glob("*.xtc")):
        jobs.append({
            "name": f"gpcr_{trajectory.stem}",
            "trajectory": trajectory,
            "topology": None,
            "status": "missing_topology",
        })
    jobs.append({
        "name": "dry_ad_md",
        "trajectory": RAW / "DRY_AD_MD/trajectories.tar.gz",
        "topology": None,
        "status": "archive_requires_extraction",
    })
    return jobs


def validate_jobs(jobs):
    ready, skipped = [], []
    for job in jobs:
        trajectory = Path(job["trajectory"])
        topology = Path(job["topology"]) if job.get("topology") else None
        if job.get("status"):
            skipped.append({**job, "reason": job["status"]})
        elif not trajectory.exists():
            skipped.append({**job, "reason": "missing_trajectory"})
        elif topology is None or not topology.exists():
            skipped.append({**job, "reason": "missing_topology"})
        else:
            ready.append({**job, "trajectory": str(trajectory), "topology": str(topology)})
    return ready, skipped


def convert_job(job):
    """Run one trajectory conversion in an isolated process."""
    import mdtraj as md
    from pymol import cmd

    output = Path(job["output"])
    gt_dir = output / "gt"
    train_dir = output / "train"
    gt_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    # PyMOL is launched headlessly inside this worker, not in the parent.
    try:
        import pymol
        pymol.finish_launching(["pymol", "-cq"])
    except Exception:
        pass

    start_frame = int(job.get("start_frame", 0))
    frame_index = start_frame
    converted_frames = 0
    started = time.time()
    with tempfile.TemporaryDirectory(prefix=f"dsr_{job['name']}_") as temp_dir:
        temp_pdb = Path(temp_dir) / "frame.pdb"
        for chunk in md.iterload(
            job["trajectory"], top=job["topology"], chunk=1, skip=start_frame
        ):
            for local_index in range(chunk.n_frames):
                frame = chunk[local_index]
                frame.save_pdb(str(temp_pdb))
                object_name = f"frame_{frame_index}"
                cmd.load(str(temp_pdb), object_name)
                cmd.show_as("surface", object_name)
                gt_path = gt_dir / f"traj_{frame_index}.obj"
                cmd.save(str(gt_path), object_name)
                if frame_index % job["train_interval"] == 0:
                    cmd.save(str(train_dir / f"traj_{frame_index}.obj"), object_name)
                cmd.delete(object_name)
                frame_index += 1
                converted_frames += 1
                if job["progress_every"] and frame_index % job["progress_every"] == 0:
                    print(
                        f"[{job['name']}] processed {frame_index} frames "
                        f"({time.time() - started:.0f}s)",
                        flush=True,
                    )
                if job.get("max_frames") and frame_index >= job["max_frames"]:
                    break
            if job.get("max_frames") and frame_index >= job["max_frames"]:
                break

    return {
        "name": job["name"],
        "frames": converted_frames,
        "first_frame": start_frame,
        "last_frame": frame_index - 1,
        "output": str(output),
        "seconds": round(time.time() - started, 1),
        "status": "done",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--only", nargs="+", default=None,
                        help="Convert only the named dataset(s), for example "
                             "--only adk_equilibrium.")
    parser.add_argument("--workers", type=int, default=2,
                        help="Concurrent PyMOL conversion processes; use 1-2 on 16-GB RAM.")
    parser.add_argument("--train-interval", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional limit for a smoke test.")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Start reading at this zero-based trajectory frame. "
                             "Use this to resume an interrupted conversion without "
                             "reprocessing earlier frames.")
    parser.add_argument("--progress-every", type=int, default=25,
                        help="Print worker progress every N frames; use 0 to disable.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List ready and skipped jobs without converting anything.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.start_frame < 0:
        parser.error("--start-frame must be non-negative")
    if args.max_frames is not None and args.max_frames <= args.start_frame:
        parser.error("--max-frames must be greater than --start-frame when resuming")

    output_root = Path(args.output_root).resolve()
    all_jobs = expand_jobs()
    if args.only:
        wanted = set(args.only)
        known = {job["name"] for job in all_jobs}
        unknown = sorted(wanted - known)
        if unknown:
            parser.error(f"unknown dataset(s): {', '.join(unknown)}")
        all_jobs = [job for job in all_jobs if job["name"] in wanted]
    jobs, skipped = validate_jobs(all_jobs)
    for job in jobs:
        job["output"] = str(output_root / job["name"])
        job["train_interval"] = args.train_interval
        job["max_frames"] = args.max_frames
        job["start_frame"] = args.start_frame
        job["progress_every"] = args.progress_every

    print(f"ready jobs: {len(jobs)}")
    for job in jobs:
        print(f"  READY  {job['name']}: {job['trajectory']}")
    print(f"skipped jobs: {len(skipped)}")
    for job in skipped:
        print(f"  SKIP   {job['name']}: {job['reason']}")
    if args.dry_run:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    print(f"starting {len(jobs)} jobs with {args.workers} worker(s)")
    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(convert_job, jobs)
    for result in results:
        print(f"DONE {result['name']}: {result['frames']} frames in {result['seconds']} s")

    manifest = {
        "output_root": str(output_root),
        "train_interval": args.train_interval,
        "start_frame": args.start_frame,
        "max_frames": args.max_frames,
        "ready_results": results,
        "skipped": skipped,
    }
    with open(output_root / "conversion_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"manifest: {output_root / 'conversion_manifest.json'}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
