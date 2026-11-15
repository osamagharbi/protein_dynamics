#!/bin/bash
# status.sh -- what is done / running / left for every run submitted by submit.sh.
#   ./jeanzay/train/status.sh        (read-only: safe while jobs are running)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$REPO/jeanzay/train_env/env_train.sh" >/dev/null 2>&1
python - "$REPO" <<'PY'
import csv, json, os, re, subprocess, sys
from pathlib import Path
import torch
repo = Path(sys.argv[1])
state, logs = repo / "jeanzay/train/state", repo / "jeanzay/train/logs"
exp, dexp, recon = repo / "MAMBA-DSR/exp", repo / "DSR/exp/dsr_paper", repo / "reconstructions"

# Slurm jobs per run name: running / pending (the rest of the chain).
queue = {}
try:
    out = subprocess.run(["squeue", "-h", "-u", os.environ["USER"], "-o", "%j|%T"],
                         capture_output=True, text=True, timeout=30).stdout
    for line in out.splitlines():
        name, st = line.split("|")
        q = queue.setdefault(name, {"RUNNING": 0, "PENDING": 0})
        q[st] = q.get(st, 0) + 1
except Exception:
    pass

def load(p):
    try: return torch.load(p, map_location="cpu", weights_only=False)
    except Exception: return None

def log_progress(name):
    """(latest Stage-A epoch, s/epoch) from the newest log that trained Stage A."""
    outs = sorted((logs / name).glob("*.out"), key=lambda p: p.stat().st_mtime, reverse=True) \
        if (logs / name).is_dir() else []
    for o in outs:
        hits = re.findall(r"\[epoch (\d+)/\d+\].*?\(([\d.]+)s elapsed\)", o.read_text(errors="replace"))
        if len(hits) >= 2:
            (e0, t0), (e1, t1) = hits[0], hits[-1]
            return int(e1), (float(t1) - float(t0)) / max(1, int(e1) - int(e0))
        if hits:
            return int(hits[-1][0]), None
    return None, None

def n_cached(d):
    return len(list((d / "frame_results").glob("frame_*.json"))) if (d / "frame_results").is_dir() else 0

all_names = sorted({p.name for base in (state, logs) if base.is_dir() for p in base.iterdir() if p.is_dir()})
names = [n for n in all_names if not n.startswith("heldout_")]
heldout = [n for n in all_names if n.startswith("heldout_")]
hdr = f"{'run':34s} {'phase':9s} {'stage A':>15s} {'s/ep':>6s} {'A left':>7s} {'eval':>9s} {'jobs R/PD':>9s}  result"
print(hdr); print("-" * len(hdr))
totals = {"DONE": 0, "FAILED": 0, "active": 0, "waiting": 0}
for name in names:
    method, rest = name.split("_", 1)
    tag = re.search(r"(_s\d+)?(_smoke)?$", rest).group(0)
    case = rest[: len(rest) - len(tag)] if tag else rest
    q = queue.get(name, {"RUNNING": 0, "PENDING": 0})
    jobs = f"{q['RUNNING']}/{q['PENDING']}"
    done, failed = (state / name / "DONE").exists(), (state / name / "FAILED").exists()
    a_txt, rate_txt, left_txt, ev_txt, res, phase = "-", "", "", "", "", "queued"

    if method == "mamba":
        ck = load(exp / f"{case}{'_paper_mamba_stageA'}{tag}/checkpoints/ckpt_latest.pth")
        end = (ck or {}).get("end_iter") or (20 if tag.endswith("_smoke") else 4000)
        ep_ck = (ck or {}).get("epoch", 0)
        ep_log, rate = log_progress(name)
        ep = max(ep_ck, ep_log or 0)
        if ck or ep_log:
            a_txt = f"{ep}/{end} {100 * ep / end:3.0f}%"
            phase = "stage A"
        if rate:
            rate_txt = f"{rate:.1f}"
            if ep < end:
                left_txt = f"{rate * (end - ep) / 3600:.1f}h"
        n_train = (ck or {}).get("num_frames")
        ev_dir = recon / f"MAMBA-DSR_{case}_{case}_paper_mamba_train_eval{tag}"
        if ck and ep_ck >= end:
            phase = "stage B"
            if (exp / f"{case}_paper_mamba_stageB{tag}/ckpt_latest.pth").is_file():
                phase = "eval"
        if n_train and (ev_dir / "frame_results").is_dir():
            ev_txt = f"{n_cached(ev_dir)}/{n_train}"
        s = exp / "paper_summary" / f"{case}{tag}.csv"
        if s.is_file():
            r = next(csv.DictReader(s.open()))
            res = (f"IoU {float(r['iou_mean']):.4f}  CD {float(r['chamfer_mean']):.6f}  "
                   f"NC {float(r['normal_consistency_mean']):.4f}")
    else:
        cks = sorted((dexp / case).glob("*/checkpoints/ckpt_latest.pth"), key=lambda p: p.stat().st_mtime)
        ck = load(cks[-1]) if cks else None
        if ck:
            ep = ck.get("epoch", 0)
            a_txt, phase = f"{ep}/4000 {100 * ep / 4000:3.0f}%", "DSR train"
            if ep >= 4000: phase = "eval"
        ev_dir = recon / f"DSR_{case}_{case}_paper_train_eval"
        if (ev_dir / "frame_results").is_dir():
            ev_txt = f"{n_cached(ev_dir)}"
        m = ev_dir / "metrics.json"
        if m.is_file():
            mean = json.loads(m.read_text())["mean"]
            res = f"IoU {mean['iou']:.4f}  CD {mean['chamfer']:.6f}  NC {mean['normal_consistency']:.4f}"

    if done:     phase = "DONE"; totals["DONE"] += 1
    elif failed: phase = "FAILED"; totals["FAILED"] += 1
    elif q["RUNNING"]: totals["active"] += 1
    elif q["PENDING"]:
        totals["waiting"] += 1
        if phase not in ("queued",): phase += "*"   # * = between links, next job pending
    else:
        phase = "STALLED" if phase != "queued" else phase   # unfinished, nothing queued -> resubmit
    print(f"{name:34s} {phase:9s} {a_txt:>15s} {rate_txt:>6s} {left_txt:>7s} {ev_txt:>9s} {jobs:>9s}  {res}")

print(f"\n{totals['DONE']} done, {totals['active']} running, {totals['waiting']} waiting in queue, "
      f"{totals['FAILED']} failed   |   * = between chain links   STALLED = unfinished with no job left "
      f"(resubmit)   'A left' = Stage-A hours at current speed")
if heldout:
    print("\nheld-out evaluation (results: cd MAMBA-DSR && python eval_heldout.py --report)")
    for name in heldout:
        q = queue.get(name, {"RUNNING": 0, "PENDING": 0})
        rest = name[len("heldout_"):]
        if rest.startswith("dsr_"):
            case = rest[len("dsr_"):]
            n = n_cached(recon / f"DSR_{case}_{case}_paper_heldout")
        else:
            m = re.fullmatch(r"(.+?)(_s\d+)?(_hold_t|_hold_all)?", rest)
            case, tag, mode = m.group(1), m.group(2) or "", m.group(3) or ""
            n = n_cached(recon / f"MAMBA-DSR_{case}_{case}_paper_mamba_heldout{tag}{mode}")
        ph = ("DONE" if (state / name / "DONE").exists() else "running" if q["RUNNING"]
              else "queued" if q["PENDING"] else "STALLED")
        print(f"  {name:34s} {ph:8s} frames evaluated {n:4d}   jobs R/PD {q['RUNNING']}/{q['PENDING']}")
for name in names:
    f = state / name / "FAILED"
    if f.exists():
        print(f"FAILED {name}: {f.read_text().strip()}  -> tail -40 {logs / name}/<newest>.out")
PY
