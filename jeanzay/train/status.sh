#!/bin/bash
# status.sh -- progress of every training run submitted by submit.sh.
#   ./jeanzay/train/status.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$REPO/jeanzay/train_env/env_train.sh" >/dev/null 2>&1
squeue -u "$USER" -o "%.10i %.32j %.8T %.10M %.10L %R" 2>/dev/null | grep -E "JOBID|mamba_|dsr_" || true
echo
python - "$REPO" <<'PY'
import csv, re, sys
from pathlib import Path
import torch
repo = Path(sys.argv[1])
state, logs = repo / "jeanzay/train/state", repo / "jeanzay/train/logs"
exp, dexp = repo / "MAMBA-DSR/exp", repo / "DSR/exp/dsr_paper"

def epoch(p):
    try: return int(torch.load(p, map_location="cpu", weights_only=False).get("epoch", 0))
    except Exception: return None

def last_rate(name):
    """s/epoch from the newest '[epoch N/M] ... (X s elapsed)' pair in the newest log."""
    outs = sorted((logs / name).glob("*.out"), key=lambda p: p.stat().st_mtime) if (logs / name).is_dir() else []
    if not outs: return ""
    hits = re.findall(r"\[epoch (\d+)/\d+\].*?\(([\d.]+)s elapsed\)", outs[-1].read_text(errors="replace"))
    if len(hits) < 2: return ""
    (e0, t0), (e1, t1) = hits[0], hits[-1]
    return f"{(float(t1) - float(t0)) / max(1, int(e1) - int(e0)):.1f} s/ep"

print(f"{'run':40s} {'status':9s} {'stage A / DSR':>16s} {'B':>5s} {'speed':>10s}  result")
for d in sorted(state.glob("*")) if state.is_dir() else []:
    name = d.name
    method, rest = name.split("_", 1)
    st = "DONE" if (d / "DONE").exists() else "FAILED" if (d / "FAILED").exists() else "running"
    tag = re.search(r"(_s\d+)?(_smoke)?$", rest).group(0)
    case = rest[: len(rest) - len(tag)] if tag else rest
    res, a, b = "", "-", "-"
    if method == "mamba":
        ea = epoch(exp / f"{case}_paper_mamba_stageA{tag}/checkpoints/ckpt_latest.pth")
        eb = epoch(exp / f"{case}_paper_mamba_stageB{tag}/ckpt_latest.pth")
        a, b = (str(ea) if ea is not None else "-"), (str(eb) if eb is not None else "-")
        s = exp / "paper_summary" / f"{case}{tag}.csv"
        if s.is_file():
            r = next(csv.DictReader(s.open()))
            res = f"IoU {float(r['iou_mean']):.4f}  CD {float(r['chamfer_mean']):.6f}  NC {float(r['normal_consistency_mean']):.4f}"
    else:
        ck = sorted((dexp / case).glob("*/checkpoints/ckpt_latest.pth"), key=lambda p: p.stat().st_mtime)
        a = str(epoch(ck[-1])) if ck else "-"
        m = repo / "reconstructions" / f"DSR_{case}_{case}_paper_train_eval" / "metrics.json"
        if m.is_file():
            import json
            mean = json.loads(m.read_text())["mean"]
            res = f"IoU {mean['iou']:.4f}  CD {mean['chamfer']:.6f}  NC {mean['normal_consistency']:.4f}"
    print(f"{name:40s} {st:9s} {a:>16s} {b:>5s} {last_rate(name):>10s}  {res}")
PY
