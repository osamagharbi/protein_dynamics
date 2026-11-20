"""Shared training utilities for MAMBA-Flow (checkpoints, EMA, LR schedule, time budget)."""
import json
import math
import os
import random
import time

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_save(obj, path):
    """torch.save through a temp file: a walltime kill never leaves a torn checkpoint."""
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(path, obj):
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(path + ".tmp", path)


def rng_state():
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(), "python": random.getstate()}


def set_rng_state(state):
    if not state:
        return
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def lr_at(step, total, base, warmup, final_frac=0.01):
    """Linear warm-up, then cosine decay to final_frac * base."""
    if step < warmup:
        return base * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


class Deadline:
    """Wall-clock budget of one Slurm link (None = unlimited)."""

    def __init__(self, minutes=0.0):
        self.end = time.time() + 60 * minutes if minutes and minutes > 0 else None

    def reached(self, margin_s=0.0):
        return self.end is not None and time.time() + margin_s >= self.end

    def left_min(self):
        return float("inf") if self.end is None else (self.end - time.time()) / 60


def check_resume(ckpt, keys, current, what):
    """Refuse to resume a checkpoint trained with different settings."""
    bad = {k: (ckpt.get(k), current[k]) for k in keys if ckpt.get(k) != current[k]}
    if bad:
        raise SystemExit(f"[{what}] checkpoint settings differ from this run: "
                         + "; ".join(f"{k}: ckpt={a!r} now={b!r}" for k, (a, b) in bad.items())
                         + "  -> use another --tag, or delete the checkpoint to restart")
