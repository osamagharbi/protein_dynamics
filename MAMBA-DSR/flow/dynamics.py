"""Stage 2: masked flow matching over sequences of latent token grids.

A training example is a short sequence of frames of ONE case, sorted by time: observed
context frames (clean latents) and target frames (generated). Time positions tau are in
units of the context spacing, so a task at a coarse stride emulates a finer one:

  forecast  context = C consecutive training frames, targets = the next H frames
            (tau = 1..H)                                    -> future_step at test time
  fine      context = C frames at stride s (s = 2..step), targets = the next H frames of the
            full training sequence (tau = 1/s, 2/s, ...)    -> future_raw (tau = k/step)
  interp    context = C frames at stride s (2..4), targets = frames inside random gaps
            (fractional tau)                                -> interp midpoints (tau = 0.5)

Targets are modelled as residuals from a fixed latent baseline: linear interpolation of the
context latents at the target's tau (= the last context latent past the end, i.e. latent
copy-last). Flow matching (rectified flow, logit-normal flow time) on those residuals.
The global condition includes log(raw frames per tau unit) of the example.
"""
import copy
import time

import numpy as np
import torch

from flow.common import (Deadline, atomic_save, check_resume, ema_update, lr_at, rng_state,
                         set_rng_state, write_json)
from flow.nets import SpaceTimeFlow, count_params

RESUME_KEYS = ("model", "cases", "steps", "batch", "tasks")


class LatentBank:
    """Normalized latents of every training frame of every case, on the GPU."""

    def __init__(self, run_dir, cases, device="cuda", stats=None):
        self.cases, self.device = list(cases), device
        zs, self.off, self.n, self.step, self.frames, anchors = [], [], [], [], [], []
        base = 0
        for case in self.cases:
            d = torch.load(run_dir / "latents" / f"{case}.pt", map_location="cpu", weights_only=False)
            zs.append(d["z"].float())
            self.off.append(base); self.n.append(len(d["z"])); base += len(d["z"])
            self.step.append(int(d["step"])); self.frames.append(list(d["frames"]))
            anchors.append(d["anchors"].float())
        z = torch.cat(zs)
        if stats is None:
            flat = z.reshape(-1, z.shape[-1])
            stats = {"mean": flat.mean(0), "std": flat.std(0).clamp_min(1e-4)}
        self.mean, self.std = stats["mean"].to(device), stats["std"].to(device)
        self.stats = {"mean": stats["mean"].cpu(), "std": stats["std"].cpu()}
        self.Z = ((z.to(device) - self.mean) / self.std).contiguous()       # (N, K, d)
        self.anchors = torch.stack(anchors).to(device)                     # (cases, K, 3)
        self.case_w = np.asarray(self.n, float)

    def denorm(self, z):
        return z * self.std + self.mean


def make_example(off, ctx_pos, ctx_tau, tgt_tau, tgt_pos=None):
    """Arrays of one sequence sorted by tau: global latent index, tau, is_target, and the
    reference (baseline) interpolation indices/weight."""
    ctx_pos, ctx_tau, tgt_tau = np.asarray(ctx_pos), np.asarray(ctx_tau, float), np.asarray(tgt_tau, float)
    c, m = len(ctx_tau), len(tgt_tau)
    j = np.searchsorted(ctx_tau, tgt_tau, side="right") - 1
    lo, hi = np.clip(j, 0, c - 1), np.clip(j + 1, 0, c - 1)
    span = ctx_tau[hi] - ctx_tau[lo]
    w = np.where(span > 0, (tgt_tau - ctx_tau[lo]) / np.where(span > 0, span, 1.0), 0.0)
    g_ctx = off + ctx_pos
    g_tgt = off + np.asarray(tgt_pos) if tgt_pos is not None else np.full(m, off)
    tau = np.concatenate([ctx_tau, tgt_tau])
    order = np.argsort(tau, kind="stable")
    return (np.concatenate([g_ctx, g_tgt])[order], tau[order],
            np.concatenate([np.zeros(c, bool), np.ones(m, bool)])[order],
            np.concatenate([g_ctx, g_ctx[lo]])[order], np.concatenate([g_ctx, g_ctx[hi]])[order],
            np.concatenate([np.zeros(c), w])[order])


class TaskSampler:
    def __init__(self, bank: LatentBank, tasks, alpha=0.5, seed=0):
        self.b, self.t, self.alpha = bank, tasks, alpha
        self.rng = np.random.default_rng(seed)

    def _case(self, need_fn):
        ok = [i for i in range(len(self.b.cases)) if need_fn(i)]
        if not ok:
            return None
        w = self.b.case_w[ok] ** self.alpha
        return ok[self.rng.choice(len(ok), p=w / w.sum())]

    def batch(self, size):
        r, t = self.rng, self.t
        for _ in range(100):
            kind = r.choice(["forecast", "fine", "interp"], p=t["probs"])
            ex, cases, dts = [], [], []
            if kind == "forecast":
                c, h = r.integers(t["ctx_min"], t["ctx_max"] + 1), r.integers(1, t["horizon"] + 1)
                for _ in range(size):
                    ci = self._case(lambda i: self.b.n[i] >= c + h)
                    if ci is None:
                        break
                    j0 = r.integers(0, self.b.n[ci] - c - h + 1)
                    ex.append(make_example(self.b.off[ci], j0 + np.arange(c), np.arange(c) - (c - 1),
                                           np.arange(1, h + 1), j0 + c + np.arange(h)))
                    cases.append(ci); dts.append(self.b.step[ci])
            elif kind == "fine":
                c, h = r.integers(3, t["fine_ctx_max"] + 1), r.integers(1, t["horizon"] + 1)
                for _ in range(size):
                    ci = self._case(lambda i: self.b.n[i] >= 2 * (c - 1) + h + 1)
                    if ci is None:
                        break
                    s_max = max(2, min(self.b.step[ci], (self.b.n[ci] - h - 1) // (c - 1)))
                    s = int(self.b.step[ci]) if r.random() < 0.5 else int(r.integers(2, s_max + 1))
                    s = min(s, s_max)
                    j0 = r.integers(0, self.b.n[ci] - (c - 1) * s - h)
                    last = j0 + (c - 1) * s
                    ex.append(make_example(self.b.off[ci], j0 + s * np.arange(c), np.arange(c) - (c - 1),
                                           np.arange(1, h + 1) / s, last + np.arange(1, h + 1)))
                    cases.append(ci); dts.append(s * self.b.step[ci])
            else:
                c = r.integers(4, t["ctx_max"] + 1)
                m = r.integers(1, min(t["interp_targets"], c - 1) + 1)
                for _ in range(size):
                    s = int(r.choice([2, 3, 4], p=[0.6, 0.2, 0.2]))
                    ci = self._case(lambda i: self.b.n[i] >= (c - 1) * 2 + 1)
                    if ci is None:
                        break
                    s = min(s, (self.b.n[ci] - 1) // (c - 1))
                    j0 = r.integers(0, self.b.n[ci] - (c - 1) * s)
                    gaps = np.sort(r.choice(c - 1, m, replace=False))
                    u = r.integers(1, s, m)
                    pos = s * gaps + u
                    ex.append(make_example(self.b.off[ci], j0 + s * np.arange(c), np.arange(c) - (c - 1) / 2,
                                           pos / s - (c - 1) / 2, j0 + pos))
                    cases.append(ci); dts.append(s * self.b.step[ci])
            if len(ex) == size:
                return ex, cases, dts
        raise RuntimeError("could not sample a task batch: are the trajectories long enough?")


def to_tensors(bank, examples, cases, dts):
    dev = bank.device
    st = lambda i, dt: torch.as_tensor(np.stack([e[i] for e in examples]), device=dev, dtype=dt)
    gidx, tau, is_t = st(0, torch.long), st(1, torch.float32), st(2, torch.bool)
    rlo, rhi, rw = st(3, torch.long), st(4, torch.long), st(5, torch.float32)
    z = bank.Z[gidx]
    ref = torch.lerp(bank.Z[rlo], bank.Z[rhi], rw[..., None, None])
    return dict(z=z, ref=ref, tau=tau, is_t=is_t, anchors=bank.anchors[torch.as_tensor(cases, device=dev)],
                log_dt=torch.log(torch.as_tensor(dts, device=dev, dtype=torch.float32)))


def flow_loss(model, b, res_scale, amp):
    x1 = (b["z"] - b["ref"]) / res_scale
    n = x1.shape[0]
    u = torch.sigmoid(torch.randn(n, device=x1.device))           # logit-normal flow time
    x0 = torch.randn_like(x1)
    xu = (1 - u)[:, None, None, None] * x0 + u[:, None, None, None] * x1
    mask = b["is_t"][..., None, None]
    state = torch.where(mask, xu, b["z"])
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        v = model(state, b["ref"], b["anchors"], b["tau"], b["is_t"], u, b["log_dt"])
    err = ((v.float() - (x1 - x0)) ** 2) * mask
    return err.sum() / (mask.sum() * x1.shape[2] * x1.shape[3])


def estimate_res_scale(sampler, bank, batches=200, size=8):
    sq, cnt = 0.0, 0
    for _ in range(batches):
        b = to_tensors(bank, *sampler.batch(size))
        d = (b["z"] - b["ref"])[b["is_t"]]
        sq += float((d ** 2).sum()); cnt += d.numel()
    return float(np.sqrt(sq / max(cnt, 1)))


def train_dynamics(cfg, run_dir, cases, deadline: Deadline):
    out = run_dir / "dynamics"
    ckpt_path, final_path = out / "ckpt_latest.pth", out / "model_ema.pth"
    lat_time = max((run_dir / "latents" / f"{c}.pt").stat().st_mtime for c in cases)
    settings = {"model": cfg["dyn_model"], "cases": list(cases), "steps": cfg["dyn_steps"],
                "batch": cfg["dyn_batch"], "tasks": cfg["tasks"]}
    if final_path.is_file() and final_path.stat().st_mtime > lat_time:
        check_resume(torch.load(final_path, map_location="cpu", weights_only=False),
                     ("model", "cases"), settings, "dynamics")
        print("[dynamics] complete -- skipped", flush=True)
        return True
    bank = LatentBank(run_dir, cases)
    sampler = TaskSampler(bank, cfg["tasks"], cfg["case_alpha"], cfg["seed"])
    model = SpaceTimeFlow(**cfg["dyn_model"], grad_checkpoint=cfg["dyn_grad_ckpt"]).cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["dyn_lr"], betas=(0.9, 0.99),
                            weight_decay=cfg["dyn_wd"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["dyn_amp"])
    step, log = 0, []
    if ckpt_path.is_file() and ckpt_path.stat().st_mtime > lat_time:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        check_resume(ck, RESUME_KEYS, settings, "dynamics")
        model.load_state_dict(ck["model_sd"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        set_rng_state(ck["rng"]); sampler.rng.bit_generator.state = ck["sampler_rng"]
        step, log, res_scale = ck["step"], ck["log"], ck["res_scale"]
        print(f"[dynamics] resumed at step {step}/{cfg['dyn_steps']}", flush=True)
    else:
        res_scale = estimate_res_scale(sampler, bank)
        print(f"[dynamics] SpaceTimeFlow {count_params(model) / 1e6:.1f}M parameters "
              f"({cfg['dyn_model']['temporal']} over time), {bank.Z.shape[0]} latent frames, "
              f"residual scale {res_scale:.4f}, {cfg['dyn_steps']} steps x {cfg['dyn_batch']} sequences",
              flush=True)

    def save():
        atomic_save({**settings, "model_sd": model.state_dict(), "ema": ema.state_dict(),
                     "opt": opt.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(),
                     "sampler_rng": sampler.rng.bit_generator.state, "step": step, "log": log,
                     "res_scale": res_scale}, ckpt_path)
        write_json(out / "loss_log.json", log)

    t0, t_save, acc, n_acc = time.time(), time.time(), 0.0, 0
    while step < cfg["dyn_steps"]:
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg["dyn_steps"], cfg["dyn_lr"], cfg["dyn_warmup"])
        loss = flow_loss(model, to_tensors(bank, *sampler.batch(cfg["dyn_batch"])), res_scale, cfg["dyn_amp"])
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            print(f"[dynamics] non-finite loss at step {step} -- skipped", flush=True)
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(opt)
        scaler.update()
        ema_update(ema, model, min(cfg["ema"], (1 + step) / (10 + step)))
        step += 1
        acc += loss.item(); n_acc += 1
        if step % cfg["log_every"] == 0 or step == cfg["dyn_steps"]:
            rate = (time.time() - t0) / n_acc
            log.append({"step": step, "loss": acc / n_acc, "lr": opt.param_groups[0]["lr"], "s_per_step": rate})
            print(f"[dynamics {step}/{cfg['dyn_steps']}] flow loss {acc / n_acc:.4f} "
                  f"lr {opt.param_groups[0]['lr']:.2e} {rate:.3f} s/step "
                  f"(~{rate * (cfg['dyn_steps'] - step) / 3600:.1f} h left)", flush=True)
            t0, acc, n_acc = time.time(), 0.0, 0
        if time.time() - t_save > 60 * cfg["save_minutes"]:
            save(); t_save = time.time()
        if deadline.reached():
            save()
            print(f"[dynamics] time budget reached at step {step}/{cfg['dyn_steps']} -- resume next link",
                  flush=True)
            return False
    save()
    atomic_save({"model": cfg["dyn_model"], "cases": list(cases), "ema": ema.state_dict(),
                 "res_scale": res_scale, "stats": bank.stats, "steps": step}, final_path)
    print(f"[dynamics] done -> {final_path}", flush=True)
    return True


def load_dynamics(run_dir, device="cuda"):
    ck = torch.load(run_dir / "dynamics" / "model_ema.pth", map_location="cpu", weights_only=False)
    model = SpaceTimeFlow(**ck["model"]).to(device).eval().requires_grad_(False)
    model.load_state_dict(ck["ema"])
    return model, ck


@torch.no_grad()
def sample_targets(model, bank, examples, cases, dts, res_scale, n_samples, n_steps, amp=True, seed=0):
    """Generate the target latents of a batch of sequences (all with the same layout).
    Returns (samples (S, B, M, K, d), reference (B, M, K, d)) as UNnormalized latents,
    M = number of targets per sequence, in sequence (tau) order."""
    b = to_tensors(bank, examples, cases, dts)
    bsz, t, k, d = b["z"].shape
    rep = lambda x: x.repeat(n_samples, *([1] * (x.dim() - 1)))
    z, ref, tau, is_t = rep(b["z"]), rep(b["ref"]), rep(b["tau"]), rep(b["is_t"])
    anchors, log_dt = rep(b["anchors"]), rep(b["log_dt"])
    gen = torch.Generator(device=z.device).manual_seed(seed)
    x = torch.randn(z.shape, device=z.device, generator=gen)
    mask = is_t[..., None, None]
    for i in range(n_steps):
        u = torch.full((x.shape[0],), i / n_steps, device=x.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            v = model(torch.where(mask, x, z), ref, anchors, tau, is_t, u, log_dt)
        x = x + v.float() / n_steps
    out = (ref + res_scale * x).view(n_samples, bsz, t, k, d)
    samples = bank.denorm(out[:, b["is_t"]].view(n_samples, bsz, -1, k, d))
    reference = bank.denorm(b["ref"][b["is_t"]].view(bsz, -1, k, d))
    return samples, reference
