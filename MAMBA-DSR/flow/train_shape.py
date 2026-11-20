"""Stage 1: train the shape VAE on the training frames of all cases (IGR loss), then
encode every training frame into its latent tokens (exp/flow/<run>/latents/<case>.pt)."""
import copy
import json
import time

import numpy as np
import torch

from flow.common import (Deadline, atomic_save, check_resume, ema_update, lr_at, rng_state,
                         set_rng_state, write_json)
from flow.data import FrameBank
from flow.nets import ShapeVAE, count_params

RESUME_KEYS = ("model", "cases", "steps", "batch", "n_enc", "q_surf", "q_local", "q_global", "k")


def igr_loss(model, batch, kl_weight, normals_lambda=1.0, eikonal_lambda=0.1, amp=False):
    """DSR's loss (manifold |f| + normals + eikonal) on a VAE sample, plus the KL term."""
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        mu, logvar = model.encode(batch["pts"], batch["nrm"], batch["anchors"])
        mu, logvar = mu.float(), logvar.float()
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        h = model.tokens(z, batch["anchors"])
        qs = batch["xs"].shape[1]
        x = torch.cat([batch["xs"], batch["xo"]], dim=1).requires_grad_(True)
        f = model.query(h, x)
    g = torch.autograd.grad(f.sum(), x, create_graph=True)[0].float()
    mnfld = f[:, :qs].abs().mean()
    normals = (g[:, :qs] - batch["ns"]).norm(dim=-1).mean()
    eikonal = ((g[:, qs:].norm(dim=-1) - 1) ** 2).mean()
    kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).mean()
    loss = mnfld + normals_lambda * normals + eikonal_lambda * eikonal + kl_weight * kl
    return loss, {"loss": loss.item(), "mnfld": mnfld.item(), "normals": normals.item(),
                  "eikonal": eikonal.item(), "kl": kl.item(),
                  "sigma": torch.exp(0.5 * logvar).mean().item()}


def train_shape(cfg, run_dir, cases, deadline: Deadline):
    """Returns True when Stage 1 is complete (False = time budget reached, resume later)."""
    out = run_dir / "shape"
    ckpt_path, final_path = out / "ckpt_latest.pth", out / "model_ema.pth"
    settings = {"model": cfg["shape_model"], "cases": list(cases), "steps": cfg["shape_steps"],
                "batch": cfg["shape_batch"], "n_enc": cfg["n_enc"], "q_surf": cfg["q_surf"],
                "q_local": cfg["q_local"], "q_global": cfg["q_global"], "k": cfg["k"]}
    if final_path.is_file():
        check_resume(torch.load(final_path, map_location="cpu", weights_only=False),
                     ("model", "cases", "k"), settings, "shape")
        print("[shape] complete -- skipped", flush=True)
        return True
    bank = FrameBank(cases, cfg["k"], "cuda", cfg["case_alpha"])
    model = ShapeVAE(**cfg["shape_model"]).cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["shape_lr"], betas=(0.9, 0.99),
                            weight_decay=cfg["shape_wd"])
    amp = cfg["shape_amp"]
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    step, log, skipped = 0, [], 0
    if ckpt_path.is_file():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        check_resume(ck, RESUME_KEYS, settings, "shape")
        model.load_state_dict(ck["model_sd"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        set_rng_state(ck["rng"]); step, log = ck["step"], ck["log"]
        if amp and ck.get("amp_off"):
            amp = False
            scaler = torch.cuda.amp.GradScaler(enabled=False)
            print("[shape] mixed precision was switched off earlier in this run -- staying in fp32", flush=True)
        print(f"[shape] resumed at step {step}/{cfg['shape_steps']}", flush=True)
    else:
        print(f"[shape] ShapeVAE {count_params(model) / 1e6:.1f}M parameters, {len(bank)} training "
              f"frames from {len(cases)} cases, {cfg['shape_steps']} steps x {cfg['shape_batch']} frames",
              flush=True)

    def save():
        atomic_save({**settings, "model_sd": model.state_dict(), "ema": ema.state_dict(),
                     "opt": opt.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(),
                     "step": step, "log": log, "amp_off": cfg["shape_amp"] and not amp}, ckpt_path)
        write_json(out / "loss_log.json", log)

    t0, t_save, acc, n_acc = time.time(), time.time(), {}, 0
    while step < cfg["shape_steps"]:
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg["shape_steps"], cfg["shape_lr"], cfg["shape_warmup"])
        batch = bank.training_batch(cfg["shape_batch"], cfg["n_enc"], cfg["q_surf"], cfg["q_local"],
                                    cfg["q_global"])
        loss, parts = igr_loss(model, batch, cfg["kl_weight"], amp=amp)
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            skipped += 1
            if amp and skipped >= 20:       # fp16 trouble: continue the run in fp32
                amp, skipped = False, 0
                scaler = torch.cuda.amp.GradScaler(enabled=False)
                print(f"[shape] repeated non-finite losses in fp16 at step {step} -- switching to fp32",
                      flush=True)
            elif skipped > 50:
                raise RuntimeError("[shape] more than 50 non-finite losses -- lower shape_lr")
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(opt)
        scaler.update()
        ema_update(ema, model, min(cfg["ema"], (1 + step) / (10 + step)))
        step += 1
        for k, v in parts.items():
            acc[k] = acc.get(k, 0.0) + v
        n_acc += 1
        if step % cfg["log_every"] == 0 or step == cfg["shape_steps"]:
            rate = (time.time() - t0) / n_acc
            row = {"step": step, "lr": opt.param_groups[0]["lr"], "s_per_step": rate,
                   **{k: v / n_acc for k, v in acc.items()}}
            log.append(row)
            print(f"[shape {step}/{cfg['shape_steps']}] loss {row['loss']:.4f} mnfld {row['mnfld']:.4f} "
                  f"normals {row['normals']:.4f} eik {row['eikonal']:.4f} kl {row['kl']:.3f} "
                  f"sigma {row['sigma']:.3f} lr {row['lr']:.2e} {rate:.3f} s/step "
                  f"(~{rate * (cfg['shape_steps'] - step) / 3600:.1f} h left)", flush=True)
            t0, acc, n_acc = time.time(), {}, 0
        if time.time() - t_save > 60 * cfg["save_minutes"]:
            save(); t_save = time.time()
        if deadline.reached():
            save()
            print(f"[shape] time budget reached at step {step}/{cfg['shape_steps']} -- resume next link",
                  flush=True)
            return False
    save()
    atomic_save({"model": cfg["shape_model"], "cases": list(cases), "k": cfg["k"], "ema": ema.state_dict(),
                 "steps": step, "centers": {c: bank.meta[c]["center"] for c in cases}}, final_path)
    print(f"[shape] done -> {final_path}", flush=True)
    return True


def load_shape(run_dir, device="cuda"):
    ck = torch.load(run_dir / "shape" / "model_ema.pth", map_location="cpu", weights_only=False)
    model = ShapeVAE(**ck["model"]).to(device).eval().requires_grad_(False)
    model.load_state_dict(ck["ema"])
    return model, ck


@torch.no_grad()
def encode_latents(cfg, run_dir, cases):
    """Posterior means of every training frame (average over cfg['encode_draws'] point subsets)."""
    out = run_dir / "latents"
    shape_ck = run_dir / "shape" / "model_ema.pth"
    todo = [c for c in cases if not (out / f"{c}.pt").is_file()
            or (out / f"{c}.pt").stat().st_mtime < shape_ck.stat().st_mtime]
    if not todo:
        print("[encode] latents up to date -- skipped", flush=True)
        return
    model, _ = load_shape(run_dir)
    bank = FrameBank(cases, cfg["k"], "cuda")
    gen = torch.Generator(device="cuda")
    for case in todo:
        lo, hi = bank.case_slices[case]
        ci = bank.cases.index(case)
        zs = []
        for f0 in range(lo, hi, 16):
            fids = torch.arange(f0, min(hi, f0 + 16), device="cuda")
            acc = 0
            for d in range(cfg["encode_draws"]):
                gen.manual_seed(1000 * d + f0)
                pts, nrm = bank.frame_points(fids, cfg["n_enc"], gen)
                anchors = bank.anchors[ci].expand(len(fids), -1, -1)
                acc = acc + model.encode(pts, nrm, anchors)[0].float()
            zs.append((acc / cfg["encode_draws"]).cpu())
        z = torch.cat(zs)
        meta = bank.meta[case]
        atomic_save({"z": z, "frames": bank.fraw[lo:hi].cpu().tolist(), "step": meta["step"],
                     "anchors": bank.anchors[ci].cpu(), "center": meta["center"], "s": meta["s"],
                     "t": meta["t"], "case": case}, out / f"{case}.pt")
        print(f"[encode] {case}: {tuple(z.shape)} latents, |z| mean {z.norm(dim=-1).mean():.3f}", flush=True)
