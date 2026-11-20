#!/usr/bin/env python3
"""MAMBA-Flow pipeline (one model for all datasets).

    cache     training frames of every case -> exp/flow/cache/<case>/ (DSR loader + normalization)
    shape     Stage 1: anchored-token shape VAE, IGR loss            -> exp/flow/<run>/shape/
    encode    latent tokens of every training frame                  -> exp/flow/<run>/latents/
    dynamics  Stage 2: masked flow matching over space x time         -> exp/flow/<run>/dynamics/
    eval      training-frame reconstruction (DSR metrics, res 128)   -> exp/flow_summary/<case><tag>.csv
Held-out frames: python eval_heldout.py --method flow --only <case>  (calls run_flow.py --heldout).

    python run_flow.py                                 # every prepared dataset (datasets/DSR_processed/*/meta.json)
    python run_flow.py --only adk_dims adk_froda       # a subset (local tests)
    python run_flow.py --smoke --only adk_dims         # minutes-long plumbing test (tiny model)
    python run_flow.py --seed 1 --tag _s1              # second seed
    python run_flow.py --set shape_model.width=256 --set dyn_steps=20000
    python run_flow.py --check                         # validate data, print the plan, no GPU
Every stage resumes from its checkpoint; --stop-after-minutes ends a link cleanly with exit
code 3 (the chained Slurm job continues). Run name = "flow" + tag; outputs in exp/flow/<run>/.
"""
import argparse
import copy
import json
import sys

from flow.common import Deadline, seed_all, write_json
from flow.data import FLOW_EXP, build_cache, case_protocol, prepared_cases
from run_multicase import CASES

DEFAULTS = dict(
    # data / sampling (per frame and step)
    k=256, n_enc=8192, q_surf=8192, q_local=8192, q_global=1024, case_alpha=0.5,
    # Stage 1
    shape_model=dict(width=384, latent_dim=32, heads=8, enc_layers=6, dec_layers=6, n_freq=8,
                     sphere_radius=0.6, query_blocks=1, head_layers=0, head_width=256),
    shape_steps=200_000, shape_batch=4, shape_lr=1e-4, shape_wd=0.01, shape_warmup=2000,
    shape_amp=True, kl_weight=1e-3, encode_draws=4,
    # Stage 2
    dyn_model=dict(width=256, heads=8, blocks=8, temporal="mamba", dropout=0.1, n_freq=8),
    dyn_steps=40_000, dyn_batch=8, dyn_lr=2e-4, dyn_wd=0.05, dyn_warmup=1000, dyn_amp=True,
    dyn_grad_ckpt=False,
    tasks=dict(probs=[0.35, 0.25, 0.40], ctx_min=4, ctx_max=16, fine_ctx_max=8, horizon=20,
               interp_targets=4),
    # shared
    ema=0.999, grad_clip=1.0, log_every=200, save_minutes=30,
    # evaluation
    resolution=128, metric_workers=8, n_samples=4, flow_steps=32, ctx_test=12, seq_batch=16,
    eval_max_frames=0,
)
SMOKE = {"k": 64, "n_enc": 1024, "q_surf": 1024, "q_local": 1024, "q_global": 128,
         "shape_model.width": 64, "shape_model.heads": 4, "shape_model.enc_layers": 2,
         "shape_model.dec_layers": 2, "shape_steps": 200, "shape_batch": 2, "shape_warmup": 20,
         "dyn_model.width": 64, "dyn_model.heads": 4, "dyn_model.blocks": 2, "dyn_steps": 200,
         "dyn_batch": 4, "dyn_warmup": 20, "log_every": 20, "encode_draws": 1, "n_samples": 2,
         "flow_steps": 4, "seq_batch": 8, "eval_max_frames": 2}


def set_key(cfg, dotted, value):
    node = cfg
    *path, last = dotted.split(".")
    for p in path:
        node = node[p]
    if last not in node:
        raise SystemExit(f"--set: unknown key {dotted!r}")
    node[last] = value


def parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def build_config(args):
    cfg = copy.deepcopy(DEFAULTS)
    if args.smoke:
        for k, v in SMOKE.items():
            set_key(cfg, k, v)
    for item in args.set or []:
        key, _, val = item.partition("=")
        set_key(cfg, key, parse_value(val))
    if args.resolution:
        cfg["resolution"] = args.resolution
    if args.metric_workers is not None:
        cfg["metric_workers"] = args.metric_workers
    cfg["dyn_model"]["latent_dim"] = cfg["shape_model"]["latent_dim"]
    cfg["seed"] = args.seed
    return cfg


def selftest():
    """GPU checks of the code paths that differ between PyTorch versions (smoke runs)."""
    import torch
    from flow.nets import HAS_SDPA, ShapeVAE, attention
    q, k, v = torch.randn(3, 2, 4, 64, 16, device="cuda").unbind(0)
    err = float((attention(q, k, v) - attention(q, k, v, True)).abs().max())
    m = ShapeVAE(width=64, heads=4, enc_layers=1, dec_layers=1).cuda()
    x = (torch.rand(1, 32, 3, device="cuda") * 2 - 1).requires_grad_(True)
    h = m.tokens(torch.randn(1, 16, 32, device="cuda"), torch.rand(1, 16, 3, device="cuda"))
    g = torch.autograd.grad(m.query(h, x).sum(), x, create_graph=True)[0]
    (g.norm(dim=-1) - 1).pow(2).mean().backward()            # double backward through the decoder
    print(f"[selftest] torch {torch.__version__} on {torch.cuda.get_device_name(0)}: SDPA "
          f"{'used' if HAS_SDPA else 'absent (plain attention)'}, max |SDPA - plain| = {err:.1e}; "
          f"double backward OK", flush=True)
    if err > 1e-4:
        raise SystemExit("[selftest] fused and plain attention disagree")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(CASES), help="cases (default: every prepared dataset)")
    ap.add_argument("--tag", default="", help="run-name suffix, e.g. _s1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny model / few steps plumbing test")
    ap.add_argument("--stages", nargs="+", default=["cache", "shape", "encode", "dynamics", "eval"],
                    choices=["cache", "shape", "encode", "dynamics", "eval"])
    ap.add_argument("--stop-after-minutes", type=float, default=0.0)
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value")
    ap.add_argument("--resolution", type=int, default=0)
    ap.add_argument("--metric-workers", type=int, default=None)
    ap.add_argument("--heldout", action="store_true", help="held-out evaluation of --only cases")
    ap.add_argument("--max-per-split", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="validate data and print the plan, then exit")
    ap.add_argument("--resume", action="store_true", help="accepted for symmetry; stages always resume")
    args = ap.parse_args()
    if args.smoke and not args.tag.endswith("_smoke"):
        args.tag += "_smoke"
    run_name = f"flow{args.tag}"
    run_dir = FLOW_EXP / run_name
    cfg = build_config(args)

    if args.heldout:
        if not args.only:
            ap.error("--heldout needs --only <case ...>")
        from flow.evaluate import eval_heldout
        seed_all(args.seed)
        for case in args.only:
            eval_heldout(cfg, run_dir, run_name, case, args.seed, args.max_per_split)
        return 0

    cases = sorted(args.only or prepared_cases())
    if not cases:
        raise SystemExit("no prepared datasets found (datasets/DSR_processed/*/meta.json); pass --only")
    stored = run_dir / "cases.txt"
    if stored.is_file() and stored.read_text().split() != cases and not args.check:
        raise SystemExit(f"{run_name} was started with cases {stored.read_text().split()}, not {cases}; "
                         "use another --tag for a different case list")
    if args.check:
        total = 0
        for case in cases:
            spec, available, train, step = case_protocol(case)
            missing = [i for i in train if i not in available]
            total += len(train)
            print(f"{case:28s} {len(train):4d} training frames (step {step}, {train[0]}..{train[-1]})"
                  + (f"  MISSING {len(missing)} meshes" if missing else ""))
        print(f"[plan] {run_name}: {len(cases)} cases, {total} training frames; Stage 1 "
              f"{cfg['shape_steps']} steps x {cfg['shape_batch']} frames, Stage 2 {cfg['dyn_steps']} steps "
              f"x {cfg['dyn_batch']} sequences -> {run_dir}")
        return 0

    if args.smoke:
        selftest()
    run_dir.mkdir(parents=True, exist_ok=True)
    stored.write_text("\n".join(cases) + "\n")
    write_json(run_dir / "config.json", {**cfg, "cases": cases, "tag": args.tag})
    deadline = Deadline(args.stop_after_minutes)
    print(f"[flow] run {run_name}: {len(cases)} cases {cases}; stages {args.stages}", flush=True)

    if "cache" in args.stages:
        for case in cases:
            build_cache(case)
    seed_all(args.seed)
    if "shape" in args.stages:
        from flow.train_shape import train_shape
        if not train_shape(cfg, run_dir, cases, deadline):
            return 3
    if "encode" in args.stages:
        from flow.train_shape import encode_latents
        encode_latents(cfg, run_dir, cases)
    if "dynamics" in args.stages:
        from flow.dynamics import train_dynamics
        if deadline.reached(margin_s=600):
            print("[flow] less than 10 min left -- Stage 2 starts in the next link", flush=True)
            return 3
        if not train_dynamics(cfg, run_dir, cases, deadline):
            return 3
    if "eval" in args.stages:
        from flow.evaluate import eval_train
        eval_train(cfg, run_dir, run_name, cases, args.tag, args.seed, cfg["eval_max_frames"])
    print(f"[flow] {run_name}: all requested stages complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
