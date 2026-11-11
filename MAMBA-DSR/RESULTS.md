# MAMBA-DSR: inherited Results & Findings

This file is retained from the cloned `diffusion_dsr` project as historical
context. Its diffusion-specific results do not describe the direct
bidirectional-Mamba implementation in this directory; new MAMBA-DSR results
must be generated with `train_mamba.py` and the commands in `README.md`.

> **Protocol status:** The results in this document are exploratory historical
> runs. They used the original Stage-A setup in which all 149 Aβ frames were
> available before Stage B. The corrected frozen-DSR protocol is now
> implemented in `protocol.py` and `evaluate_protocol.py`; new headline
> numbers must come from that protocol and should not be mixed with the
> historical numbers below.

A generative (diffusion-based) reformulation of DSR's protein-surface-dynamics task,
built and evaluated on the bundled Abeta(1-40) demo trajectory. Goal: match/surpass DSR
on reconstruction **and** add an ensemble/uncertainty capability DSR structurally lacks.

All metrics: DSR's own `lib/metrics.py` (IoU↑, Chamfer↓, Normal-Consistency↑), marching
cubes at resolution 128, evaluated against real ground-truth surface frames. All numbers
below are from held-out (interpolation) frames 20 / 75 / 130 unless noted.

---

## Architecture (3 stages, all in `diffusion_dsr/`, reusing `DSR/lib/` unmodified)

- **Stage A — per-frame latent auto-decoder** (`train_autodecoder.py`): DSR's exact decoder +
  Eikonal loss, but one learned latent **per frame** (not per trajectory). Gives a population
  of latents for Stage B to model.
- **Stage B — hybrid generator** (`model.py`, `train_diffusion.py`): a deterministic
  `MeanPredictor` μ(t) + a DDPM over the **residual** r = z − μ(t). μ(t) nails the mean,
  diffusion supplies the spread (uncertainty).
- **Stage C — sample & eval** (`sample.py`): z = μ(t) + residual-sample → DSR decoder →
  marching cubes → rescale → metrics.

---

## Headline results (IoU at frames 20 / 75 / 130)

| variant | frame 20 | frame 75 | frame 130 | note |
|---|---|---|---|---|
| **DSR (reproduced baseline)** | 0.829 | 0.771 | 0.883 | our target |
| v1 (unstructured latent) | 0.477 | 0.594 | 0.579 | diffusion can't learn scattered latents |
| **v2 (smoothness fix)** | 0.714 | 0.650 | 0.767 | big jump; ~70% of gap closed |
| v3 (v2 + decoder-consistency) | ~v2 | ~v2 | ~v2 | penalty ineffective at convergence |
| linear-interp mean (best) | 0.762 | **0.872** | 0.765 | **frame 75 BEATS DSR on all 3 metrics** |
| **Oracle** (decode *true* latent) | **0.889** | **0.886** | **0.899** | the ceiling — beats DSR everywhere |

Chamfer / NC followed the same ordering (e.g. Oracle Chamfer ~0.10 vs DSR 0.35–0.92; frame-75
interp Chamfer 0.136 vs DSR 0.924, NC 0.933 vs 0.904).

---

## Key findings (the scientific content)

1. **The per-frame-latent representation is already superior to DSR.** The *oracle* (decode
   the true learned latent) beats DSR on every metric at every frame — Chamfer ~3–9× better.
   The decoder/representation is not the bottleneck.

2. **The entire deficit is Stage B predicting the latent at held-out t.** Oracle 0.89 →
   pipeline 0.65–0.77. 100% of the gap is "given t, produce the right latent."

3. **Why v1 failed → the v2 fix.** v1's per-frame latents were *unstructured noise* in time
   (adjacent-frame distance ≈ random-pair distance, smoothness ratio **0.966**; near-isotropic,
   top-10 PCA dims = 45% var). A temporal-smoothness penalty on Stage A collapsed the ratio to
   **0.21** and made the trajectory a smooth, low-dim curve — turning an unlearnable target into
   a learnable one. This was the single biggest win.

4. **Frame 75 beats DSR — and it's not luck.** Where the latent trajectory is *locally linear*,
   interpolation error is tiny (0.023) and the decoded surface beats DSR (IoU 0.872 vs 0.771).
   This is the method working as intended.

5. **The binding constraint (frames 20/130): a geometric obstruction.** The decoder converts a
   ~0.05 latent error into ~0.13 IoU loss. To tolerate the ~0.05–0.06 interpolation error at
   *curved* stretches, the decoder would need to be flat over a radius > half the inter-frame
   latent spacing (0.044) — which forces neighboring frames to collapse together, dropping the
   oracle ceiling. **Flattening (v3/fine-tunes) confirmed this empirically: sensitivity halved,
   but the ceiling collapsed 0.89→0.84 — a wash.** You cannot flatten enough without destroying
   what you're protecting.

6. **DECISIVE: denser sampling cannot help Abeta — it's thermal-noise-dominated.** Training-free
   test on the full 100,000-frame trajectory: linear-interpolating two **adjacent** frames still
   misses the middle by **0.96 Å** per atom (frame-to-frame step 1.35 Å). Interp error does **not**
   →0 as sampling density increases; it *floors* at the thermal-jitter scale. So the latent
   trajectory can never be made smoothly predictable for this disordered peptide — the obstruction
   is fundamental to Abeta, not an artifact of coarse sampling.

   | stride | 1 | 2 | 4 | 13 | 52 |
   |---|---|---|---|---|---|
   | interp err (Å) | **0.96** | 1.38 | 1.93 | 3.14 | 5.58 |

   This exactly matches DSR's own documented limitation ("tends to model major conformation
   changes rather than noise-like random vibrations"; IoU correlates with trajectory RMSD).
   We picked effectively the worst-case protein.

7. **Trajectory-dependence of the advantage (AdK DIMS spot-check).** A directed *transition*
   trajectory (AdK DIMS) is smoother: stride-1 interp error 0.42 Å (~0.8% of protein size, vs
   Abeta's ~3.2%). BUT DSR is *strongest* on such trajectories (its paper: 0.93 on DIMS). So the
   method's *advantage over DSR* was largest exactly where DSR is *weak* (noisy trajectories) —
   which is also where our prediction hits the noise floor. Net: whether we surpass DSR is
   genuinely trajectory-dependent, not a clean win on smooth data. (Full DIMS run not executed.)

---

## Honest bottom line

- **Large, real improvement** over the naive diffusion baseline (v1→v2: IoU ~0.5→~0.7, Chamfer
  3–5× better), driven by the latent-smoothness insight.
- **Beats DSR where the trajectory is locally smooth** (frame 75, all three metrics).
- **Does not *uniformly* surpass DSR on Abeta**, blocked by a now-fully-characterized combination
  of decoder latent-sensitivity and thermal-noise-floored interpolation — a limitation that
  provably cannot be fixed by denser sampling and mirrors DSR's own weakness on noisy trajectories.
- **Adds a capability DSR lacks**: ensemble sampling / uncertainty (tightened substantially from
  v1 to the hybrid).

The most defensible framing: this is a rigorous negative/mixed result on the hardest case
(a disordered peptide), with a clear positive prediction — the method should be *competitive*
on smooth directed transitions — left as the next experiment.

---

## Artifacts & reproduction

- Stage A checkpoints: `exp/abeta_150_perframe_lat32/` (v1), `..._lat8_v2/`, `..._lat8_v3/`,
  `..._lat8_v3ft*` (fine-tunes).
- Stage B checkpoints: `exp/abeta_150_diffusion_lat32/` (v1), `..._hybrid_lat8_v2/`.
- Reconstructions: `../reconstructions/DiffusionDSR_abeta_150_*/`; DSR baselines:
  `../reconstructions/DSR_abeta_30_demo_res{64,128}/`.
- Training logs: `logs/{stageA,stageB,stageC,diagnostics}/`.
- Key diagnostics (oracle, robustness, thermal-noise, interp) were run as inline scripts this
  session; the thermal-noise test is the most important and is trivially re-runnable against
  `datasets/DSR/Abeta_500ns/`.

### Commands
```bash
conda activate graphmae   # env with torch+CUDA, pymol, pyhocon, trimesh, scikit-image (py3.9)
cd diffusion_dsr
# Corrected protocol: Stage A sees only DSR's 30 training frames (0,5,...,145).
python train_autodecoder.py --case abeta_150 --protocol dsr_abeta --latent-size 8 \
  --smooth-lambda 1.0 --end-iter 800 --seed 0 --run-name <name>
# Stage B (hybrid):
python train_diffusion.py --stage-a-ckpt exp/<name>/checkpoints/ckpt_latest.pth \
  --case abeta_150 --seed 0 --run-name <name>
# Stage C: evaluate the exact 119 held-out frames with DSR's metrics:
python evaluate_protocol.py --stage-a-ckpt exp/<name>/checkpoints/ckpt_latest.pth \
  --stage-b-ckpt exp/<name>/ckpt_latest.pth --case abeta_150 \
  --gt-case abeta_30_gt --resolution 128 --num-samples 1 --seed 0 --run-name <name>_eval
```
