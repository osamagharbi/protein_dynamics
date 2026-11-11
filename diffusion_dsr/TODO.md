# TODO — turning Diffusion-DSR into a validated contribution

Current status (see `RESULTS.md`): strong groundwork on ONE demo trajectory (Abeta), does
not uniformly beat DSR, components individually non-novel. The shortest honest path to a real
contribution is the **analysis framing**: *"When does generative/interpolation-based surface-
dynamics modeling beat DSR, and when does it hit a fundamental noise floor?"* — established
rigorously across many proteins, with the smooth↔noisy axis as the organizing result. Our
diagnostics (oracle gap, interp-error floor, geometric obstruction) become the method.

Target claim to validate: **"On smoothly-varying (directed-transition) trajectories our
generative model matches/beats DSR AND adds calibrated uncertainty; on thermal-noise-dominated
(disordered/equilibrium) trajectories all interpolation-based methods — including DSR — hit a
quantifiable noise floor we characterize."**

Legend: [P0] blocker for any credible claim · [P1] needed for the paper · [P2] strengthens it.

---

## 1. Fix methodology / fairness before scaling  [P0]
- [x] **Fix the eval protocol to match DSR exactly.** Implemented in `protocol.py`: Stage A
      trains only on frames 0,5,...,145; Stage B uses only those latents; Stage C evaluates
      the 119 interleaved frames at exact frame indices. DSR-compatible time normalization,
      resolution 128, and metrics are documented in `PROTOCOL.md`.
- [x] **Decide and document the held-out set once**, apply identically to DSR and to us.
      The existing DSR checkpoint remains frozen; the corrected Diffusion-DSR run must use
      the same split and target frames.
- [ ] **Multi-seed everything** (≥3 seeds for Stage A latents + Stage B) — report mean ± std.
      Single-run numbers are not admissible.
- [ ] **Pin the exact DSR baseline**: retrain DSR per-protein with its published config so every
      comparison is same-data, same-resolution, same-metric (reuse `DSR/lib/metrics.py`).

## 2. Multi-protein evaluation  [P0]
- [ ] Build a dataset table spanning the smooth↔noisy axis. Candidates already downloaded:
      - smooth/directed: **AdK DIMS**, **AdK FRODA** (conformational transitions)
      - mid: **I-FABP**, **AdK equilibrium**
      - noisy/disordered: **Abeta** (done), **NhaA**, **YiiP** (membrane, large)
      - + the 14 DRY_AD_MD proteins DSR reported (protG, bba, cspa, ntl9, ww, T-series…) for
        direct comparability to DSR's Table 1.
- [ ] **Precompute a per-trajectory "smoothness score"** (the training-free interp-error-vs-stride
      metric from `RESULTS.md` finding #6) for every protein — this is the paper's x-axis.
- [ ] Surface extraction pipeline (adapt `DSR/data_process.py`, PyMOL) run per protein, organized
      under `datasets/.../<protein>/obj/` (nest, don't dump).
- [ ] Run the full 3-stage pipeline + matched DSR baseline on **≥10 proteins**; collect IoU /
      Chamfer / NC per held-out frame.

## 3. The core scientific result  [P0]
- [ ] **The money plot**: per-protein (our IoU − DSR IoU) vs the smoothness score. Hypothesis:
      Δ > 0 on smooth trajectories, → 0 or < 0 as noise rises. This is the contribution.
- [ ] **Quantify the noise floor generally**: show interp-error-floor (finding #6) across proteins,
      and that it predicts the achievable-IoU ceiling for *both* DSR and us → reframes "we don't
      beat DSR on noisy data" as "no interpolation method can, and here's the law."
- [ ] Confirm the **oracle > DSR** result holds across proteins (representation superiority is
      general, not an Abeta fluke).

## 4. Make the uncertainty capability a real result  [P1]
- [ ] **Calibration test**: does ensemble spread at a frame *predict* that frame's reconstruction
      error? (correlation of per-frame ensemble-variance vs per-frame Chamfer-to-GT). DSR cannot
      do this at all — so if ours is calibrated, that's a clean standalone contribution.
- [ ] Show a use-case: uncertainty flags the noisy/hard frames automatically.
- [ ] Tighten the ensemble (v1 spread was noise-sized; hybrid better) — report spread vs error.

## 5. Method rigor / ablations  [P1]
- [ ] Ablate the components that matter: smoothness-λ (v1 vs v2 — already have the story),
      latent dim, hybrid (mean+residual) vs pure diffusion, consistency penalty on/off.
- [ ] Report the **geometric obstruction** cleanly: flat-radius vs half-inter-frame-spacing, and
      the flatten-vs-ceiling wash (finding #5) — this is a genuine methodological insight.
- [ ] Honest efficiency comparison: our inference (DDIM steps + decode) vs DSR (single forward).

## 6. Positioning / novelty  [P1]
- [ ] Literature check: how is this different from LION / latent point-cloud diffusion / DiffMD /
      the MoE-DSR follow-up? Sharpen the delta (time-conditioned generative *surface* dynamics +
      the noise-floor analysis + uncertainty). Confirm no one has published exactly this.
- [ ] Decide the paper's spine: **analysis/when-it-works** (recommended, most defensible) vs
      **method/we-beat-DSR** (only if §3 money plot shows robust wins on a real protein class).

## 7. Generalization stretch goals  [P2]
- [ ] Cross-protein / family generalization (toward the MoE-DSR problem) — needs ATLAS
      (deferred; ~TB-scale, see `datasets/MoE-DSR/`).
- [ ] Test on the *real* (non-demo) full trajectories, not bundled demos.
- [ ] Extrapolation (future-frame) behavior vs DSR, framed honestly (both weak).

---

## Immediate next step (smallest useful increment)
Run the **AdK DIMS full experiment** with the fixed protocol (§1) + matched DSR baseline: it's
the first data point of the §3 money plot and directly tests the central hypothesis on the
most favorable case. If Δ(ours−DSR) > 0 there with clean protocol + multi-seed, the analysis
paper is viable and worth scaling to §2's full protein set. If not, the honest result is the
noise-floor analysis alone (still a modest contribution) and we stop.
