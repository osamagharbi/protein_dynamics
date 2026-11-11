# Protein Dynamics Project Constitution

**Status:** active baseline
**Last verified:** 2026-10-12
**Scope:** DSR and Diffusion-DSR experiments in this repository; ProtMD is literature/data context only unless an experiment is explicitly added.

## 1. Mission

Determine when time-conditioned generative or interpolation-based molecular-surface models improve on DSR, and when trajectory noise creates a measurable ceiling that neither method can overcome. The project is an empirical analysis of protein-surface dynamics, not a claim to reproduce full physical MD or to replace MD simulation.

The primary scientific question is:

> Does the advantage of Diffusion-DSR over DSR depend predictably on trajectory smoothness, and can ensemble spread provide useful uncertainty information?

## 2. Current validated baseline

The following statements are supported by repository artifacts and should be treated as the current baseline:

- The working implementation has three stages: a per-frame DSR-style auto-decoder, a time-conditioned mean predictor plus residual DDPM, and mesh extraction/evaluation with DSR metrics.
- The Aβ demo contains 149 OBJ frames for the Stage A runs. The directory `DSR/datasets/abeta_150/obj` is a symlink to `../abeta_30_gt`; preserve this provenance when reporting results.
- Stage A checkpoints and logs exist for the unstructured latent baseline (v1), the temporally smoothed latent model (v2), and consistency-penalty variants (v3/fine-tunes).
- Historical v2 runs held out every fifth frame only for Stage B, after Stage A had already seen all frames. The corrected protocol is now implemented in `diffusion_dsr/protocol.py`: Stage A trains only on DSR's 30 frames (0,5,...,145), Stage B uses only those latents, and Stage C evaluates the 119 interleaved frames.
- Saved Aβ logs support the reported qualitative ordering: v1 is poor, v2 is substantially better, and v2 produces nonzero ensemble diversity. The saved v2 single-sample evaluations are approximately IoU 0.714, 0.650, and 0.767 at the three reported target times; the four-sample ensemble at the middle target has IoU 0.654 ± 0.076 and nonzero pairwise Chamfer.
- The saved results support a mixed result, not a universal win: the v2 method beats the reproduced DSR baseline at the middle Aβ target in the recorded comparison, but the headline claim must remain trajectory- and protocol-dependent.
- The project has not run the complete Diffusion-DSR pipeline on AdK DIMS or on a multi-protein benchmark. ProtMD has not been run.

## 3. Evidence labels

Every future claim must use one of these labels in notes, tables, or result files:

- **Verified:** directly reproducible from a checked-in script plus a saved artifact or log, with the exact dataset, split, seed, resolution, and metric stated.
- **Repository-supported:** visible in existing logs/checkpoints, but missing at least one requirement for independent reproduction.
- **Reported-only:** present in narrative notes or a pasted summary, without a saved output that can currently be checked.
- **Open:** not tested or not established.

At the current baseline, the historical Aβ v1/v2 qualitative improvement and ensemble diversity are repository-supported, but they are not results from the corrected protocol. The exact oracle table, latent-distance ratios, geometric-obstruction numbers, 0.96 Å thermal-noise result, 0.42 Å AdK DIMS result, calibration, and any claim about ten or more proteins are reported-only or open unless their raw diagnostics are added.

## 4. Non-negotiable scientific rules

1. **Fair splits first.** Stage A and Stage B must use the same training frames. For the default interpolation benchmark, train on every fifth frame and evaluate only on the interleaved frames, identically for Diffusion-DSR and DSR.
2. **Matched baselines.** Retrain DSR with the same frames, spatial resolution, target frames, mesh extraction settings, and metric implementation. Do not compare a saved DSR run with a differently sampled Diffusion-DSR run and call it a definitive improvement.
3. **Multiple seeds.** The minimum publishable comparison is three independent seeds per method and protein. Report mean ± standard deviation and retain seed-specific logs/checkpoints.
4. **No leakage.** A latent learned for a target frame is not an oracle-free prediction. Oracle decoding is a diagnostic ceiling and must never be presented as the predictive model’s result.
5. **Uncertainty requires calibration.** Ensemble spread is only an uncertainty result after testing whether spread predicts per-frame reconstruction error, preferably using held-out frames and a stated correlation or calibration curve.
6. **Use consistent units.** IoU and normal consistency are higher-is-better; Chamfer is lower-is-better. State whether Chamfer is in normalized coordinates or physical Å. Keep IoU inputs normalized exactly as required by the current metric implementation.
7. **Separate shape fit from physical validity.** These experiments evaluate surface agreement with classical-MD reference frames. They do not establish experimental agreement, energy conservation, solvent realism, or physical validity.
8. **Do not generalize from Aβ.** Aβ is a useful stress test and currently the only completed protein. It cannot support claims about smooth trajectories, cross-protein generalization, or broad superiority.
9. **Record provenance.** Each run must record dataset path/version, frame indices, split rule, seed, latent size, smoothness and consistency weights, DDPM/DDIM settings, resolution, code revision, and output paths.

## 5. Result interpretation

The default interpretation of the current evidence is:

- Temporal latent smoothness is a promising conditioning prior and clearly improves the Aβ diffusion baseline.
- The representation/decoder may have a higher ceiling than DSR, but this must be re-established with a leakage-free oracle diagnostic and a matched predictive comparison.
- Aβ does not establish that denser temporal sampling solves the problem. The thermal-noise floor claim remains a hypothesis until its raw diagnostic script/output is restored.
- The intended contribution is currently an analysis of when the method works and where noise limits interpolation, with uncertainty calibration as a secondary contribution. A “Diffusion-DSR universally beats DSR” framing is prohibited by this constitution unless future matched experiments justify it.

## 6. Required next milestone

Run **AdK DIMS** end to end under the corrected protocol before scaling to ten proteins:

1. materialize and checksum the DIMS surface frames;
2. choose and freeze the every-fifth-frame train/holdout split;
3. train matched DSR and Diffusion-DSR with at least three seeds;
4. evaluate IoU, Chamfer, and normal consistency at identical resolution;
5. save per-frame predictions, metrics, logs, configuration, and seed metadata;
6. compute the training-free smoothness score and compare it with the method-minus-DSR gain;
7. run the ensemble-spread-versus-error calibration test.

The milestone succeeds only if the outputs make the result independently auditable. If Diffusion-DSR does not improve on DIMS, revise the project framing toward the noise-floor/uncertainty analysis before adding more proteins.

## 7. Repository conventions

- `KNOWLEDGE_BASE.md`: literature survey and source provenance; distinguish paper claims from project measurements.
- `diffusion_dsr/RESULTS.md`: completed experiments and numbers, with evidence labels and artifact links.
- `diffusion_dsr/TODO.md`: executable backlog; do not mark an item complete without artifacts.
- `DSR/`: reference implementation and matched-baseline code; avoid modifying it unless the change is explicitly isolated and recorded.
- `datasets/`: source data and acquisition notes; record external download locations and checksums where possible.
- `reconstructions/`: generated meshes and previews; never treat meshes alone as sufficient evidence without the run configuration and metric log.
- New scripts should be deterministic where possible, expose a `--seed`, and write a machine-readable metrics file alongside human-readable logs.

## 8. Decision log

- **2026-10-12:** Constitution created after auditing the pasted findings, `KNOWLEDGE_BASE.md`, `diffusion_dsr/RESULTS.md`, `diffusion_dsr/TODO.md`, Stage A/B/C code, logs, checkpoints, and dataset layout. The current evidence supports a mixed Aβ result and a strong methodological lead, but not a publishable general claim.
