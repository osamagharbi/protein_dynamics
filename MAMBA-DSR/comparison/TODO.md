# Comparison plan and living TODO

Status markers: `[x]` complete, `[~]` in progress or partially complete,
`[ ]` not complete, `[!]` blocked by missing data or an unresolved validity
question.

## A. Freeze the comparison rules

- [x] Use the same source trajectory and converted OBJ surface meshes for both
      methods.
- [x] Use the DSR paper's frame-selection rules for each experiment.
- [x] Use the same normalization and time normalization.
- [x] Use resolution 128 for IoU, Chamfer distance, and normal consistency.
- [x] Keep training-frame reconstruction and held-out temporal prediction as
      separate result categories.
- [ ] Add a machine-readable validation script that checks every run's frame
      list, normalization, resolution, metric space, and ground-truth paths.

## B. Reproduce the paper's reconstruction table (Table 1)

- [x] AdK equilibrium: 1500-frame crop, every tenth frame, 150 training
      frames; train, evaluate, and compare.
- [x] Re-infer/report AdK normal consistency with an explicit invalid-normal
      policy; frame 1060 has three zero-normal ground-truth vertices.
- [ ] NhaA equilibrium: convert/validate first 1500 frames, train and evaluate.
- [ ] YiiP equilibrium: convert/validate first 1500 frames, train and evaluate.
- [ ] I-FABP water: first 400 frames, every fifth frame, 80 training frames.
- [!] DIMS: obtain/extract the trajectory and topology from the local archive.
- [!] FRODA: obtain/extract the trajectory and topology from the local archive.
- [!] DRYAD_MD: obtain the paper's source trajectory/archive.
- [!] Obtain the remaining small-protein trajectories: protG, ntl9, bba,
      cspa, T0765, ww, T0855, T0773, T0816, gpW, bbl, hyp, T0769, T0771.
- [ ] Run every available dataset sequentially with disk-safe evaluation.
- [ ] Produce the final MAMBA-DSR Table 1 reproduction and a per-dataset
      comparison against the paper values.

## C. Reproduce the Aβ experiment

- [~] Aβ-30 MAMBA-DSR exploratory run exists, but its held-out split is not
      the same as the paper's Section 4.1 reconstruction report.
- [ ] Resolve and document the paper's Aβ frame list and exact ground-truth
      set; the paper states the metrics but not the complete split in the main
      text.
- [ ] Train/evaluate MAMBA-DSR on that exact protocol.
- [ ] Compare against the paper's reported Aβ values:
      IoU `0.9361±0.0265`, Chamfer `0.0009±0.0008`, NC `0.9967±0.0011`.

## D. Reproduce temporal interpolation

- [ ] Implement the paper's interpolation frame lists from the selected
      training frames.
- [ ] Evaluate MAMBA-DSR on every missing intermediate frame with IoU,
      Chamfer, and NC.
- [ ] Implement the same linear-interpolation baseline used in the paper.
- [ ] Compare MAMBA-DSR, DSR, and linear interpolation in one table/figure.

## E. Reproduce future prediction

- [ ] Define the exact prefix and future-frame windows used by the paper.
- [ ] Evaluate future frames 1--10 and 11--20 for DIMS, I-FABP, and FRODA.
- [ ] Report all three metrics with the same aggregation and units as DSR.

## F. Reproduce the GPCRmd case study

- [!] Obtain compatible topology files for the four GPCRmd trajectories.
- [ ] Recreate the four-trajectory MAMBA-DSR run and the paper's time-sampling
      strategy.
- [ ] Save qualitative figures and, where ground truth is available, metrics.

## G. Validity and reporting

- [ ] Use at least three seeds for the final MAMBA-DSR rows where feasible.
- [ ] Record failed mesh extractions and invalid-normal counts explicitly.
- [ ] Never replace NaN metrics silently; report official and robust diagnostic
      values separately.
- [ ] Keep training-frame metrics clearly labeled as reconstruction, not
      generalization.
- [ ] Generate a final report with data provenance, commands, checkpoints,
      metrics, runtime, and unresolved limitations.
