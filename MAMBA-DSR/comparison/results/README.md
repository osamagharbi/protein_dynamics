# Results storage

The dataset folders directly under this directory are reserved for
training-frame reconstruction results corresponding to DSR Table 1.

Additional experiment families have their own folders:

- `interpolation/<dataset>/`: held-out temporal interpolation and linear
  interpolation baselines;
- `future_prediction/<dataset>/`: future frames 1--10 and 11--20;
- `gpcrmd/`: the four-trajectory GPCRmd case study.

Each completed dataset should contain a short README, raw evaluator JSON/CSV,
an aggregate row in `../comparison_summary.csv`, and a corresponding entry in
`../artifacts/artifact_index.csv`.
