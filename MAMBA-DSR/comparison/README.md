# MAMBA-DSR versus DSR: comparison workspace

This directory is the experiment ledger for a fair comparison between
MAMBA-DSR and the published DSR results.

The comparison target is the same source data, frame-selection protocol,
normalization, reconstruction resolution, and metrics. The optimization
algorithm and batching are allowed to differ because MAMBA-DSR is a different
architecture.

## Current model status

Only **AdK equilibrium** currently has completed paper-protocol MAMBA-DSR
Stage-A and Stage-B checkpoints. Aβ-30 also has older exploratory Stage-A and
Stage-B checkpoints, but those runs use a different evaluation setup and are
not registered as a final paper comparison. Models for NhaA, YiiP, I-FABP,
DIMS, FRODA, DRYAD_MD, the remaining small proteins, and GPCRmd still need to
be trained after their data/protocols are prepared.

## Start here

- [`TODO.md`](TODO.md): living checklist and current status.
- [`protocol/comparison_protocol.md`](protocol/comparison_protocol.md): rules
  that make a result comparable.
- [`protocol/dataset_manifest.csv`](protocol/dataset_manifest.csv): dataset
  availability and required frame splits.
- [`protocol/data_readiness_matrix.csv`](protocol/data_readiness_matrix.csv):
  current raw-data, conversion, protocol-view, and checkpoint readiness.
- [`paper_reference/table1_reconstruction.csv`](paper_reference/table1_reconstruction.csv):
  values reported by the DSR paper.
- [`results/comparison_summary.csv`](results/comparison_summary.csv): current
  MAMBA-DSR results alongside the paper baseline.
- [`artifacts/artifact_index.csv`](artifacts/artifact_index.csv): checkpoints,
  metrics, and preserved reconstructions. Large files are linked rather than
  duplicated.

## Directory layout

```text
comparison/
├── TODO.md
├── README.md
├── protocol/
│   ├── comparison_protocol.md
│   └── dataset_manifest.csv
├── paper_reference/
│   ├── reported_experiments.md
│   ├── table1_reconstruction.csv
│   └── table2_future_prediction.csv
├── results/
│   ├── comparison_summary.csv
│   ├── <dataset>/                 # Table 1 / reconstruction results
│   ├── interpolation/<dataset>/
│   └── future_prediction/<dataset>/
├── artifacts/
│   ├── artifact_index.csv
│   ├── checkpoints/<dataset>/
│   └── reconstructions/<dataset>/
└── logs/<dataset>/
```

Only results that pass the protocol checks belong in `results/`. Smoke tests,
legacy Aβ experiments, and physical-space metrics remain outside this ledger
and may be referenced as exploratory artifacts.

## Update procedure after each run

1. Preserve the raw evaluator JSON/CSV in the dataset result folder via a
   lightweight link to `../reconstructions/`.
2. Add the aggregate row to `results/comparison_summary.csv`.
3. Add checkpoints, loss logs, and retained meshes to `artifacts/artifact_index.csv`.
4. Update the dataset status in `protocol/dataset_manifest.csv`.
5. Check off the corresponding item in `TODO.md` and record any invalid meshes,
   NaNs, or protocol deviations in the dataset README.
