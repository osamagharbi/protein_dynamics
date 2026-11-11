# DSR comparison protocol

The living comparison ledger is maintained in
[`comparison/`](comparison/README.md). It contains the TODO plan, dataset
manifest, paper reference tables, result summaries, and artifact links.

The comparison has two distinct parts:

1. **Same-data method comparison:** official DSR and MAMBA-DSR are trained on
   the same selected frames and evaluated with the same normalized metrics.
2. **Paper-protocol reproduction:** the selected frames follow the sampling
   rules documented in the DSR paper, not the earlier exploratory Aβ split.

The official DSR code uses one latent code per sequence. MAMBA-DSR keeps its
per-frame Stage-A latent table because Stage B must learn a time-to-latent
mapping. This is an architectural difference; it does not change the data,
time normalization, mesh resolution, or metric space in the comparison.

## Paper sampling rules

The rules are encoded explicitly in `paper_protocol.py`:

| Case | Protocol frames | Training frames |
|---|---:|---:|
| `abeta_30` official demo | 149 (`0..148`) | `0,5,...,145` (30) |
| `abeta_wt` | first 2000 | all first 2000, because the paper states “first 2000” and gives no thinning interval |
| `abeta_e22g` | first 2000 | all first 2000, same documented ambiguity |
| `adk_equilibrium` | first 1500 | `0,10,...,1490` (150) |
| `ifabp_water` | first 500 | `0,5,...,395` (80) |

For the reconstruction table, the DSR paper reports reconstruction on the
training frames. It is therefore valid to include those frames in the paper
comparison. Held-out interpolation is a separate experiment and must be
reported separately.

The local AdK trajectory now contains the required 1500 frames. The exact AdK
comparison has been completed and is registered in
`comparison/results/comparison_summary.csv`. `--allow-truncated-local-data`
remains available only for an explicitly labeled non-exact diagnostic run.

## Metrics

Both evaluators use the DSR implementation at resolution 128:

- IoU: higher is better;
- symmetric Chamfer distance: lower is better;
- normal consistency: higher is better.

All three metrics are computed after mapping prediction and ground truth to
the same normalized coordinate system. This is required for comparison with
the official DSR evaluator. The older `--metric-space physical` option is
retained only for historical MAMBA-vs-Diffusion-DSR reports.

## Disk-safe commands

First validate the available data and create only symlinked train views:

```bash
cd /home/ogharbi/projects/protein_dynamics/MAMBA-DSR
python run_multicase.py --gpu 0 --only abeta_30 ifabp_water --dry-run
python run_dsr_paper.py --gpu 0 --only abeta_30 ifabp_water --dry-run
```

The actual MAMBA-DSR paper-protocol run is sequential:

```bash
python run_multicase.py --gpu 0 --stage-a-epochs 4000 --stage-b-epochs 5000 \
  --resolution 128 --seed 0 --only abeta_30 ifabp_water
```

The evaluator removes each generated OBJ after computing its metrics. Add
`--keep-reconstructions` only when meshes are needed for inspection. The
symlinked views never duplicate the source OBJ files.

The official baseline uses the original DSR runner:

```bash
python run_dsr_paper.py --gpu 0 --epochs 4000 --resolution 128 \
  --seed 0 --only abeta_30 ifabp_water
```

The two runs write metrics under `reconstructions/` and can be compared only
after checking that their `train_indices`, `resolution`, `metric_space`, and
protocol frame counts match.
