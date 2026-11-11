# Fixed comparison protocol

## Objective

Compare the MAMBA-DSR architecture with DSR on the same data and evaluation
definitions. Matching the optimizer, number of updates, or batching is not a
requirement.

## Required invariants

Every comparable run must record and preserve:

1. source trajectory and topology;
2. converted OBJ mesh directory and frame count;
3. exact training and evaluation frame indices;
4. spatial scale and translation;
5. normalized time mapping;
6. reconstruction resolution (`128` for the paper comparison);
7. metric space (`normalized`);
8. IoU, symmetric Chamfer distance, and normal consistency;
9. stage-A and stage-B checkpoints;
10. seed and command line.

The current data and artifact state is tracked in
[`data_readiness_matrix.csv`](data_readiness_matrix.csv). Update it whenever a
trajectory is converted, trained, evaluated, or blocked by missing source
data.

## Paper reconstruction protocol

The DSR paper evaluates reconstruction on the selected training frames. This
is intentionally different from held-out interpolation and must be labeled
`training_frame_reconstruction`.

| Dataset | Source crop | Training indices |
|---|---:|---|
| AdK equilibrium | first 1500 | `0,10,...,1490` (150) |
| NhaA equilibrium | first 1500 | `0,10,...,1490` (150) |
| YiiP equilibrium | first 1500 | `0,10,...,1490` (150) |
| I-FABP water | first 400 | `0,5,...,395` (80) |
| DIMS | first 80 | every second frame |
| FRODA | first 120 | every third frame |
| DRYAD_MD | paper-defined source crop | verify from source/archive |

For Aβ, the paper states that the first 2000 frames were used but does not
fully specify the thinning rule in the main text. Aβ results must therefore
carry an explicit split note until the source protocol is verified.

## Metric policy

Use the DSR metric implementation at resolution 128. Store the raw metric
output first. If a mesh contains zero-length normals, retain the official NaN
result and additionally report a robust diagnostic value with the excluded
vertices/count documented. Do not silently substitute one for the other.

## Artifact policy

Large checkpoints and reconstruction directories remain in the existing
`exp/` and `../reconstructions/` locations. This comparison directory stores
small summaries, manifests, notes, and links to those artifacts to avoid
duplicating data and worsening the disk bottleneck.
