# AdK equilibrium

## Comparable protocol

- source crop: first 1500 frames;
- training frames: `0,10,...,1490` (150 frames);
- evaluation: the same 150 training frames, matching DSR Table 1;
- metric space: normalized;
- reconstruction resolution: 128;
- seed: 0.

## MAMBA-DSR result

- IoU: `0.8061159444 ± 0.0389827346`;
- Chamfer: `0.0011089605 ± 0.0005699346`;
- NC: `0.8204910757 ± 0.0335414882` robust diagnostic aggregate.

Frame 1060 has three zero-length ground-truth vertex normals. The official
metric is therefore NaN for that frame; the aggregate above excludes those
three invalid normal entries and records the policy explicitly.

## Artifacts

- Stage A: see [`../../artifacts/artifact_index.csv`](../../artifacts/artifact_index.csv).
- Stage B: see [`../../artifacts/artifact_index.csv`](../../artifacts/artifact_index.csv).
- Full metrics: `metrics.json` in the linked reconstruction artifact.
