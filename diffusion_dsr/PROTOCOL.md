# Frozen DSR comparison protocol

This is the protocol implemented by `protocol.py`, `train_autodecoder.py`,
`train_diffusion.py`, and `evaluate_protocol.py`.

## Split

The existing DSR Aβ baseline trains on:

```text
0, 5, 10, ..., 145
```

That is 30 training frames. The valid trajectory contains frames 0 through
148, so the evaluation set is the 119 indices not divisible by 5.

Stage A must load only the 30 training frames. Stage B must use only the
latents learned for those 30 frames. Stage C evaluates only the 119 held-out
frames. Frame 149 is excluded because the repository has no
`traj_149.obj` ground-truth mesh.

Time is normalized using the first and last training frames, 0 and 145, as
DSR does. Therefore frames 146–148 receive times slightly above 1.0; this is
intentional and matches DSR's reconstruction script.

## Resolution and metrics

The default surface extraction resolution is 128, matching the saved DSR
res128 artifacts and the current Diffusion-DSR metrics logs.

Both methods use DSR's metrics:

- volumetric IoU, higher is better;
- symmetric Chamfer distance, lower is better;
- normal consistency, higher is better.

IoU is computed by the DSR metric implementation on its 128³ normalized grid.
Chamfer uses the DSR default of 300,000 sampled surface points.

## Commands

From `projects/protein_dynamics/diffusion_dsr`:

```bash
python train_autodecoder.py \
  --case abeta_150 \
  --protocol dsr_abeta \
  --latent-size 8 \
  --smooth-lambda 1.0 \
  --end-iter 800 \
  --seed 0 \
  --run-name abeta_dsr_protocol_v2
```

The Stage-A checkpoint records the split in
`train_frame_indices` and `eval_frame_indices`.

```bash
python train_diffusion.py \
  --stage-a-ckpt exp/abeta_dsr_protocol_v2/checkpoints/ckpt_latest.pth \
  --case abeta_150 \
  --seed 0 \
  --run-name abeta_dsr_protocol_v2
```

Stage B refuses a Stage-A checkpoint without split metadata unless
`--allow-legacy-split` is explicitly supplied for exploratory work.

To evaluate all held-out frames in one process:

```bash
python evaluate_protocol.py \
  --stage-a-ckpt exp/abeta_dsr_protocol_v2/checkpoints/ckpt_latest.pth \
  --stage-b-ckpt exp/abeta_dsr_protocol_v2/ckpt_latest.pth \
  --case abeta_150 \
  --gt-case abeta_30_gt \
  --resolution 128 \
  --num-samples 1 \
  --seed 0 \
  --run-name abeta_dsr_protocol_v2_eval
```

The evaluator writes `metrics.csv` and `metrics.json`, plus one reconstruction
directory per held-out frame. For an ensemble, set `--num-samples` above one;
the per-frame CSV reports the mean and standard deviation across samples.

## Fairness boundary

The existing DSR checkpoint is not retrained. This makes the comparison a
fixed-baseline comparison rather than a new multi-seed DSR benchmark. The
Diffusion-DSR run must nevertheless obey the same frame split, target frames,
resolution, coordinate normalization, and metric implementation. The oracle
is excluded from the main comparison because it uses the target frame's
learned latent.
