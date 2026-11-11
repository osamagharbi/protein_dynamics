# MAMBA-DSR

MAMBA-DSR is the direct bidirectional-Mamba version of `diffusion_dsr`.
Stage B is replaced completely by a deterministic Mamba regressor:

```text
t -> time embedding + latent-channel tokens -> bidirectional Mamba blocks -> z
```

There is no DDPM/DDIM process, diffusion timestep, residual diffusion model,
or stochastic latent ensemble. Repeated queries at the same time are
deterministic.

For comparison with the published DSR results, use the explicit manifest and
commands in [PROTOCOL.md](PROTOCOL.md). The old Aβ commands below are retained
for the original exploratory held-out-split experiment and must not be mixed
with paper-protocol results.

The bidirectional backend is bundled under `mamba_backend/` and was taken from
the locally used `Mamba3D` implementation. Its CUDA selective-scan kernels
require the `mamba3d` environment (or an equivalent environment containing
PyTorch, `einops`, `causal_conv1d`, `selective_scan_cuda`, `triton`, and the
DSR dependencies).

From this directory:

```bash
conda activate mamba3d
python train_autodecoder.py --case abeta_150 --protocol dsr_abeta \
  --latent-size 8 --smooth-lambda 1.0 --end-iter 800 --seed 0 \
  --run-name abeta_dsr_protocol_v2_mamba

python train_mamba.py \
  --stage-a-ckpt exp/abeta_dsr_protocol_v2_mamba/checkpoints/ckpt_latest.pth \
  --case abeta_150 --seed 0 --run-name abeta_dsr_protocol_v2_mamba

python evaluate_protocol.py \
  --stage-a-ckpt exp/abeta_dsr_protocol_v2_mamba/checkpoints/ckpt_latest.pth \
  --stage-b-ckpt exp/abeta_dsr_protocol_v2_mamba/ckpt_latest.pth \
  --case abeta_150 --gt-case abeta_30_gt --resolution 128 \
  --num-samples 1 --seed 0 --run-name abeta_dsr_protocol_v2_mamba_eval
```

`train_diffusion.py` remains as a compatibility-named entry point and invokes
the same direct-Mamba training code. The source `diffusion.py` module is not
part of this project.
