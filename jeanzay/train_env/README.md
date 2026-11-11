# Training env on Jean-Zay (MAMBA-DSR / Diffusion-DSR / DSR)

No CUDA compilation: python 3.10 + torch 2.1.2/cu118 + the official prebuilt
`mamba-ssm 1.1.1` and `causal-conv1d 1.1.1` wheels (both built for sm_70 = V100).
`MAMBA-DSR/mamba_backend` only needs their `selective_scan_cuda` / `causal_conv1d_cuda` kernels.

```bash
cd $WORK/protein_dynamics            # repo root, after ./setup_dsr.sh
./jeanzay/train_env/setup_train_env.sh     # login node, ~10 min, env in $WORK/envs/dsr_train
sbatch jeanzay/train_env/gpu_selftest.slurm   # qos_gpu-dev, <10 min
cat dsr_train_selftest_*.out                  # expect "SELFTEST PASSED on Tesla V100..."
```
Before training: `source jeanzay/train_env/env_train.sh`.
