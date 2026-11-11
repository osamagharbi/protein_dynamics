# Training env on Jean-Zay (MAMBA-DSR / Diffusion-DSR / DSR)

Quota-light, no CUDA compilation: the IDRIS module `pytorch-gpu/py3/2.1.1`
(torch 2.1.1, CUDA 11.8, python 3.11, C++11 ABI) + `pip --user` extras in
`$WORK/envs/dsr_train` + the official prebuilt `mamba-ssm 1.1.1` / `causal-conv1d 1.1.1`
wheels matching that build (both include sm_70 = V100).
`MAMBA-DSR/mamba_backend` only needs their `selective_scan_cuda` / `causal_conv1d_cuda` kernels.

```bash
cd $WORK/protein_dynamics            # repo root, after ./setup_dsr.sh
./jeanzay/train_env/setup_train_env.sh     # login node, a few min, extras in $WORK/envs/dsr_train
sbatch jeanzay/train_env/gpu_selftest.slurm   # qos_gpu-dev, <10 min
cat dsr_train_selftest_*.out                  # expect "SELFTEST PASSED on Tesla V100..."
```
Before training: `source jeanzay/train_env/env_train.sh`.
