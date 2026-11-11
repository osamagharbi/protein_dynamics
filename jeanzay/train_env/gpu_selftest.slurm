#!/bin/bash
#SBATCH --job-name=dsr_train_selftest
#SBATCH --account=acu@v100
#SBATCH --qos=qos_gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
# V100 check of the MAMBA-DSR fast-path kernels + DSR decoder, forward and backward.
# Submit from the repo root:  sbatch jeanzay/train_env/gpu_selftest.slurm
set -euo pipefail
module purge
REPO="${SLURM_SUBMIT_DIR}"
source "$REPO/jeanzay/train_env/env_train.sh"
cd "$REPO/MAMBA-DSR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python - <<'PY'
import sys, os, time, torch
sys.path.insert(0, os.path.abspath("../DSR"))
from model import MambaLatentPredictor
from lib.models.network import ImplicitNet
dev = "cuda"
m = MambaLatentPredictor(latent_dim=8).to(dev)
t = torch.rand(32, device=dev)
z = m(t); z.square().mean().backward()
g = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
assert z.shape == (32, 8) and torch.isfinite(z).all() and g > 0, (z.shape, g)
torch.cuda.synchronize(); t0 = time.time()
for _ in range(50):
    m.zero_grad(); m(t).square().mean().backward()
torch.cuda.synchronize()
print(f"MAMBA fast path OK: out {tuple(z.shape)}, grad-sum {g:.3e}, {1000*(time.time()-t0)/50:.1f} ms/iter")
print("ImplicitNet import OK:", ImplicitNet.__name__)
print("SELFTEST PASSED on", torch.cuda.get_device_name(0))
PY
