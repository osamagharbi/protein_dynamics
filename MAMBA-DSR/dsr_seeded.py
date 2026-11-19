#!/usr/bin/env python3
"""Run an unmodified upstream DSR script with fixed random seeds.

Upstream DSR (runner_igr_t.py) never seeds its RNGs, so every run starts from a
different random initialization. This wrapper seeds Python, NumPy and PyTorch
and then executes the script as __main__ with the remaining arguments, from the
current directory (run it with cwd = DSR/):

    python ../MAMBA-DSR/dsr_seeded.py 1 runner_igr_t.py --conf ... --mode train
"""
import os
import random
import runpy
import sys

import numpy as np
import torch

if len(sys.argv) < 3:
    sys.exit(__doc__)
seed, script = int(sys.argv[1]), os.path.abspath(sys.argv[2])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
print(f"[dsr_seeded] seed {seed} -> {os.path.basename(script)}", flush=True)
# Import DSR's modules (lib/...), not this directory's (MAMBA-DSR has its own model.py etc.).
here = os.path.dirname(os.path.abspath(__file__))
sys.path = [os.path.dirname(script)] + [p for p in sys.path if os.path.abspath(p or ".") != here]
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name="__main__")
