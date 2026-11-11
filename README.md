# protein_dynamics

Protein surface dynamics as implicit neural representations: DSR baseline, Diffusion-DSR and MAMBA-DSR.

| folder | what |
|---|---|
| `MAMBA-DSR/` | Stage A per-frame latent auto-decoder + Stage B bidirectional-Mamba t→latent |
| `diffusion_dsr/` | Stage A + MeanPredictor + residual latent DDPM |
| `jeanzay/data_prep/` | Jean-Zay pipeline: download + protein-only PyMOL surfaces for every DSR dataset (see its README) |
| `patches/`, `setup_dsr.sh` | clones upstream [DSR](https://github.com/Sundw-818/DSR) into `DSR/` and applies numpy/scikit-image compat fixes |
| `CONSTITUTION.md` | evaluation rules (held-out frames, evidence labels) |

Data, checkpoints and outputs are not in git. On Jean-Zay they live in `$WORK/protein_dynamics/datasets/`.

```bash
git clone https://github.com/osamagharbi/protein_dynamics.git && cd protein_dynamics
./setup_dsr.sh
```
