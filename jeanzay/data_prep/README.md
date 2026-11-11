# Protein-surface-dynamics data prep on Jean-Zay (DSR / MAMBA-DSR)

Downloads every trajectory used by DSR (plus ATLAS on request), converts each
frame into a **protein-only** PyMOL molecular surface (`traj_<i>.obj`), and lays
the result out exactly where `MAMBA-DSR/paper_protocol.py` expects it.

```
PROC_ROOT/<dataset>/gt/traj_<i>.obj              one surface per frame (i = 0..N-1)
PROC_ROOT/<dataset>/source.json                  raw files, frame count, atom selection
PROC_ROOT/<dataset>/meta.json                    QC + provenance + DSR normalisation (s, t)
PROC_ROOT/<dataset>/views/paper_train/obj/       symlinks to the DSR-paper training frames
PROC_ROOT/_plans/<timestamp>/                    task lists, job manifest, all logs
```

## 1. Copy the code (from your machine)
```bash
rsync -avP ~/projects/protein_dynamics/jeanzay <jz>:'$WORK/protein_dynamics/'
```

## 2. One-time env (Jean-Zay login node, has internet)
```bash
cd $WORK/protein_dynamics/jeanzay/data_prep
chmod +x *.sh *.py        # needed if the copy dropped the exec bits (scp, VS Code upload, Windows mount)
./setup_env.sh              # conda env in $WORK/envs/dsr_data + selftest -> writes env.sh
```
If no conda module is found: `module avail 2>&1 | grep -iE "conda|forge"`, then
`CONDA_MODULE=<name> ./setup_env.sh`.

## 3. Launch
```bash
DRY_RUN=1 ./run_data_prep.sh     # plan + sbatch --test-only; submits nothing
./run_data_prep.sh               # submit
```
Each raw source gets its own independent chain:

| step | where | what |
|---|---|---|
| download | `prepost` partition (internet, not charged) | fetch + checksum + index (frame count, selection, XTC offsets) |
| convert | Slurm array, 1 task = `CHUNK_SIZE` frames, `cpus-per-task` PyMOL processes | `gt/traj_<i>.obj` |
| finalize | `prepost` | validate all frames, bbox outliers, `meta.json`, train view |

Monitor: `squeue -u $USER` and `python prep.py status --proc-root $PROC_ROOT`.
Everything is resumable: re-running the same command skips finished downloads and frames.

## Default dataset set (`DATASETS=paper`, 22 datasets, ~30k frames)

| dataset | source | frames | paper training rule |
|---|---|---|---|
| abeta_wt, abeta_e22g | Harvard Dataverse (md5-checked) | first 2000 | all (paper states no thinning) |
| adk_equilibrium | MDAnalysisData | first 1500 | every 10th |
| adk_dims / adk_froda | MDAnalysisData (`dims0001`, `pathway1`) | 102 / 142 | every 2nd of 80 / every 3rd of 120 |
| ifabp_water | MDAnalysisData | 500 | every 5th of 400 |
| nhaa_equilibrium, yiip_equilibrium | MDAnalysisData | first 1500 | every 10th |
| dryad_<14 proteins> | Dryad (see below) | first 1500 of `<p>_native.xtc` | every 10th |

Other groups: `dryad_excluded` (9 proteins DSR screened out), `gpcrmd`, `atlas`, `all`;
or list dataset names: `DATASETS="adk_dims ifabp_water"`.

## Sources that need a manual step
* **DRY_AD_MD (Dryad)**: Dryad rejects anonymous scripted downloads. Either
  `export DRYAD_TOKEN=...` (Dryad API account) or upload the local copy once:
  `rsync -avP ~/projects/protein_dynamics/datasets/DSR/DRY_AD_MD/trajectories.tar.gz <jz>:'$WORK/protein_dynamics/datasets/raw/DRY_AD_MD/'`
  (md5 is verified before extraction).
* **GPCRmd**: no bulk API. Put `<id>.xtc` + topology `<id>.pdb|psf|gro` into
  `RAW_ROOT/GPCRmd/`, then `DATASETS=gpcrmd ./run_data_prep.sh`. Unstaged ones are skipped.
* **ATLAS** (MoE-DSR, 1,938 proteins, ~0.9 GB zip each): opt-in,
  `DATASETS=atlas ATLAS_LIMIT=50 ./run_data_prep.sh` (or `ATLAS_IDS=file`). Keeps
  `<id>.pdb` + replica `R$ATLAS_REPLICA` only; every `$ATLAS_STRIDE`th frame (default
  100 -> 101 surfaces/protein). Check `idrquota -w` before going large (inodes too).

## Knobs (env vars)
| var | default | meaning |
|---|---|---|
| `RAW_ROOT` / `PROC_ROOT` | `$WORK/protein_dynamics/datasets/{raw,DSR_processed}` | storage |
| `PHASE` | `all` | `download` · `convert` (+finalize) · `finalize` |
| `PROC_TARGET` | `gpu` | `gpu` = `acu@v100`, 10 CPUs + 1 V100 per task (any V100; `CONSTRAINT=v100-32g` to restrict) (known to work; CPU work billed as GPU-hours, ~10 GPU-h total) · `cpu` = `CPU_ACCOUNT` on `cpu_p1` · `prepost` = uncharged, shared nodes |
| `CHUNK_SIZE` / `CONCURRENCY` | 200 / 40 | frames per task / running tasks per source |
| `DL_TARGET` | `prepost` | `login` = download on the login node (run inside tmux) if prepost has no internet |
| `ACCOUNT` / `DL_ACCOUNT` | `acu@v100` | Slurm accounts |

## What differs from the earlier local conversion
* **Protein-only surfaces** (`selection="protein"`). The earlier
  `scripts/convert_dsr_datasets_parallel.py` surfaced the whole system: NhaA and
  YiiP included POPE/POPG lipids (YiiP and I-FABP also water), giving ~88 MB
  frames of membrane+protein. NhaA is now ~23 MB/frame.
* For protein-only inputs the result is byte-for-byte the same surface
  (AdK equilibrium frames 0–1: identical vertex counts and bounding boxes).
* Frames are renumbered 0..N-1 after crop/stride; `meta.json` maps back
  (`orig = start + i * stride`).
* `finalize` flags frames whose bounding box is >1.5x the median (broken PBC or
  unfolding). It reports them and does not fix them.

## Using the output with MAMBA-DSR
Copy/rsync `MAMBA-DSR/` next to `datasets/` (`$WORK/protein_dynamics/`), so
`paper_protocol.py` resolves `datasets/DSR_processed/<name>/gt`. For DSR's
`RawDataset`, pass `data_dir=PROC_ROOT/<name>/views/paper_train` together with `s`/`t`
from `meta.json -> dsr_normalization` (same rule as `run_multicase.normalization`).
