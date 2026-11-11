# Training on Jean-Zay (MAMBA-DSR + DSR baseline)

Prerequisites: data prepared by `jeanzay/data_prep` (datasets/DSR_processed/<case>/meta.json)
and the env from `jeanzay/train_env` (GPU self-test passed).

```bash
cd $WORK/protein_dynamics
SMOKE=1 CASES=adk_dims ./jeanzay/train/submit.sh   # 30-min qos_gpu-dev plumbing test first
./jeanzay/train/status.sh                          # progress table (+ squeue)
./jeanzay/train/submit.sh                          # MAMBA-DSR on every prepared dataset
METHOD=dsr CASES="adk_equilibrium ifabp_water" ./jeanzay/train/submit.sh   # DSR baseline
```

* One case = one GPU; each case is a chain of `NCHAIN` (4) x 20 h `qos_gpu-t3` jobs linked
  `afterany`. Every link resumes from the last checkpoint (Stage A saves every 200 epochs and
  stops `RESERVE_MIN`=90 min before the walltime; DSR is stopped 5 min before and resumes with
  `--is_continue`). Spare links of a finished case exit in ~1 min.
* Protocol = `MAMBA-DSR/paper_protocol.py`: fixed paper rules for Aβ / AdK-eq / I-FABP, the
  data-prep registry rule for the other prepared datasets, and a labelled NON-PAPER default
  (first 1500, every 10th) for GPCRmd. Evaluation = DSR paper task (reconstruct the training
  frames, res 128, normalized coordinates).
* Outputs: `MAMBA-DSR/exp/<case>_paper_mamba_stage{A,B}<tag>/`, metrics in
  `reconstructions/MAMBA-DSR_<case>_*/metrics.json`, one-line summary in
  `MAMBA-DSR/exp/paper_summary/<case><tag>.csv`. DSR: `DSR/exp/dsr_paper/<case>/`.
* Logs: `jeanzay/train/logs/<method>_<case><tag>/<jobid>.out`; markers DONE / FAILED in
  `jeanzay/train/state/<method>_<case><tag>/`. A FAILED case is skipped by its remaining links;
  re-running `submit.sh` for it clears the marker.
* Large meshes (NhaA, YiiP, GPCR) may need `CONSTRAINT=v100-32g`.
