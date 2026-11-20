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

## Held-out evaluation (after training)

Frames the model never saw, evaluated from the finished checkpoints (no retraining):
interpolation (midpoints between training frames), future prediction (next 1-10 and
11-20 frames, both as thinned steps and as raw frames, since the DSR paper does not
say which), and reference baselines that copy the nearest / last training frame.

    ./jeanzay/train/submit_heldout.sh                 # every finished seed-0 run (2 x 10 h chain each)
    CASES="adk_dims ifabp_water" ./jeanzay/train/submit_heldout.sh
    ./jeanzay/train/status.sh                         # progress (bottom section)
    cd MAMBA-DSR && python eval_heldout.py --report   # results table

Frames without a ground-truth mesh in the prepared dataset are skipped and counted.

    METHOD=dsr ./jeanzay/train/submit_heldout.sh      # DSR baseline, same frames (after its training is DONE)
    MODE=hold_t ./jeanzay/train/submit_heldout.sh     # diagnostic, future frames: decoder time held at t=1
    MODE=hold_z ./jeanzay/train/submit_heldout.sh     # diagnostic: Mamba latent held at t=1, decoder time free
    MODE=hold_all ./jeanzay/train/submit_heldout.sh   # diagnostic: decoder time AND Mamba latent held at t=1
    MODE=lerp ./jeanzay/train/submit_heldout.sh       # ablation, interpolation: Stage-A codes linearly
                                                      #   interpolated, no Stage B ("code-lerp")

`--report` merges every method into one table per dataset, then prints mean +- sd over
seeds (training frames and held-out) and paired per-frame comparisons (MAMBA-DSR vs DSR,
vs code-lerp, vs copy baselines; Wilcoxon signed-rank) plus one across-dataset line per pair.
Re-submitting is cheap: evaluations with a DONE marker exit at once.

## DSR baseline with more seeds

Upstream DSR never seeds its RNGs; training now goes through `MAMBA-DSR/dsr_seeded.py`,
which seeds Python/NumPy/PyTorch and runs the unmodified `runner_igr_t.py`. Extra seeds
get their own directories (`DSR/exp/dsr_paper/<case>_s1`, `dsr_<case>_s1.conf`):

    METHOD=dsr SEED=1 CASES="adk_dims adk_froda ifabp_water" ./jeanzay/train/submit.sh
    METHOD=dsr SEED=1 ./jeanzay/train/submit_heldout.sh      # after those are DONE

The seed-0 DSR runs were trained before this wrapper existed (unseeded), which is an
equally valid random draw.

## MAMBA-Flow (new architecture: one model for all datasets)

`MAMBA-DSR/run_flow.py` + `MAMBA-DSR/flow/`. Same training frames, normalization, grid and
metrics as MAMBA-DSR / DSR, so its numbers go straight into `eval_heldout.py --report`.

* **Stage 1 - shape VAE** (`flow/nets.py: ShapeVAE`, 25M parameters): 256 fixed anchor points
  per protein (farthest-point sampling over its training surfaces); the anchors cross-attend
  to 8,192 surface points (Fourier position + normal) -> 6 self-attention layers -> 256 latent
  tokens x 32 per frame. Decoder: 6 self-attention layers over the tokens, a query point
  cross-attends to them -> signed distance. Trained on the training frames of ALL datasets
  with DSR's IGR loss (manifold + normals + eikonal) + a small KL term. No time input.
* **Stage 2 - dynamics** (`flow/nets.py: SpaceTimeFlow`, 18.5M parameters): flow matching over
  short sequences of token grids (frames x 256 tokens). Each block: adaLN(flow time, frame
  spacing) -> attention across the 256 tokens of a frame -> bidirectional Mamba across the
  frames of each token -> MLP. Observed frames are context, the rest are generated; tasks
  (forecast / fine forecast / interpolation) are sampled at random strides so that test-time
  queries (midpoints, next thinned frames, next raw frames) are in the training distribution.
  Targets are residuals from a latent baseline (interpolated / last context latent).
* Not implemented from the design note: per-frame Kabsch superposition (prepared data keeps no
  atom coordinates, 4/8 trajectories are already core-fitted at source, and scoring unseen
  frames needs their original pose) and ESM conditioning (8 proteins; the context frames
  identify the protein). Both matter for ATLAS-scale training, not for these 8 datasets.

```bash
SMOKE=1 METHOD=flow ./jeanzay/train/submit.sh        # 30-min dev-QoS plumbing test, tiny model
./jeanzay/train/status.sh                            # row flow_all_smoke
METHOD=flow ./jeanzay/train/submit.sh                # the real run: ONE chain "flow_all" (4 x 20 h)
METHOD=flow SEED=1 ./jeanzay/train/submit.sh         # second seed (run flow_s1)
METHOD=flow ./jeanzay/train/submit_heldout.sh        # after DONE: held-out eval, one job per dataset
cd MAMBA-DSR && python eval_heldout.py --report      # MAMBA-Flow rows next to MAMBA-DSR / DSR
```

* The chain runs `cache -> shape -> encode -> dynamics -> eval` (training frames); every
  stage resumes from its checkpoint (saved every 30 min and when the link's time budget ends;
  exit 3 = continue in the next link). Outputs: `MAMBA-DSR/exp/flow/flow<tag>/` (`shape/`,
  `latents/`, `dynamics/`, `config.json`, `cases.txt`), shared data cache
  `MAMBA-DSR/exp/flow/cache/<case>/`, training-frame summary
  `MAMBA-DSR/exp/flow_summary/<case><tag>.csv`, per-frame metrics in
  `reconstructions/MAMBA-Flow_<case>_flow<tag>_{train,heldout}/`.
* Config overrides go through `FLOW_SET` ('+'-joined `key=value`, dotted keys), e.g.
  `FLOW_SET="shape_steps=60000+dyn_model.temporal=attn"` (temporal attention instead of
  Mamba: the ablation that tests whether Mamba matters). All defaults: `DEFAULTS` in
  `run_flow.py`; `python run_flow.py --check` prints the plan without a GPU.
* Held-out rows: `MAMBA-Flow` (mean signed-distance field of 4 samples), `MAMBA-Flow [sample]`
  (one sample), `MAMBA-Flow [ref]` (latent baseline only: what Stage 2 adds is the
  difference), `MAMBA-Flow [AE]` (oracle: the held-out frame itself encoded and decoded =
  ceiling of Stage 1; not a prediction). Paired tests include MAMBA-Flow vs DSR, vs MAMBA-DSR,
  vs its [ref] and vs the copy baselines.
