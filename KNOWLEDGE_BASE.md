# Knowledge Base: Deep Learning for Protein Molecular Surface / Structure Dynamics

This document summarizes the state of the art surveyed for this project, based on:

- **Repo 1 — `DSR/`**: [Sundw-818/DSR](https://github.com/Sundw-818/DSR) — code for *"DSR: Dynamical Surface Representation as Implicit Neural Networks for Protein"* (Sun, Huang, Li, Gong, Ye — NeurIPS 2023).
- **Repo 2 — `ProtMD/`**: [smiles724/ProtMD](https://github.com/smiles724/ProtMD) — code for *"Pre-Training of Equivariant Graph Matching Networks with Conformation Flexibility for Drug Binding"* (Wu, Jin, Jiang, Jin, Tang, Niu, Liu, Zhang, Zeng, Li — Advanced Science 2022, a.k.a. "ProtMD").
- **`resources/` papers** (3 PDFs):
  1. `NeurIPS-2023-dsr-...pdf` — the DSR paper above.
  2. `ADVS-9-2203796.pdf` — the ProtMD paper above.
  3. `00595-WuF.pdf` — *"Generalized Implicit Neural Representations for Dynamic Molecular Surface Modeling"* (Wu, Hu, Li — AAAI 2025), which introduces **MoE-DSR**, a direct follow-up/extension of DSR by an overlapping author (Fang Wu, who also wrote ProtMD).

These three papers represent **two distinct research lines** for modeling protein dynamics with deep learning, plus one paper that bridges/extends the first line. Both lines are candidate foundations for this project.

---

## 1. The Big Picture: Why Model Protein Dynamics with Deep Learning?

Proteins are not static — their function (catalysis, transport, ligand binding, signaling) emerges from continuous atomic-scale motion. The traditional tool for studying this is **molecular dynamics (MD) simulation**:

- **Ab initio / first-principles MD** (density functional theory, e.g. VASP, GPAW): accurate but computationally explosive — cost scales with the number of electronic degrees of freedom.
- **Classical force-field MD** (AMBER, CHARMM, GROMOS): faster, but still requires femtosecond integration steps (to resolve H-bond vibrations), so long-timescale or large-system simulation is expensive. Even state-of-the-art specialized hardware (Anton 3, 512 nodes) only reaches ~100 µs/day for a 1M-atom system.
- **Coarse-grained MD** (CGnet, DeePCG, ARCG, flow-matching, diffusion force fields): trades resolution for speed by eliminating some degrees of freedom — loses all-atom detail.
- **ML force-field methods** (Behler-Parrinello, DTNN, SchNet, sGDML): learn potential energy surfaces / forces to accelerate simulation, but mostly validated on small molecules, not large proteins.

**The core gap both research lines attack**: existing deep learning approaches for MD are largely limited to *small molecules* or *coarse-grained* representations, and cannot practically model the *long-term, large-scale, atomic-or-surface-resolution* dynamics of real proteins. The two lines below attack this gap from different representational angles.

---

## 2. Research Line A — Implicit Neural Surface Representations (DSR → MoE-DSR)

### 2.1 Core idea

Instead of modeling proteins as discrete particles (atoms), represent the protein **surface** as the zero-level set of a **signed distance function (SDF)**, and make that SDF a continuous function of **space *and* time**, parameterized by a neural network (an "implicit neural representation," INR — the same family as NeRF/SIREN/DeepSDF in computer vision).

Why surfaces? Molecular surfaces (not buried atomic structure) are what governs protein–protein and protein–ligand interactions ("key-lock" complementarity; MaSIF-style geometric deep learning). Surfaces had previously been used for docking/interaction fingerprinting (MaSIF) but never for modeling *dynamics*.

### 2.2 DSR (NeurIPS 2023) — method details

**Formalism.** Let Ω = [-1,1]³ (space), τ = [-1,1] (time). A protein surface at time *t* is a 3D manifold M_t. Its SDF is:

```
f_Mt(x) =  d(x, M_t)   if x outside M_t
           0            if x on M_t
          -d(x, M_t)   if x inside M_t
```

The zero-level set {x | f_Mt(x) = 0} *is* the surface at time *t*.

**Learning without pre-computed SDF ground truth.** Computing SDF via grid discretization + approximation (e.g. 8SSEDT) is expensive and error-prone for rugged protein surfaces. DSR instead learns the SDF **directly from raw point clouds** (surface points + normals extracted from MD trajectory frames via PyMOL `show_as('surface')`), using the **Eikonal-regularized loss from IGR** (Implicit Geometric Regularization, Gropp et al. 2020):

```
ℓ(θ) = ℓ_X(θ) + λ·E_x(‖∇_x f(x,t;θ)‖ - 1)²  +  α‖z‖
ℓ_X(θ) = (1/|I|) Σ_i ( |f(x_i,t;θ)| + τ‖∇_x f(x_i,t;θ) - n_i‖ )
```

- Term 1 (`ℓ_X`): drives f→0 on the known surface points, and drives ∇f to match the ground-truth surface normals.
- Term 2 (Eikonal term): forces ‖∇f‖=1 everywhere, the defining gradient property of a true SDF — this is what makes the *unsupervised* SDF learning well-posed (no pre-computed ground-truth distance values are needed at all).
- Term 3: L2 penalty on a per-protein/per-sequence **learnable latent code z** (size 192) that lets one model represent multiple distinct proteins/trajectories.

**Architecture ("DSR Model").** An MLP `f(x,t;θ): R^(4+dz) → R` (3 spatial + 1 time + latent dim), 8 layers × 512 hidden units, one skip connection from input to the middle layer, **Softplus(β=100)** activations (smoother than ReLU — controls surface smoothness vs. sharpness), geometric weight initialization (from IGR/SAL). A separate SIREN-style sine-activation variant (`decoder_siren.py`, ω=30) is also implemented in the repo. Gradients ∇_x f are obtained via PyTorch autograd (`lib/models/network.py: gradient()`).

**Inference.** Discretize space into an N³ grid, add the desired time value, query the network, and run **Marching Cubes** (scikit-image) to extract the zero-level-set mesh. Because the model is continuous in (x,t), reconstruction/interpolation/extrapolation can be done at **arbitrary spatial resolution and arbitrary time** — this is the key practical payoff over precomputed-grid SDF approaches.

**Codebase (`DSR/`)**:
- `data_process.py` — turns an MD trajectory (`.pdb`, loaded via `mdtraj`) into per-frame surface `.obj` meshes via PyMOL (`cmd.show_as('surface')`); builds `datasets/<case>/obj/` (training, subsampled every 5 frames) and `datasets/<case>_gt/` (ground truth, all frames).
- `lib/dataset.py` — `RawDataset` loads all `.obj` frames per sequence into memory, computes global normalization scale/translation (`s`, `t`) from bounding box across all frames.
- `lib/models/network.py` — `ImplicitNet` (ReLU/Softplus MLP variant used by `runner_igr_t.py`).
- `lib/models/decoder_siren.py` — SIREN/DeepSDF-style alternative decoder.
- `lib/models/sample.py` — `Sampler` classes that draw the non-manifold ("space") points needed for the Eikonal term (Gaussian around each surface point with σ = distance to k=50th nearest neighbor, mixed with a uniform distribution — this *is* the distribution "D" referenced in the paper).
- `lib/metrics.py` — IoU / Chamfer distance / Normal Consistency implementations, plus an `inside_mesh`-derived fast mesh-containment test (compiled Cython, from Occupancy Networks) used for IoU.
- `runner_igr_t.py` — main `Runner` class: training loop (manifold + Eikonal + normals + latent-regularization losses), reconstruction mode, interpolation/extrapolation mode, checkpointing, wandb logging, loss-curve plotting.
- `confs/abeta.conf` (HOCON format via `pyhocon`) — all hyperparameters: network dims `[512]*8`, skip at layer 4, latent size 192, λ=0.1 (Eikonal), normals_lambda=1, latent_lambda=1e-3, Adam with step LR schedules for network (1e-4) and latent codes (1e-3, both halving every 200 epochs).
- `run_scripts/*.sh` — train / reconstruct / interpolate / eval entry points wrapping `runner_igr_t.py --mode {train,reconstruction,interpolation} --case abeta_30 ...`.
- Demo data included: `datasets/abeta.pdb` + `abeta_0.pdb` (a single Aβ peptide trajectory) and a pretrained checkpoint under `exp/abeta_30/igrt_debug_sphere/.../ckpt_latest.pth`.

**Datasets used in the paper** (far larger than the repo's toy demo):
| Dataset | Description | Frames used |
|---|---|---|
| 500ns Trajectory Data | Aβ(1-40), wild-type + E22G, implicit solvent, 310K | first 2000 frames (wild type) |
| MDAnalysisData: AdK equilibrium | adenylate kinase | 4187 frames (cropped to 2000) |
| MDAnalysisData: AdK conformational transition (DIMS) | dynamic importance sampling | 102 frames |
| MDAnalysisData: AdK conformational transition (FRODA) | rigidity-optimized dynamics | 142 frames |
| MDAnalysisData: I-FABP | intestinal fatty-acid binding protein | 500 frames |
| MDAnalysisData: NhaA | membrane transporter | 5000 frames (cropped) |
| MDAnalysisData: YiiP | membrane transporter, long traj | 9000 frames (cropped) |
| GPCRmd | 4 GPCR-family trajectories, 500ns×3 replicates each | 4 trajectories, multi-protein joint training |
| DRY_AD_MD | 23 small proteins (Jumper et al.); 14 kept after RMSD<1.0 filtering | up to 2000 frames |

**Evaluation metrics** (all standard 3D-shape-similarity metrics, all normalized to [0,1] except Chamfer):
- **Volumetric IoU** — intersection-over-union of reconstructed vs. GT occupied volume (higher=better). Computed via a fast mesh-containment/ray-casting test on an N³ grid (N=128 in `lib/metrics.py`, N=512 in experiments).
- **Chamfer distance** — symmetric nearest-point distance between sampled surface point sets (lower=better).
- **Normal Consistency (NC)** — mean dot-product between GT and predicted surface normals at nearest-point correspondences (higher=better).

**Key results / findings**:
- Best reconstruction: IoU 0.93, Chamfer 0.0009, NC 0.997 on Aβ.
- Across the full 20-protein benchmark, IoU ranges from 0.42 (T0771) to 0.93 (DIMS) — **strong correlation between per-trajectory RMSD (noise/vibration level) and reconstruction quality**: smoother, less noisy trajectories are easier to model. This is the paper's headline limitation.
- Interpolation quality ≈ reconstruction quality, and clearly beats naive linear interpolation.
- Extrapolation (future prediction) degrades with horizon: future 10 frames >> future 11-20 frames.
- Multi-protein joint training (GPCRmd case study, one latent code per protein) works — the model can learn several proteins' dynamics simultaneously and denoises MD's inherent random jitter, recovering smoother "functional motion."

**Explicitly stated limitations** (from the paper's Discussion/Conclusion — important for framing this project's contribution):
1. **Sensitive to trajectory noise** — high-RMSD/high-vibration trajectories are hard to fit.
2. **Poor generalization** — no way to transfer to unseen proteins; a new protein needs its own training (or its own latent code within a jointly-trained batch, but the latent code itself must be learned in-distribution).
3. **No equivariance** — coordinates are fed to the MLP raw; rotating/translating the protein changes the SDF value the model must relearn.
4. **Weak long-horizon extrapolation** — no explicit velocity/motion field is modeled, just position.

### 2.3 MoE-DSR (AAAI 2025, `00595-WuF.pdf`) — how the gaps get addressed

This is **not in either cloned repo** (no public code found for it yet, only the paper) — but it's the natural "next step" reference for this project since it directly targets DSR's stated weaknesses (generalization + equivariance). Written by Fang Wu (Stanford) — same author as ProtMD.

**Key additions over vanilla DSR:**
1. **Mixture-of-Experts (MoE) over protein families.** A *router* looks at the protein's **initial static conformation only** (P₀) and decides which of N_E expert INNs (each an independent DSR-style MLP) should model this protein's dynamics. Standard sparsely-gated MoE (Shazeer et al. 2017 / Switch Transformer style) with top-K gating (K=1-2) + noise for load balancing, plus an auxiliary load-balancing loss L_MoE so experts specialize roughly evenly rather than collapsing to one.
2. **Router = novel geometric surface network ("Q-MPNN").** Borrows from dMaSIF: computes chemical features per surface point from nearby residues, then propagates a **quasi-geodesic message-passing** scheme using an approximate geodesic distance `d_ij = ‖x_i - x_j‖·(2 - ⟨n_i,n_j⟩)` (accounts for surface curvature, not just Euclidean distance) combined with **tensor-product / spherical-harmonics equivariant message passing** (Equiformer-style depthwise tensor product layers) to produce a protein-level feature vector h_P used both for expert-selection and as conditioning input to each expert.
3. **Equivariant experts.** Each expert MLP takes irreducible-representation (irrep) features `‖x‖·SH(x/‖x‖)` instead of raw (x,y,z) — since spherical harmonics are E(3)-equivariant and ‖x‖ is invariant, the resulting INR is **O(3)-equivariant** (robust to rotation/reflection of the input coordinate frame), directly fixing DSR's limitation #3.
4. **Loss** = SDF loss (same manifold + Eikonal + latent-code form as DSR) + λ₂·L_MoE (load balancing), λ₂=0.01.

**Dataset: ATLAS** (Vander Meersche et al. 2024) — the largest all-atom MD database for single-chain proteins (1,390 proteins, chosen for structural diversity by ECOD classification). Split by **sequence identity ≤40%** between train and test — a genuinely held-out generalization test, unlike DSR's per-protein training. 100 snapshots/protein (every 300 of 30,000 timesteps). Splits: 1290/50/50 train/val/test ensembles.

**Results**: MoE-DSR vs. re-trained vanilla DSR baseline on held-out (unseen) proteins — **+30.2% IoU, -39.4% Chamfer, +16.2% NC**. A "naive extension" baseline (just reusing the initial conformation P₀ as the prediction for all future times) gets IoU 0.34 — confirming the task is non-trivial and the gains are real. Ablations show: the Q-MPNN router beats generic point-cloud backbones (PointNet++, PointCNN, Point Transformer, PointMLP, dMaSIF); removing equivariance (raw coords instead of SH irreps) drops IoU from 0.70 to 0.69 (smaller but consistent); more experts (N_E≥16) doesn't help, unlike LLM-scale MoE — attributed to protein MD data being orders of magnitude smaller than LLM corpora.

**Takeaway for this project**: MoE-DSR demonstrates that surface-INR dynamics modeling *can* generalize across protein families (something DSR could not), at the cost of needing a much larger, more diverse training corpus (ATLAS, 1290 proteins) and considerably more architectural machinery (equivariant geometric router + MoE). No public code exists yet for MoE-DSR — reimplementing/reproducing it, or exploring simpler alternatives to reach similar generalization, is an open opportunity.

---

## 3. Research Line B — Pretraining Graph Networks on MD Trajectories for Downstream Property Prediction (ProtMD)

### 3.1 Core idea

Instead of trying to *reconstruct* the full dynamic surface, treat MD trajectories purely as a **self-supervised pretraining signal** for a protein-ligand graph encoder, then transfer the encoder to *static* downstream prediction tasks (binding affinity, ligand efficacy) where MD data is *not* available at inference time. The premise: MD trajectories reveal *flexibility*, and flexibility (not just a single static pose) is what actually determines binding — the old "lock-and-key" model is known to be wrong; binding is dynamic and conformationally flexible.

### 3.2 Method details

**Representation.** At each MD timestep *t*, ligand and receptor are each a graph of atoms with 3D coordinates: ligand graph 𝒢_L=(𝒱_L,ℰ_L), receptor graph 𝒢_R=(𝒱_R,ℰ_R); atom features h (types, electronegativity, etc.); edges = all atom pairs within 4 Å. Crucially, **only the binding pocket is modeled**, not the whole protein: atoms within 6 Å of the ligand — this keeps graphs small (faster training) while retaining the region that changes most during binding.

**Backbone: a variant of E(3)-Equivariant Graph Matching Network (EGMN)**, refined from Ganea et al.'s EGMN. Two changes from the original EGMN:
1. The receptor is **allowed to move** (original EGMN treated the receptor as rigid).
2. **Intra-graph edges** (within 𝒢_L, within 𝒢_R) and **cross-graph edges** (𝒢_LR, ligand↔receptor pairs) are explicitly distinguished and each gets its own message/update functions — rather than naively merging both graphs into one fully-connected graph, which would confound ligand-only and receptor-only relationships. Attention weights (`a_{j→i}`) over cross-graph messages let the model learn *which* atoms across the interface matter.
3. Coordinate updates: `x_i^{l+1} = x_i^l + Σ_j (x_i^l - x_j^l)·φ_x(...)` — standard E(n)-GNN style equivariant coordinate update (this is architecturally the same family as `egnn/egnn_pytorch.py`'s `EGNN`/`EGNN_Network` in the repo, which is what `main.py`/`pretrain.py` actually import and run — `EGMN` in the paper ≈ the repo's `EGNN_Network`, a Lucidrains-style EGNN with optional global linear attention every N layers).

**Two self-supervised pretraining tasks, run jointly:**

1. **Prompt-based denoising conformation generative task (atom-level).** Given conformation at time *t* (with injected Gaussian noise σ, so the model must "denature → renature"), predict the coordinates at time *t+Δt* for a *set* of possible time gaps Δt ∈ {1, 5, 10} (from `confs`: `--prompt` arg). Each Δt has its own learnable **prompt embedding** appended to the atomic features, letting one model condition its prediction horizon (short vs. long term) via the prompt rather than needing separate models or losing horizon-specific nuance under naive multi-task learning. Loss = MSE(predicted next-coords, true next-coords). *Why the noise*: (a) mimics enhanced-sampling MD (energy injection smooths the potential surface, helps escape local minima), (b) prevents the trivial "just copy the input" shortcut, since real adjacent-frame displacements are tiny.
2. **Snapshot ordering task (conformation-level).** Given a shuffled set of n temporally-close conformations, predict their correct temporal order (formulated as pairwise topological-sort constraints solved via an MLP classifier over pairs). This is the MD analogue of NLP's "next sentence prediction" and is meant to capture *global* shape/surface information that the atom-level denoising task might miss.

Ablations in the paper (Table 4) confirm **both tasks help**, and **noise + prompting help more for linear-probing than fine-tuning** — i.e., they matter most when you can't update the backbone weights.

**Pretraining data**: MD trajectories of **64 protein-ligand pairs from PDBbind**, simulated with **Amber** (NPT ensemble, 303.15K, 1 atm, SHAKE algorithm for H-bonds), 100ns production runs, snapshots saved every 1ps → **10,000 timesteps per pair, 62.8K total snapshots**. ~20h on an RTX3080 per 100ns/complex. Only the **pocket** (protein atoms within 6Å of ligand) is used as model input, both because it's where the interesting motion happens and because it's much cheaper computationally than the whole protein. Train/val split of trajectories: 9:1.

**Downstream task 1 — Ligand Binding Affinity (LBA), regression.** PDBbind dataset via **ATOM3D**, split by 30% (or 60%) sequence identity to prevent homolog leakage. Predict `pK = -log(K)` (Ki or Kd). Metrics: **RMSE, Pearson correlation (Rp), Spearman correlation (Rs)**. Splits (30% identity): 3507 train / 466 val / 490 test.

**Downstream task 2 — Ligand Efficacy Prediction (LEP), binary classification.** ATOM3D dataset of protein structures in "active" vs "inactive" state bound to 527 small molecules (activators vs. non-activators), input = region within 5.5 Å of ligand. Metrics: **AUROC, AUPRC**. Splits by protein target: 608 train / 208 val / 208 test.

**Fine-tuning vs. linear-probing.** Fine-tuning updates the whole pretrained encoder; linear-probing freezes the encoder and only trains a small MLP predictor head on the pooled atom features. This distinction (and comparing both) is central to the paper's evidence that the *pretrained representations themselves* (not just extra fine-tuning capacity) carry useful signal.

**Generalization to *predicted* (not crystallographic) structures.** Real-world inference often lacks a crystal structure — you only have a *docked* pose. The paper re-runs LBA evaluation using **EquiBind**-docked structures instead of ground-truth cocrystal structures (`ProtMD/unknown_structure/dataloader_equidock.py`), after discarding 9 pairs where docking failed to produce a pocket (ligand >6Å from receptor). ProtMD fine-tuned still gets RMSE 1.474 (vs. 1.367 on true structures, vs. 1.668 no-pretrain-baseline on docked structures) — i.e., pretraining still helps substantially even when the downstream input structure is noisy/wrong, though performance is bounded by EquiBind's own docking error (avg. ligand RMSD 8.2 Å).

**Results summary (headline numbers from the paper):**
- LBA (30% split): ProtMD fine-tuned RMSE 1.367 vs. best prior baseline (3DCNN) 1.429 — **~4.3% RMSE reduction**, and highest Pearson/Spearman among all baselines (sequence-based: DeepDTA/LSTM/TAPE/ProtTrans; surface-based: MaSIF; multi-scale: HoloProt; structure-based: 3DCNN, IEConv). Even **linear-probing** ProtMD (RMSE 1.413, ~10K trainable params) beats *all* baselines that require full training.
- LEP: ProtMD fine-tuned AUROC 0.742 / AUPRC 0.724 vs. best baseline (DeepDTA) AUROC 0.696 / (3DGCN) AUPRC 0.598 — **~13.8% average AUROC/AUPRC improvement**.
- Pretraining sample-efficiency: even though pretraining used only ~64 protein-ligand pairs, it generalizes to >3K unseen PDBbind proteins; benefit for linear-probing saturates past ~50 pretraining proteins, but fine-tuning keeps improving (with diminishing returns) as pretraining set grows — "why does ProtMD work" analysis (§3.6 of the paper) shows the model's **predicted 3D coordinate shift magnitude** (Δx_LR, i.e. how much the pretrained model predicts the pocket will move) correlates strongly with the ground-truth binding affinity and efficacy — smaller predicted motion ⇒ tighter/more stable binding. This is the mechanistic explanation for *why* MD-trajectory self-supervision transfers to static-structure binding prediction.

**Codebase (`ProtMD/`)**:
- `egnn/egnn_pytorch.py`, `egnn/egnn_pytorch_geometric.py`, `egnn/utils.py` — the EGNN backbone (dense variant `EGNN`/`EGNN_Network`, sparse/PyG variant `EGNN_Sparse`/`EGNN_Sparse_Network`), plus `predictor` (pooling + MLP head) and `Classifier` (used for the snapshot-ordering auxiliary head).
- `data/dataset.py` (`PDBMDDataset`) — loads per-protein `.npy` MD trajectories (`R`=coords, `z`=atom types), builds (x_0, x_t) pairs for each prompt/Δt in `args.prompt`, adds Gaussian noise (σ≈0.1) during training only.
- `data/nc2xtc.ipynb`, `data/xtc2npy.ipynb` — trajectory format conversion notebooks (GROMACS/Amber `.xtc`/`.nc` → `.npy`).
- `pretrain.py` — the self-supervised pretraining loop (denoising + optional ordering task via `--ordering`), mixed-precision (`torch.cuda.amp`), multi-GPU `DataParallel`, `ReduceLROnPlateau` scheduler.
- `main.py` — downstream fine-tuning/linear-probing loop for `--data {lba,lep}`, loads pretrained checkpoint, swaps in a fresh `predictor` head, supports `--linear_probe` (freezes backbone) and `--unknown` (use EquiBind-docked test structures).
- `pdbbind/dataloader_pdb.py`, `pdbbind/statistics_pdb.py`, `lep/dataloader_lep.py`, `unknown_structure/dataloader_equidock.py` — dataset preparation (ATOM3D `LMDBDataset` → padded tensors saved as `.pt`), including the 60%-identity-split caveats (ATOM3D's published 60% split changed over time; the paper had to use HoloProt's older split via `identity60_split.json`).
- `utils/utils.py` (`parse_args`) — the full hyperparameter surface: `--ordering`, `--noise`, `--prompt` ("1,5,10" etc.), `--num_nearest` (32 nearest-neighbor graph), `--depth` (6 EGNN layers), `--dim` (128, doubled to 256 hidden per the paper text), `--dropout` 0.15, `--max_len` 10000 atoms, `--linear_probe`, `--pretrain <ckpt path>`, `--split {30,60}`.
- `utils/figure/*`, `utils/reproduce/*` — plotting scripts (ablation bar plots, t-SNE, y vs y-hat) and reproductions of baselines (3DCNN, SchNet, MXNet-based MXMNet) used for the comparison tables.
- **Caveat**: the raw MD trajectory data itself is **not public** (MindRankAI, the industry partner, would not release it) — the README instead documents in detail how to regenerate equivalent data via GROMACS/Amber + MDAnalysis/mdtraj/VMD. This is an important practical constraint if this project wants to reproduce ProtMD's *pretraining* stage from scratch (the *downstream* PDBbind/ATOM3D data is public).

---

## 4. Side-by-Side Comparison of the Two Lines

| | **DSR / MoE-DSR** | **ProtMD** |
|---|---|---|
| Representation | Continuous implicit surface (SDF over R³×time) | Discrete atom graph (pocket-level), explicit per-frame coordinates |
| What's learned | A generative model of the *shape trajectory itself* | A *representation/encoder* pretrained on trajectories, transferred to static tasks |
| Backbone | MLP (INR), optionally MoE + equivariant geometric router (MoE-DSR) | E(3)/E(n)-equivariant GNN (EGNN/EGMN), with optional global linear attention |
| Time handled how | Time is a literal continuous input dimension to the network | Time only appears during pretraining (as a Δt "prompt"); downstream tasks are single-snapshot |
| Training signal | Unsupervised SDF regression (Eikonal-regularized), no labels needed | Self-supervised denoising + ordering pretext tasks, then *supervised* fine-tuning on labeled downstream data |
| Main task solved | Reconstruct/interpolate/extrapolate a protein's dynamic *shape* | Predict a static *property* (affinity, efficacy) better, by having learned dynamics-aware features |
| Generalizes across proteins? | DSR: no (per-protein/per-batch latent code). MoE-DSR: yes, via MoE + held-out-family evaluation on ATLAS | Yes — pretrained on 64 pairs, evaluated on 3000+ unseen PDBbind pairs |
| Equivariance to rotation/translation? | DSR: no. MoE-DSR: yes (spherical harmonics irreps) | Yes, by construction (EGNN/E(3)-equivariant message passing) |
| Public training data available? | Yes (all datasets are public: MDAnalysisData, GPCRmd, ATLAS, etc.) | Downstream data yes (PDBbind/ATOM3D); **pretraining MD trajectories are proprietary/unreleased** |
| Metrics | IoU, Chamfer distance, Normal Consistency (3D shape similarity) | RMSE/Pearson/Spearman (regression); AUROC/AUPRC (classification) |
| Code maturity in this project | Full repo, runnable demo + pretrained checkpoint included | Full repo, but needs external MD data generation (no bundled trajectories) |

---

## 5. Cross-Cutting Challenges for This Project to Keep in Mind

1. **Data scarcity & cost.** Both lines are fundamentally bottlenecked by how much MD simulation data exists/can be generated. ProtMD needed ~20h/complex on an RTX3080 for just 100ns; MoE-DSR needed the (large, but still finite — 1390 proteins) ATLAS database to get real cross-family generalization.
2. **Noise vs. signal in raw MD trajectories.** DSR's own analysis shows model quality anti-correlates with trajectory RMSD/vibration — raw MD is noisy, and distinguishing "functional motion" from "thermal jitter" is an open problem that both papers touch (DSR observes its model implicitly denoises; ProtMD explicitly injects noise as a *regularizer* during pretraining for the opposite reason — to avoid trivial shortcuts).
3. **Equivariance is necessary but was originally missing.** Both DSR (fixed by MoE-DSR) and the base EGMN (already equivariant in ProtMD) illustrate that raw-coordinate models fail to generalize across arbitrary reference frames, and this is treated as a first-class design constraint in the more recent work.
4. **Generalization to unseen proteins/families** is the dominant open problem: DSR explicitly could not do this; MoE-DSR solves it via a MoE (but doesn't scale to many experts on today's data sizes); ProtMD solves it for a *narrower* prediction task (a scalar/binary property, not full dynamics) with encoder pretraining + transfer.
5. **Generalization to *predicted* (non-crystallographic) structures** — the ProtMD EquiBind experiment is the only place either line stress-tests robustness to structural inaccuracy, and shows real degradation tied to docking error. Any pipeline aiming at real-world drug-discovery deployment (vs. curated PDBbind benchmarks) needs to budget for this.
6. **Metrics don't transfer between the two lines** — shape-similarity metrics (IoU/Chamfer/NC) answer "did we reconstruct the right shape," while regression/classification metrics (RMSE/Pearson/AUROC) answer "did we predict the right downstream number." A project that wants to do *both* (e.g., predict dynamics *and* use that for a property prediction task, in the spirit of ProtMD but with an explicit dynamics model instead of an implicit pretrained encoder) will need both metric families and should think about whether an intermediate dynamics model (DSR-style) that then feeds a downstream predictor could out-perform ProtMD's implicit approach.

---

## 6. Open Questions / Possible Directions for This Project

(Framed as questions raised by gaps in the surveyed work — not a plan, just candidate angles.)

- **Combine the two lines?** Could a DSR/MoE-DSR-style continuous surface-dynamics model be used to *generate* synthetic trajectory frames or flexibility descriptors that feed into a ProtMD-style downstream predictor — reducing reliance on expensive real MD data for pretraining?
- **Reproduce MoE-DSR.** No public code exists for MoE-DSR yet; this project's `DSR/` codebase is a plausible base to extend with the router/MoE/equivariance ideas from the paper, validated against ATLAS (public, downloadable).
- **Data generation pipeline.** Since ProtMD's actual pretraining trajectories are proprietary, and DSR's `data_process.py` demonstrates a full PDB→surface-mesh pipeline (mdtraj + PyMOL), a reusable "MD trajectory → training data" pipeline (for either representation) is itself a useful project deliverable, potentially targeting public sources (ATLAS, MDAnalysisData, GPCRmd, or self-run Amber/GROMACS on PDBbind).
- **Noise-robustness / denoising as a first-class objective**, rather than a side effect (DSR) or a pretraining regularizer (ProtMD) — e.g., explicitly separating "functional/collective motion" from thermal noise (normal-mode / PCA-informed priors, or diffusion-model-style explicit denoising as cited in both papers' related work — DiffMD, coarse-grained diffusion force fields).
- **Long-horizon extrapolation** remains weak in DSR (quality drops sharply beyond ~10 frames) — modeling velocity/momentum fields (as DSR's own "future work" suggests) rather than raw position is untested in either repo.

---

## 7. Practical Repo Notes (for actually running this code)

- **DSR**: Python 3.8, PyTorch 1.12.0, plus `numpy pyhocon plotly scikit-image trimesh pymol mdtraj GPUtil wandb`. `pymol` and `mdtraj` are the two least-standard deps (used only in `data_process.py`/pre-processing, not in training itself). GPU required (`.cuda()` hardcoded throughout, `nn.DataParallel`). A working demo (`abeta_30`) and a pretrained checkpoint are bundled — good smoke-test before scaling to bigger data.
- **ProtMD**: Python 3.9.10, PyTorch 1.10.2+cu113, `einops`, plus `atom3d` (for LMDBDataset access to PDBbind/ATOM3D splits) and optionally `torch_geometric` (only for the unused-by-default sparse EGNN variant). Multi-GPU (`DataParallel`) + AMP (`torch.cuda.amp`) baked into both `pretrain.py` and `main.py`. **No bundled MD trajectory data** — `data/md/` must be populated externally (own Amber/GROMACS run, or a substitute public dataset) before `pretrain.py` will do anything; only a single pretrained checkpoint (`save/model.pt`) ships, so downstream fine-tuning/linear-probing (`main.py`) can be exercised immediately once ATOM3D's PDBbind/LEP splits are downloaded and preprocessed via `pdbbind/dataloader_pdb.py` / `lep/dataloader_lep.py`.
- Both repos are **research-grade, not production**: hardcoded paths/GPU ids, minimal error handling, some dead/commented-out code (e.g. `decoder_relu` import commented in `runner_igr_t.py`), Chinese-language inline comments in parts of ProtMD (`data/dataset.py`, `utils/utils.py`) — expect to need light refactoring before building on either as a foundation.

---

## 8. Critical Weaknesses & Concrete Improvement Opportunities

This goes beyond each paper's own stated limitations — it's a critical read of the method, the code, and the evaluation protocol.

### 8.1 DSR — weaknesses

1. **It's per-protein overfitting dressed up as modeling.** DSR trains a fresh network (or a small jointly-trained batch sharing a latent code) per protein/trajectory — architecturally identical to "fit a NeRF to one video." There is no mechanism for a trained model to say anything about a protein it wasn't trained on. The latent-code trick only interpolates *within* a training batch of already-seen proteins; it is not zero-shot generalization (MoE-DSR exists precisely because of this gap).
2. **Zero physical inductive bias.** The model has no notion of bonded/non-bonded interactions, mass, temperature, or energy. Nothing stops the network from producing surfaces that violate basic physical constraints (volume discontinuities, non-smooth velocity, bond-length-inconsistent motion). It is a pure geometric interpolator/extrapolator over shapes, not a physically constrained dynamical system.
3. **Evaluation only checks shape self-consistency, never reality.** IoU/Chamfer/NC all compare the model's output against *held-out frames of the same classical-MD trajectory* — never against any experimental flexibility signal (X-ray B-factors, NMR order parameters/S², cryo-EM ensemble heterogeneity). A model could hit high IoU while faithfully reproducing force-field artifacts rather than real protein behavior, and the current metrics cannot detect that.
4. **No topology handling.** Marching-cubes-based zero-level-set extraction assumes a fixed surface topology; large conformational changes (domain opening/closing, oligomerization/dissociation) that split or merge the surface are not discussed and likely break down under this formalism.
5. **Deterministic output for a stochastic process.** MD trajectories are inherently stochastic (thermal noise); DSR predicts one point estimate per (x,t) with no uncertainty/ensemble output, so there's no way to express "the surface could plausibly be in any of these configurations at time t."
6. **Implicit solvent / stripped water.** All the training surfaces are computed from trajectories with solvent removed or implicit. Real surface fluctuations are heavily water/ion-mediated — this is plausibly *part of* why DSR's own analysis found high-RMSD/"noisy" trajectories hard to fit: some of that "noise" may be physically meaningful solvent-coupled motion, not just nuisance jitter.
7. **Training cost doesn't amortize.** Scaling "one network per protein" to a database of hundreds/thousands of proteins means hundreds/thousands of independent training runs — this is the direct cause of DSR's inability to generalize, and MoE-DSR's whole reason for existing.

**Improvements to pursue:**
- Replace the free per-sequence latent code with a **conditioning embedding computed from the sequence/structure itself** (e.g., an ESM protein-language-model embedding, or a structure encoder) so a genuinely novel sequence gets a sensible starting latent without retraining — this is a lighter-weight alternative to MoE-DSR's router+experts machinery and could be tried first.
- Add a **physically-motivated regularizer**: penalize non-physical volume/area jumps between adjacent predicted timesteps, or tie the Eikonal/normal loss to a smoothness constraint on the *time* derivative (i.e. constrain ∂f/∂t, not just ∇_x f), which directly targets the weak long-horizon extrapolation the paper admits to.
- **Validate against experimental flexibility data** (PDB B-factors, or NMR-derived order parameters for at least a handful of well-studied proteins) as an external sanity check that the model is learning "real" flexibility, not classical-force-field-specific artifacts.
- Model an explicit **velocity/flow field** rather than raw position at each t (the paper's own stated future-work direction) — likely to help extrapolation quality degrade less sharply past 10 frames.
- Consider a **probabilistic/ensemble output** (e.g., a small diffusion or flow-based head on top of the latent code) to reflect that MD is a distribution over conformations, not a single deterministic path.

### 8.2 MoE-DSR — weaknesses

1. **No public code** — this is currently a paper-only artifact; nothing to run or verify independently yet.
2. **Router only sees the initial static structure P₀.** Two proteins with near-identical folds can have very different dynamics depending on sequence, ligand occupancy, oligomeric state, or environment (membrane vs. soluble) — none of which the router uses. It is fundamentally a structure-similarity clustering mechanism, not a dynamics-aware one.
3. **"Generalization" is tested at a fairly forgiving bar.** The train/test split uses ≤40% sequence identity — this rules out near-duplicates but is far short of testing generalization to a truly novel fold/family. The paper's own Table 3 (expert→protein assignment) shows experts cluster by sequence identity within their assigned bucket, consistent with "memorizing clusters of similar training proteins" rather than learning transferable dynamics.
4. **No parameter-matched ablation.** N_E experts of size M each vs. one monolithic MLP of size N_E×M is never compared — so it's unclear how much of the reported gain is "specialization" per se vs. simply more total capacity than vanilla DSR. (The finding that N_E≥16 *hurts* argues against pure capacity being the story, but the missing ablation leaves this unproven either way.)
5. **Still monomers only** (explicitly acknowledged) — doesn't cover protein-ligand or protein-protein complexes, so it still doesn't overlap with ProtMD's problem setting even after fixing DSR's generalization gap.
6. **Same evaluation blind spot as DSR** — IoU/Chamfer/NC only, no experimental cross-check.
7. **Requires an existing 3D structure to even start** (the router consumes a surface point cloud) — for a protein with no crystal/cryo-EM/AlphaFold structure, there's no fallback (unlike sequence-only pretraining approaches).

**Improvements to pursue:**
- **Reproduce it first** — there's no code, so before extending it this project would need to reimplement the router (Q-MPNN) + gated experts + equivariant SDF learner from the paper description in section 2.2 of this document.
- Enrich the router's conditioning signal with a **sequence embedding (ESM) and/or context features** (ligand-bound vs. apo, temperature, membrane vs. soluble) rather than structure geometry alone.
- Run the **parameter-matched single-expert-vs-MoE ablation** the paper is missing.
- Push the sequence-identity generalization bar down (test on genuinely different folds/CATH classes, not just <40% SI) to get an honest read on cross-family transfer.
- Extend to **protein-ligand complexes** — this is the natural bridge point between this line and ProtMD, and would let a single project's dynamics model feed a ProtMD-style downstream predictor.
- Replace hard top-K expert routing with a **continuous/soft mixture or hypernetwork** conditioned on a continuous embedding — avoids the "which discrete bucket does this novel protein fall into" brittleness that hard routing has when the training population is only ~1,300 proteins (small relative to typical MoE-LLM corpora, which the paper itself flags as the likely reason more experts stop helping).

### 8.3 ProtMD — weaknesses

1. **The actual pretraining data is proprietary and was never released** (MindRankAI declined). This is the single biggest reproducibility blocker in either repo — the README's Amber/GROMACS regeneration instructions require real MD expertise plus ~20 GPU-hours *per complex* (≈1,280 GPU-hours to reproduce just the 64-pair pretraining set from scratch).
2. **Pretraining set is tiny (64 pairs / 62.8K snapshots)**, and the paper never reports whether pretraining proteins overlap in family/fold with the PDBbind downstream test set — given PDBbind's known internal redundancy, some of the reported transfer could be partially attributable to family-level leakage rather than pure learned "flexibility" signal.
3. **Hard Markov (k=1) assumption in the generative task.** The prediction target is only ever conditioned on *one* previous frame plus a scalar time-gap prompt — there is no sequence/temporal encoder over a window of past frames, so genuinely non-Markovian conformational dynamics (slow gating motions, hysteresis) can't be captured by construction.
4. **Downstream tasks never see a trajectory at inference** — LBA/LEP fine-tuning and linear-probing both operate on a single static snapshot. All of the "dynamics awareness" is baked in at pretraining time and never operationally used at inference; the correlation between predicted coordinate-shift and binding affinity (§3.6 of the paper) is suggestive but only a post-hoc correlation, not something the downstream task actually conditions on.
5. **Baselines and the EquiBind docking front-end are now stale.** All comparison baselines (DeepDTA, TAPE, LSTM, ProtTrans, 3DCNN, IEConv, HoloProt, MaSIF) and the EquiBind docking model are 2019–2022-era. Docking accuracy has improved substantially since (EquiBind's own reported ligand RMSD of 8.2 Å is well above what modern docking (DiffDock-class, or AlphaFold3) achieves) — the "generalization to predicted structures" experiment would likely look different (probably better) with a modern docking front-end, and the main-task comparison would benefit from re-benchmarking against post-2022 protein representation learning (ESM-2/3, GVP-GNN, Uni-Mol2).
6. **Code robustness**: `pretrain.py`/`main.py` wrap their entire training loop in a bare `try/except:` that swallows *any* exception (`except: log.logger.exception(...)`) — real bugs and legitimate crashes are silently logged and training simply stops, rather than failing loudly. Worth fixing before relying on this as a base.
7. **Snapshot-ordering task is only tested at its weakest setting.** The paper explicitly notes that with n=2 conformations, the task degenerates to NSP-style ordering, which it also cites as a known weak pretext task — n>2 orderings (which the topological-sort formulation supports) are not separately ablated.

**Improvements to pursue:**
- **Substitute the proprietary pretraining trajectories with MISATO** (Zenodo/GitHub/HuggingFace, see §9 below) — ~20,000 protein-ligand complexes with explicit-solvent MD (>170 µs total), built directly from PDBbind. This is ~300× more complexes than ProtMD's original 64-pair set, fully public, and a drop-in replacement for the pretraining stage.
- **Explicitly audit pretrain/downstream family overlap** before reporting any transfer numbers, to rule out leakage as an explanation for the reported gains.
- Replace the k=1 Markov denoising objective with a **short temporal-window encoder** (e.g. a small transformer/RNN over the last few frames) so the pretext task can capture non-Markovian dynamics, not just adjacent-frame deltas.
- **Re-run the EquiBind generalization experiment with a modern docking model** (DiffDock, Uni-Mol docking, or AlphaFold3-predicted complexes) to get a currently-relevant read on how well pretrained dynamics-aware representations transfer to realistic (non-crystallographic) input structures.
- **Re-benchmark against 2023–2025 baselines** (ESM-2/ESM-3 embeddings, GVP-GNN, Uni-Mol2, equivariant transformers) for a meaningful current-day comparison.
- Fix the bare-except training loops and add real exception handling/logging if building on this code.
- Feed a short trajectory (not a single frame) at inference time where feasible (e.g. a cheap few-ns local MD relaxation of a docked pose) so the fine-tuned model is actually used the way its pretraining objective was designed for.

### 8.4 Cross-cutting weaknesses (all three works)

1. **No comparison to physics-based free-energy methods** (FEP/TI, alchemical binding free energy calculations) — all evaluation is DL-vs-DL; none of the three papers benchmarks against the actual gold-standard physical calculations that ML methods are implicitly trying to approximate or accelerate.
2. **No uncertainty quantification anywhere.** All three produce point predictions (a single surface, a single affinity number, a single class probability) with no calibrated confidence — a serious gap for anything meant to support real decisions (e.g., drug discovery triage).
3. **No energy/force conservation constraints.** Unlike the ML-force-field literature they all cite (SchNet, DTNN, and more recent equivariant force fields like NequIP/MACE — not discussed), none of these three methods enforces energy conservation, time-reversal symmetry, or other physical invariants beyond spatial equivariance (and DSR doesn't even have that).
4. **All three treat classical-force-field MD as ground truth.** They train DL models to imitate MD simulations, which are themselves approximations (classical force fields, e.g. AMBER/CHARMM/GROMOS have known, well-documented biases). Any systematic force-field bias is inherited — and potentially amplified — by the learned models, with no mechanism in any of the three papers to detect or correct for this against experiment.
5. **Solvent is stripped or implicit everywhere.** DSR's demo data and most MDAnalysisData trajectories used are solvent-free or implicit-solvent; ProtMD's Amber protocol does include explicit solvent during simulation but the model itself never sees water/ion atoms (pocket-only, protein/ligand atoms only). Explicit water and ion effects, known to matter for binding thermodynamics and surface fluctuation, are consistently discarded for computational tractability.

---

## 9. Dataset Retrieval Guide (verified against current sources)

Access details for MD/structure databases can drift (paywalls appear, hosts move, preprocessed mirrors get taken down for licensing reasons — several of those changes have in fact happened since these papers were published, noted below). This table reflects what's currently accessible.

| Dataset | Used by | How to get it | Access notes / gotchas |
|---|---|---|---|
| **Aβ(1-40) 500ns trajectories** (Grazioli & Butts) | DSR | Harvard Dataverse: [doi:10.7910/DVN/ERYOZS](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ERYOZS) | Open, direct download, no registration. |
| **MDAnalysisData** (AdK equilibrium, AdK DIMS/FRODA transitions, I-FABP, NhaA, YiiP) | DSR | `pip install MDAnalysisData` (or `conda install -c conda-forge mdanalysisdata`), then e.g. `from MDAnalysisData import datasets; datasets.fetch_adk_equilibrium()` / `fetch_nhaa_equilibrium()` / `fetch_yiip_equilibrium_long()` / `fetch_ifabp_water()` | Package is a thin fetch/cache layer over external hosts (figshare/Zenodo/Dryad) — no manual URL-hunting needed, just call the `fetch_*` function for the dataset you want. Fully open. |
| **GPCRmd** (4 GPCR trajectories) | DSR | Browse [gpcrmd.org](https://www.gpcrmd.org/) via the "GPCRome"/family-tree interface or `/dynadb/datasets/` listing, open the specific PDB/dynamics ID, download per-simulation | No confirmed bulk/API download was found in this pass — access looks per-simulation via the web UI (likely needs a free account for downloads); check `gpcrmd-docs.readthedocs.io` or email `info@gpcrmd.org` if you need the same 4 trajectory IDs the DSR paper used (`10792_trj_81`, `10840_trj_87`, `10912_trj_95`, `15711_trj_791`). |
| **DRY_AD_MD** (23-protein subset of Jumper et al.'s 450-protein set) | DSR | Dryad: [doi:10.5061/dryad.h9f8sb7](https://datadryad.org/dataset/doi:10.5061/dryad.h9f8sb7) | Open, direct download. DSR only used 23 of the ~450 proteins there (screened for mean RMSD < 1.0) — you'll need to re-derive which 23 or re-do the RMSD screen yourself, since the paper doesn't list the exact PDB IDs. |
| **ATLAS** (1,390-protein all-atom MD database) | MoE-DSR | [dsimb.inserm.fr/ATLAS](https://www.dsimb.inserm.fr/ATLAS) — bulk `download_ATLAS.py` script (uses `aria2c`) or the documented REST API for targeted protein/trajectory pulls | Free, CC-BY-NC 4.0 (non-commercial only; contact the authors for commercial use). This is by far the most "just works" dataset of the ones surveyed, and is the natural one to start with for any generalization-across-proteins experiment. |
| **PDBbind** (protein-ligand structures + binding affinities) | ProtMD | Register at [pdbbind.org.cn](http://www.pdbbind.org.cn/enroll.asp) | ⚠️ **Changed since the paper**: the ≤2020 release is still free for academic use at the original `pdbbind.org.cn`; PDBbind has since been rebranded/relaunched as **PDBbind+** (`pdbbind-plus.org.cn`), and versions **2021 and later now require paid access**. If you only need what ProtMD used, the free 2020-or-earlier version is sufficient — don't assume the newer releases are free. |
| **ATOM3D LBA split** (PDBbind repackaged, 30%/60% identity splits) | ProtMD | `pip install atom3d`, then `atom3d.datasets.download_dataset('lba', target_path, split='sequence-identity-30')` — or manually from Zenodo [doi:10.5281/zenodo.4914718](https://zenodo.org/record/4914718) | Free, no registration. This is the actual dataset `pdbbind/dataloader_pdb.py` expects at `split-by-sequence-identity-*/data/`. |
| **ATOM3D LEP split** (active/inactive ligand efficacy) | ProtMD | Same `atom3d` package: `download_dataset('lep', target_path)` — or Zenodo [doi:10.5281/zenodo.4914734](https://zenodo.org/record/4914734) | Free, no registration. Matches what `lep/dataloader_lep.py` expects. |
| **Legacy HoloProt 60%-identity split** (`identity60_split.json`) | ProtMD | [github.com/vsomnath/holoprot](https://github.com/vsomnath/holoprot) | Needed only if reproducing ProtMD's 60%-split numbers exactly; per the ProtMD README this old split has some corrupted/missing PDB files baked in (acknowledged by HoloProt's own author) — treat any 60%-split reproduction as inherently slightly lossy. |
| **EquiBind-docked test structures** (generalization-to-predicted-structures experiment) | ProtMD | Originally Zenodo record 6034088 | ⚠️ **No longer available** — this preprocessed mirror was taken down because PDBbind's license doesn't permit redistribution. To reproduce this experiment now: download PDBbind yourself (see above) and run [EquiBind](https://github.com/HannesStark/EquiBind) locally to regenerate the docked poses, or substitute a newer docking model (see improvement suggestions above). |
| **Raw ProtMD pretraining trajectories** (64 protein-ligand pairs, 62.8K snapshots) | ProtMD | **Not available** — proprietary to MindRankAI, never released | This is the one dataset in this whole survey with no retrieval path at all. **Recommended substitute: MISATO** (below) — public, ~300× larger, built the same way (Amber-family MD on PDBbind complexes). |
| **MISATO** (public protein-ligand MD dataset — *not used by either paper, but the best available substitute for ProtMD's proprietary pretraining data*) | *(new)* | [github.com/t7morgen/misato-dataset](https://github.com/t7morgen/misato-dataset), also on [Zenodo](https://zenodo.org/record/7711953) and Hugging Face; paper: [Nature Computational Science 2024](https://www.nature.com/articles/s43588-024-00627-2) | ~20,000 experimental protein-ligand complexes (from PDBbind), QM-refined, with explicit-solvent MD trajectories totaling >170 µs. Ships with Python data loaders and ML baselines already. This is the strongest lead for reproducing/rebuilding a ProtMD-style pretraining pipeline entirely on public data. |

**Bottom line on data access**: the surface-dynamics line (DSR/MoE-DSR) is essentially fully reproducible from public data today, with ATLAS as the star resource for anything needing cross-protein generalization. The pretraining line (ProtMD) has one real hole — its original pretraining trajectories are gone for good — but **MISATO closes that hole with a public dataset that's two orders of magnitude larger**, making it arguably a better foundation than what ProtMD itself used, if this project wants to build in that direction.
