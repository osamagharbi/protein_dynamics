# DSR and Diffusion-DSR: Conceptual and Architectural Report

**Project:** Protein Dynamics  
**Date:** 2026-10-14  
**Purpose:** Consolidate the discussion of DSR, Diffusion-DSR, Stage A, Stage B, inference, and cross-protein generalization.

## 1. Project context

Proteins are dynamic objects. Their functions—binding, catalysis, transport, and signaling—depend on changes in three-dimensional structure over time. Molecular dynamics (MD) simulation describes this motion, but it is expensive because conventional simulations require very small integration steps.

This project studies neural models that learn from MD trajectories. The main representation is the molecular surface, modeled as a continuous signed-distance function (SDF). Given a spatial location, the model predicts the signed distance to the protein surface. The surface is extracted as the zero level-set of that SDF.

The project contains two related systems:

1. **DSR:** a deterministic implicit neural representation of a protein surface over time.
2. **Diffusion-DSR:** a generative extension that models latent uncertainty and can produce multiple plausible surfaces for the same time.

The current implementation has been tested primarily on one Aβ trajectory. It should therefore be understood as a single-protein temporal modeling system, not yet as a cross-protein model.

## 2. DSR: the basic model

DSR represents a protein’s molecular surface using a neural signed-distance function. The decoder receives:

    (x, y, z, t, z_protein) → SDF value

where:

- (x, y, z) is a query point in three-dimensional space;
- t is the normalized trajectory time;
- z_protein is a learned protein-specific latent code;
- the output is the signed distance from the query point to the molecular surface.

The current DSR implementation uses ImplicitNet, an MLP with Softplus activations, geometric initialization, and a skip connection. Its objective includes surface, normal, Eikonal, and latent regularization terms.

### 2.1 Spatial coordinates

The spatial coordinates are not the original atom coordinates. During inference, the decoder is queried over a regular three-dimensional grid covering a normalized bounding box, approximately:

    (x, y, z) ∈ [-1, 1]^3

At resolution 128, this is approximately 128 × 128 × 128 = 2,097,152 query points. The predicted SDF values are passed to marching cubes, which extracts the zero-level surface as a mesh. The normalized mesh is then transformed back to physical coordinates.

### 2.2 Purpose of the protein latent

The protein latent acts as global context. The shared decoder learns how to represent surfaces generally, while the latent tells it which protein or trajectory it is modeling. It can encode overall shape, surface details, and trajectory-specific characteristics.

For a single protein, the latent is technically not necessary. A model could instead learn:

    (x, y, z, t) → SDF

However, the learned latent provides extra adjustable parameters and acts like a global conditioning vector. It becomes more meaningful when one decoder is trained on several proteins, because each protein receives a different code.

The latent is not a biological sequence embedding, a physical state variable, a guaranteed transferable representation, or an uncertainty variable. In the original DSR setup, a new protein normally requires learning a new protein-specific latent and often training or adapting the model for that protein. This limits cross-protein generalization.

## 3. DSR training and inference

### 3.1 Data preparation

An MD trajectory is converted into one molecular-surface mesh per frame:

    MD trajectory
        ↓
    surface extraction
        ↓
    frame_000.obj, frame_001.obj, ..., frame_N.obj

Surface points and normals are sampled from these meshes. Coordinates are normalized into the decoder’s coordinate system.

### 3.2 Training

The decoder is trained to satisfy the geometric properties of an SDF:

- surface points should have SDF approximately zero;
- SDF gradients on the surface should agree with surface normals;
- the gradient norm away from the surface should be approximately one;
- latent codes should remain regularized.

### 3.3 Inference

For a known protein, the learned protein latent is passed together with a query time and every spatial grid point:

    (x, y, z, t, z_protein) → SDF → marching cubes → surface mesh

DSR is deterministic: one query time produces one surface prediction. It does not explicitly represent multiple plausible conformations or uncertainty.

## 4. Diffusion-DSR: the generative extension

Diffusion-DSR keeps the DSR-style implicit surface decoder but changes how the latent code is obtained. Instead of using only a fixed protein-level latent, the current implementation learns a latent vector for each observed frame and then models those latent vectors over time.

The current pipeline is:

    Stage A: learn decoder + per-frame latent vectors
        ↓
    Stage B: learn a time-conditioned latent distribution
        ↓
    Stage C: sample latents, decode surfaces, and evaluate them

The method is sequential rather than end to end in the current code. Stage A is trained first and saved. Stage B is then trained using Stage A’s saved latent vectors. Stage C performs inference and evaluation.

## 5. Stage A

### 5.1 Stage A is an auto-decoder, not an autoencoder

A conventional autoencoder contains an encoder and decoder:

    frame → encoder → latent → decoder → reconstructed frame

Stage A does not contain an encoder. It is an auto-decoder:

    frame index i → learned latent z_i
    (x, y, z, t_i, z_i) → SDF decoder → reconstructed surface

The per-frame latent vectors are directly stored as learnable parameters.

### 5.2 Stage A data preparation

The trajectory is converted into surface meshes. For each frame, the data loader provides:

- surface points;
- surface normals;
- normalized time t_i;
- frame index i;
- local scales used for off-surface sampling.

The model is trained on surface geometry rather than directly on raw atomic coordinates.

### 5.3 Stage A latent table

For a trajectory with N frames, Stage A creates a trainable table:

    z_0, z_1, ..., z_(N−1)

For example, with 149 frames and latent size 8, the table has shape 149 × 8.

The vectors are randomly initialized and learned from scratch. They are not obtained from a pretrained protein model.

### 5.4 Stage A training task

Stage A is trained on per-frame molecular-surface reconstruction. For a selected frame i, the model retrieves z_i and evaluates:

    (x, y, z, t_i, z_i) → predicted SDF

The loss contains:

1. **Surface/manifold loss:** the SDF should be zero on surface points.
2. **Normal loss:** predicted SDF gradients should match surface normals.
3. **Eikonal loss:** SDF gradients away from the surface should have norm close to one.
4. **Latent regularization:** prevents excessively large latent vectors.
5. **Temporal smoothness loss:** encourages adjacent frame latents to be close.
6. **Optional consistency loss:** makes the decoder less sensitive to small latent perturbations.

The temporal smoothness term is:

    ||z_(i+1) − z_i||²

It encourages the learned latent vectors to form a smooth trajectory rather than a collection of unrelated codes.

### 5.5 What Stage A updates

The following are optimized jointly:

    shared decoder parameters
    per-frame latent vectors

The decoder learns how to represent protein surfaces. Each z_i specializes the decoder to the geometry of frame i.

### 5.6 Freezing Stage A

At the end of Stage A, the following are saved:

- decoder weights;
- per-frame latent table;
- frame and time metadata;
- training configuration;
- checkpoint and loss logs.

In the current implementation, the decoder and latent table are then treated as fixed. Stage B does not backpropagate through the decoder to update Stage A.

## 6. Stage B

Stage B receives the latent trajectory learned by Stage A:

    z_0, z_1, ..., z_(N−1)

It learns to produce a latent at a requested time. The current implementation uses a hybrid design:

    mean predictor: μ(t)
    diffusion model: residual around μ(t)

The mean predictor estimates the central latent:

    t → μ(t)

The diffusion model learns the residual distribution:

    residual = z_i − μ(t_i)

At inference, a latent is generated as:

    z(t) = μ(t) + sampled_residual(t)

The main Stage B components are:

- MeanPredictor, a time-conditioned MLP;
- LatentDenoiser, a residual diffusion denoiser conditioned on diffusion time and physical trajectory time;
- GaussianDiffusion, which implements the diffusion and DDIM sampling process.

## 7. Stage C inference

For a requested time t, the current inference process is:

1. Query t.
2. MeanPredictor produces μ(t).
3. The diffusion model samples one or more residuals.
4. Combine them into latent vectors z(t).
5. Build a three-dimensional spatial grid.
6. Evaluate the frozen Stage A decoder:
   
       (x, y, z, t, z(t)) → SDF

7. Extract the zero-level surface with marching cubes.
8. Evaluate IoU, Chamfer distance, and normal consistency if a target frame exists.

For an ensemble, Stage B samples:

    z_1(t), z_2(t), ..., z_K(t)

Each latent produces a separate surface. Variation between surfaces provides a possible uncertainty signal. However, ensemble diversity alone is not calibration; calibration requires showing that larger ensemble spread predicts larger reconstruction error.

## 8. The oracle diagnostic

The oracle is an upper-bound diagnostic, not a deployable predictor.

Stage A learns a latent for each real frame. The oracle gives the decoder the actual learned latent for the target frame instead of asking Stage B to predict it:

    true learned z_i + decoder → reconstruction of frame i

It answers:

> How well could the decoder perform if the correct latent were known?

If the oracle outperforms DSR, the decoder representation has a higher reconstruction ceiling. The remaining gap is due to latent prediction, not necessarily surface representation.

The oracle must not be reported as normal inference because it uses information learned from the target frame and therefore leaks target information.

## 9. Current limitation: known protein versus unseen time

The current Diffusion-DSR implementation is protein-specific. It can predict new times or held-out frames within a protein trajectory used to train Stage A:

    known protein + new time → predicted latent → predicted surface

It cannot directly handle a completely unseen protein:

    new protein + time → no learned decoder or latent trajectory → no valid prediction

The reason is that Stage B currently learns only:

    t → latent for the one known protein

The protein identity is implicit in the loaded Stage A decoder, latent table, and Stage B checkpoint. The requested time alone does not tell the model which protein is being queried.

The code contains optional dataset-ID conditioning in LatentDenoiser, but it is configured for one dataset and is not a solution to true unseen-protein generalization.

## 10. Why a protein-ID embedding is not enough

If a model learns one ID embedding per training protein, it has a table such as:

    protein_A → e_A
    protein_B → e_B
    protein_C → e_C

For a new protein, there is no learned vector. Creating a random vector does not solve the problem.

One could optimize a new vector using observed frames from the new protein:

    new protein surfaces → optimize e_new

That is few-shot adaptation or per-protein fine-tuning, not zero-shot cross-protein generalization. It requires target data from the new protein.

True generalization requires producing the protein embedding from information available for the new protein without observing its future trajectory:

    new sequence or structure → protein encoder → new protein embedding

## 11. Proposed cross-protein architecture

The proposed architecture separates static protein identity from time-dependent dynamics:

    protein structure or sequence → protein encoder → c_p
                                                   ↓
    time t → dynamic latent model → h_p(t)
                                                   ↓
    (x, y, z, t, c_p, h_p(t)) → SDF decoder

Here:

- c_p is the static protein embedding;
- h_p(t) is the dynamic frame latent;
- the decoder uses both to reconstruct the molecular surface.

### 11.1 Protein encoder

The encoder should read information available for an unseen protein, such as:

- amino-acid sequence;
- initial three-dimensional structure;
- molecular surface point cloud;
- atom or residue graph;
- context such as ligand-bound state, membrane environment, or temperature.

For structure-based input, an E(3)-equivariant graph or point-cloud encoder is preferable. The protein-level output should be invariant to arbitrary global rotation and translation of the input structure.

### 11.2 Stage A with protein conditioning

For training proteins, the redesigned Stage A would compute:

    structure or sequence of protein p → encoder → c_p

It would also maintain auxiliary per-frame dynamic latents:

    h_(p,0), h_(p,1), ..., h_(p,N_p−1)

The decoder would be trained on:

    (x, y, z, t_i, c_p, h_(p,i)) → SDF

The per-frame latents are training variables that allow the decoder to fit observed surfaces. They are not the mechanism for zero-shot inference.

### 11.3 Stage B with protein conditioning

Stage B must learn the dynamic latent distribution conditioned on both protein and time:

    (c_p, t) → μ_p(t)

and:

    (c_p, t, diffusion step) → residual distribution

This allows one Stage B model to produce different latent trajectories for different proteins.

### 11.4 Inference on a new protein

For an unseen protein:

    new protein structure or sequence
            ↓
    same frozen protein encoder
            ↓
    c_new
            ↓
    (c_new, requested time t) → Stage B → h_new(t)
            ↓
    (x, y, z, t, c_new, h_new(t)) → frozen decoder → surface

No target future frame is needed. The encoder-generated protein embedding supplies protein identity, and Stage B generates the time-dependent latent.

## 12. Relationship to MoE-DSR

MoE-DSR is designed primarily for cross-protein generalization. It uses structure-based routing and multiple experts, whereas the current Diffusion-DSR implementation is designed primarily for temporal generative modeling of a known protein trajectory.

Their intended strengths differ:

| Capability | Current Diffusion-DSR | MoE-DSR concept |
|---|---|---|
| New times within known protein | Yes | Yes |
| Unseen proteins | No | Intended capability |
| Probabilistic latent sampling | Yes | Not the main focus |
| Ensemble output | Yes | Not the main focus |
| Public implementation in this project | Yes | No verified implementation |
| Current experimental status | Aβ artifacts available | Not independently reproduced here |

A future model could combine the ideas:

    structure-conditioned encoder or MoE router
        +
    conditional latent diffusion
        +
    implicit SDF decoder

The encoder or router would provide cross-protein transfer. The diffusion head would provide multiple plausible dynamic latents and uncertainty.

## 13. Current evidence and limitations

Existing Aβ artifacts support the following conclusions:

- Stage A per-frame latent learning works as a surface-fitting procedure.
- Temporal smoothness substantially improves the learnability of the latent trajectory.
- Diffusion-DSR v2 improves significantly over the unstructured v1 latent baseline.
- The method can outperform the recorded DSR baseline at locally smooth time points.
- The oracle can exceed DSR, indicating a higher representational ceiling.
- Ensemble samples are non-identical, showing diversity.

The following are not yet established:

- universal improvement over DSR;
- true uncertainty calibration;
- cross-protein generalization;
- performance on AdK DIMS under a fully matched protocol;
- performance across ten or more proteins;
- experimental or physical validity beyond agreement with classical-MD surface frames.

There is also a protocol issue: the current Stage A run sees all frames, while Stage B holds out every fifth frame. This means the current comparison is not a fully leakage-free end-to-end evaluation. A fair experiment must apply the same train/holdout split to Stage A, Stage B, and the DSR baseline.

## 14. Recommended development plan

### Phase 1: fair single-protein baseline

1. Freeze the every-fifth-frame train/holdout split.
2. Train Stage A only on training frames.
3. Train Stage B using only those Stage A latents.
4. Retrain DSR on exactly the same frames.
5. Use at least three random seeds.
6. Save per-frame metrics and configurations.

### Phase 2: test the scientific hypothesis

1. Run AdK DIMS as the first smooth-transition case.
2. Compare Diffusion-DSR and DSR using IoU, Chamfer, and normal consistency.
3. Compute a trajectory smoothness score.
4. Measure performance gain relative to DSR.
5. Test whether the gain depends on smoothness.

### Phase 3: validate uncertainty

1. Generate ensembles at held-out times.
2. Measure ensemble spread.
3. Compare spread with actual reconstruction error.
4. Report calibration curves or correlations.

### Phase 4: extend to multiple proteins

1. Begin with multi-protein training using protein-ID conditioning as a debugging baseline.
2. Replace protein IDs with a sequence- or structure-based encoder.
3. Train a shared decoder and conditional Stage B across proteins.
4. Hold out entire proteins from training.
5. Evaluate on sequence-identity and fold-level splits.

### Phase 5: add equivariance and richer context

Add E(3)-equivariant geometric processing and, where available, sequence and environmental information. This should be done after the simpler conditional baseline is verified.

## 15. Final conceptual summary

The current Diffusion-DSR model is a sequential, protein-specific generative surface model:

    Stage A:
    surface frames → decoder + learned per-frame latents

    Stage B:
    time → distribution over those learned latents

    Stage C:
    sampled latent + time + spatial grid → surface mesh

The model can predict multiple plausible surfaces for new times within a known protein trajectory, but it does not yet know how to identify or model an unseen protein.

To achieve true cross-protein generalization, the architecture must add a protein encoder that produces a protein embedding from sequence or structure, then condition both the decoder and the latent diffusion model on that embedding:

    new protein structure or sequence → c_new
    (c_new, t) → dynamic latent
    (c_new, dynamic latent, spatial coordinates, t) → surface

The key distinction is between:

- **static protein embedding:** generated by an encoder and usable for unseen proteins;
- **dynamic per-frame latent:** learned during Stage A for training frames and generated by Stage B at inference.

Keeping these roles separate is the foundation for extending Diffusion-DSR beyond a single known protein trajectory.

