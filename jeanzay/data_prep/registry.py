"""Single source of truth for every protein-surface-dynamics dataset.

Two tables:

* ``SOURCES``  -- where raw trajectories come from and how to fetch them.
* ``datasets`` -- what to convert: topology/trajectory (relative to RAW_ROOT),
  atom selection, which frames, and the DSR-paper training rule.

Output frames are numbered 0..N-1 *after* cropping/striding, so the DSR loader
(which derives time from the integer in ``traj_<i>.obj``) always sees a uniform
time axis. The original trajectory index of every output frame is recorded in
``meta.json`` by ``prep.py finalize``.

Paper rules follow the DSR paper (NeurIPS 2023) as already encoded in
``MAMBA-DSR/paper_protocol.py`` and ``MAMBA-DSR/comparison/paper_reference``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# --------------------------------------------------------------------------
# Raw sources. Layout under RAW_ROOT mirrors the local datasets/DSR/ tree.
# --------------------------------------------------------------------------
SOURCES = {
    # MDAnalysisData fetchers verify SHA-256 themselves; data_home=RAW_ROOT/MDAnalysisData
    "mdad_adk_equilibrium": {"kind": "mdanalysisdata", "fetch": "fetch_adk_equilibrium"},
    "mdad_adk_dims":        {"kind": "mdanalysisdata", "fetch": "fetch_adk_transitions_DIMS"},
    "mdad_adk_froda":       {"kind": "mdanalysisdata", "fetch": "fetch_adk_transitions_FRODA"},
    "mdad_ifabp":           {"kind": "mdanalysisdata", "fetch": "fetch_ifabp_water"},
    "mdad_nhaa":            {"kind": "mdanalysisdata", "fetch": "fetch_nhaa_equilibrium"},
    "mdad_yiip":            {"kind": "mdanalysisdata", "fetch": "fetch_yiip_equilibrium_long"},
    # Harvard Dataverse, per-file md5 from the dataset API (verified 2026-11-10)
    "abeta": {"kind": "dataverse", "doi": "doi:10.7910/DVN/ERYOZS",
              "server": "https://dataverse.harvard.edu", "subdir": "Abeta_500ns"},
    # Dryad blocks anonymous scripted downloads (401/403). Either export
    # DRYAD_TOKEN, or rsync the archive into RAW_ROOT/DRY_AD_MD/ beforehand.
    "dryad": {"kind": "dryad", "file_id": 15918, "subdir": "DRY_AD_MD",
              "archive": "trajectories.tar.gz", "md5": "1ac0ab33d974b7bf9664c88ad377339f",
              "size": 2828821333},
    # No public bulk API; stage RAW_ROOT/GPCRmd/<id>.xtc + topology by hand.
    "gpcrmd": {"kind": "manual", "subdir": "GPCRmd"},
    # ATLAS (MoE-DSR). ~0.9 GB/protein zip, 1,938 entries -> opt-in only.
    "atlas": {"kind": "atlas",
              "list_url": "https://www.dsimb.inserm.fr/ATLAS/data/download/distributions/2024_11_18_ATLAS_pdb.txt",
              "zip_url": "https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{id}/{id}_protein.zip",
              "subdir": "ATLAS"},
}


@dataclass(frozen=True)
class Dataset:
    name: str
    source: str
    topology: tuple            # candidates relative to RAW_ROOT; first existing wins
    trajectory: str            # relative to RAW_ROOT
    n_frames: Optional[int]    # output frames to convert; None = all available
    stride: int = 1            # original-trajectory step between output frames
    start: int = 0             # first original frame
    selection: str = "protein" # MDAnalysis selection: lipids/water/ions are excluded
    train_stop: Optional[int] = None   # DSR-paper training rule, OUTPUT indices
    train_step: Optional[int] = None
    groups: tuple = ("paper",)
    note: str = ""


MD = "MDAnalysisData"
DRYAD_PAPER = ["T0765", "T0769", "T0771", "T0773", "T0816", "T0855", "bba", "bbl",
               "cspa", "gpW", "hyp", "ntl9", "protG", "ww"]           # DSR Table 1 (14)
DRYAD_EXCLUDED = ["T0803", "a3d", "homeo", "lambda", "nug2", "protB", "protL",
                  "top7", "ubq"]                                       # screened out by DSR
GPCRMD_IDS = ["10792_trj_81", "10840_trj_87", "10912_trj_95", "15711_trj_791"]


def _static() -> dict:
    d = [
        Dataset("abeta_wt", "abeta", ("Abeta_500ns/abeta_1_40_mono.psf",),
                "Abeta_500ns/abeta_1_40_mono_phystempLong.dcd", 2000,
                train_stop=2000, train_step=1,
                note="paper: first 2000 frames; no thinning interval stated"),
        Dataset("abeta_e22g", "abeta", ("Abeta_500ns/abeta_E22G_1_40_mono.psf",),
                "Abeta_500ns/abeta_E22G_1_40_mono_phystempLong.dcd", 2000,
                train_stop=2000, train_step=1,
                note="listed in paper datasets; no separate metric reported"),
        Dataset("adk_equilibrium", "mdad_adk_equilibrium", (f"{MD}/adk_equilibrium/adk4AKE.psf",),
                f"{MD}/adk_equilibrium/1ake_007-nowater-core-dt240ps.dcd", 1500,
                train_stop=1500, train_step=10, note="paper Table 1: first 1500, every 10th"),
        Dataset("adk_dims", "mdad_adk_dims", (f"{MD}/adk_transitions_DIMS/DIMS/topologies/adk4ake.psf",),
                f"{MD}/adk_transitions_DIMS/DIMS/trajectories/dims0001_fit-core.dcd", 102,
                train_stop=80, train_step=2,
                note="102-frame trajectory (matches paper); Table 1 uses every 2nd of first 80"),
        Dataset("adk_froda", "mdad_adk_froda", (f"{MD}/adk_transitions_FRODA/FRODA/topologies/1ake.pdb",),
                f"{MD}/adk_transitions_FRODA/FRODA/trajectories/pathway1_fit-core.dcd", 142,
                train_stop=120, train_step=3,
                note="142-frame trajectory (matches paper); Table 1 uses every 3rd of first 120"),
        Dataset("ifabp_water", "mdad_ifabp", (f"{MD}/ifabp_water/ifabp_water.psf",),
                f"{MD}/ifabp_water/rmsfit_ifabp_water_1.dcd", 500,
                train_stop=400, train_step=5,
                note="explicit water in source; protein-only surface. Table 1: every 5th of first 400"),
        Dataset("nhaa_equilibrium", "mdad_nhaa", (f"{MD}/nhaa_equilibrium/NhaA_non_water.gro",),
                f"{MD}/nhaa_equilibrium/NhaA_non_water.xtc", 1500,
                train_stop=1500, train_step=10,
                note="POPE/POPG membrane in source; protein-only surface"),
        Dataset("yiip_equilibrium", "mdad_yiip", (f"{MD}/yiip_equilibrium/YiiP_system.pdb",),
                f"{MD}/yiip_equilibrium/YiiP_system_90ns_center.xtc", 1500,
                train_stop=1500, train_step=10,
                note="membrane + water in source; protein-only surface"),
    ]
    for p in DRYAD_PAPER + DRYAD_EXCLUDED:
        d.append(Dataset(
            f"dryad_{p}", "dryad", (f"DRY_AD_MD/trajectories/{p}/{p}.pdb",),
            f"DRY_AD_MD/trajectories/{p}/{p}_native.xtc", 1500,
            train_stop=1500, train_step=10,
            groups=("paper", "dryad") if p in DRYAD_PAPER else ("dryad_excluded",),
            note="Upside coarse-grained (backbone+CB+O+NH); native-start replica"))
    for g in GPCRMD_IDS:
        d.append(Dataset(
            f"gpcr_{g}", "gpcrmd",
            tuple(f"GPCRmd/{g}.{ext}" for ext in ("pdb", "psf", "gro")),
            f"GPCRmd/{g}.xtc", None, groups=("gpcrmd",),
            note="topology must be staged manually from gpcrmd.org; paper frame rule unverified"))
    return {x.name: x for x in d}


STATIC = _static()
GROUPS = sorted({g for ds in STATIC.values() for g in ds.groups} | {"atlas", "all"})


def atlas_dataset(pdb_chain: str, replica: int = 1, stride: int = 100,
                  n_frames: Optional[int] = 101) -> Dataset:
    """ATLAS entry. Default: replica R1, every 100th of the 10,001 frames ->
    101 surfaces (MoE-DSR used 100 snapshots/protein)."""
    return Dataset(
        f"atlas_{pdb_chain}", "atlas", (f"ATLAS/{pdb_chain}/{pdb_chain}.pdb",),
        f"ATLAS/{pdb_chain}/{pdb_chain}_prod_R{replica}_fit.xtc", n_frames,
        stride=stride, groups=("atlas",),
        note=f"ATLAS protein-only, replica R{replica}, stride {stride}")


def get_dataset(name: str, atlas_replica: int = 1, atlas_stride: int = 100) -> Dataset:
    if name in STATIC:
        return STATIC[name]
    if name.startswith("atlas_"):
        return atlas_dataset(name[len("atlas_"):], atlas_replica, atlas_stride)
    raise KeyError(f"unknown dataset {name!r}")


def select(spec: str, atlas_ids=(), atlas_replica: int = 1, atlas_stride: int = 100) -> list:
    """Resolve 'paper', 'all', group names and/or dataset names (comma/space separated)."""
    out = []
    for tok in spec.replace(",", " ").split():
        if tok == "all":
            out += list(STATIC.values()) + [atlas_dataset(i, atlas_replica, atlas_stride) for i in atlas_ids]
        elif tok == "atlas":
            out += [atlas_dataset(i, atlas_replica, atlas_stride) for i in atlas_ids]
        elif tok in GROUPS:
            out += [ds for ds in STATIC.values() if tok in ds.groups]
        else:
            out.append(get_dataset(tok, atlas_replica, atlas_stride))
    seen, uniq = set(), []
    for ds in out:
        if ds.name not in seen:
            seen.add(ds.name)
            uniq.append(ds)
    return uniq
