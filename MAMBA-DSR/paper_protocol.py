"""Explicit data protocol for comparison with the published DSR experiments.

The paper does not train on every raw frame for the reconstruction table:

* AdK equilibrium: first 1500 frames, every 10th frame (150 frames)
* I-FABP: first 400 frames, every 5th frame (80 frames)
* long trajectories: crop to 2000 frames before applying the experiment rule

For the A-beta trajectories the paper explicitly says that the first 2000
frames were used, but does not state a thinning interval in the main text.
This manifest therefore treats those first 2000 frames as the documented
training set and records that choice explicitly instead of silently inheriting
the old every-fifth-frame preprocessing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PaperCase:
    name: str
    full_dir: Path
    gt_case: str
    paper_limit: Optional[int]
    train_start: int
    train_stop: Optional[int]
    train_step: int
    note: str

    def train_indices(self, available_frames: int) -> list[int]:
        limit = available_frames if self.paper_limit is None else min(
            available_frames, self.paper_limit
        )
        stop = limit if self.train_stop is None else min(limit, self.train_stop)
        return list(range(self.train_start, stop, self.train_step))

    def required_frames(self) -> int:
        if self.paper_limit is not None:
            return self.paper_limit
        if self.train_stop is not None:
            return self.train_stop
        return 0


# Rule for prepared datasets whose paper training rule is unknown (GPCRmd):
# the same "first 1500, every 10th" used for AdK / NhaA / YiiP / Dryad.
DEFAULT_RULE = (1500, 10)


def discovered_cases(processed: Path) -> dict[str, PaperCase]:
    """Every complete dataset written by jeanzay/data_prep (``<name>/meta.json``)."""
    import json
    cases = {}
    for meta_path in sorted(processed.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("status") != "complete":
            continue
        name, n = meta_path.parent.name, int(meta["n_expected"])
        rule = meta.get("paper_train_rule") or {}
        if rule.get("step"):
            stop, step = rule.get("stop") or n, rule["step"]
            note = f"jeanzay registry rule: first {min(stop, n)}, every {step}"
        else:
            stop, step = DEFAULT_RULE
            note = (f"NON-PAPER default: first {min(stop, n)}, every {step} "
                    "(paper rule for this dataset unverified)")
        cases[name] = PaperCase(name, meta_path.parent / "gt", f"mamba_{name}_gt",
                                n, 0, stop, step, note)
    return cases


def paper_cases(project_root: Path, dsr_root: Path) -> dict[str, PaperCase]:
    processed = project_root / "datasets" / "DSR_processed"
    return {**discovered_cases(processed), **_fixed_cases(processed, dsr_root)}


def _fixed_cases(processed: Path, dsr_root: Path) -> dict[str, PaperCase]:
    return {
        # Official DSR demo: 30 observed frames, every fifth frame, with 149
        # available ground-truth frames in the checked-in data.
        "abeta_30": PaperCase(
            "abeta_30", dsr_root / "datasets" / "abeta_150" / "obj",
            "abeta_30_gt", 149, 0, 150, 5,
            "official DSR demo split: 0,5,...,145",
        ),
        # The paper states first 2000 frames for the Abeta trajectories, but
        # does not state a thinning rule in the main text.
        "abeta_wt": PaperCase(
            "abeta_wt", processed / "abeta_wt" / "gt",
            "mamba_abeta_wt_gt", 2000, 0, 2000, 1,
            "paper: first 2000 frames; no thinning interval stated",
        ),
        "abeta_e22g": PaperCase(
            "abeta_e22g", processed / "abeta_e22g" / "gt",
            "mamba_abeta_e22g_gt", 2000, 0, 2000, 1,
            "paper: first 2000 frames; no thinning interval stated",
        ),
        "adk_equilibrium": PaperCase(
            "adk_equilibrium", processed / "adk_equilibrium" / "gt",
            "mamba_adk_equilibrium_gt", 1500, 0, 1500, 10,
            "paper Table 1: first 1500 frames diluted by 10",
        ),
        "ifabp_water": PaperCase(
            "ifabp_water", processed / "ifabp_water" / "gt",
            "mamba_ifabp_water_gt", 500, 0, 400, 5,
            "paper Table 1: every fifth frame in first 400",
        ),
    }
