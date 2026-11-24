"""How smoothly do the Stage-1 latents move along each trajectory? (CPU, seconds)

    python -m flow.latent_check                 # run "flow"
    python -m flow.latent_check flow_s1

Per case, on the encoded training frames (z: frames x 256 tokens x 32), all as RMS per
latent value and relative to the case's own spread (RMS of z minus its time mean):
    step        next training frame minus this one       (~1.41 = consecutive frames unrelated)
    lerp        frame minus the mean of its two neighbours (what interpolation must fix)
    copy        frame minus the previous frame            (= step; what copy-last must fix)
    case/global spread of the case vs. spread over all cases and tokens (Stage 2's unit)
Small step/lerp relative to 1 = the latent changes smoothly in time and Stage 2 has an easy
target; values near 1-1.4 = codes of neighbouring frames are nearly independent.
"""
import sys
from pathlib import Path

import torch

FLOW_EXP = Path(__file__).resolve().parent.parent / "exp" / "flow"


def rms(x):
    return float(x.pow(2).mean().sqrt())


def main(run="flow"):
    lat = FLOW_EXP / run / "latents"
    files = sorted(lat.glob("*.pt"))
    if not files:
        raise SystemExit(f"no latents in {lat}")
    data = {f.stem: torch.load(f, map_location="cpu", weights_only=False) for f in files}
    allz = torch.cat([d["z"].float() for d in data.values()])
    glob = rms(allz - allz.reshape(-1, allz.shape[-1]).mean(0))
    print(f"{run}: global spread (RMS about the per-channel mean, Stage 2's unit) = {glob:.3f}")
    print(f"{'case':22s} {'frames':>6s} {'spread':>7s} {'step':>6s} {'lerp':>6s}  "
          f"{'spread/global':>13s}  {'step(raw)/global':>16s}")
    for case, d in data.items():
        z = d["z"].float()
        spread = rms(z - z.mean(0))
        step = rms(z[1:] - z[:-1])
        lerp = rms(z[1:-1] - 0.5 * (z[:-2] + z[2:]))
        print(f"{case:22s} {len(z):6d} {spread:7.3f} {step / spread:6.3f} {lerp / spread:6.3f}  "
              f"{spread / glob:13.3f}  {step / glob:16.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
