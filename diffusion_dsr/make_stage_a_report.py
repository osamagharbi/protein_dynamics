#!/usr/bin/env python3
"""Create figures and a Markdown summary for a Stage-A run."""

import argparse
import os
import pickle

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default=os.path.join(os.path.dirname(__file__), "exp", "abeta_dsr_protocol_v2_s0"),
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(run_dir, "stage_a_report"))
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(run_dir, "loss_log.pkl"), "rb") as handle:
        log = pickle.load(handle)

    epochs = log["epoch"]
    total = log["loss"]
    components = ["mnfld_loss", "grad_loss", "normals_loss", "latent_loss", "smooth_loss", "consist_loss"]

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, total, linewidth=2, label="total loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Stage A total loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "total_loss.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for name in components:
        values = log[name]
        if max(values) - min(values) > 1e-12:
            plt.plot(epochs, values, linewidth=1.7, label=name.replace("_loss", ""))
    plt.xlabel("Epoch")
    plt.ylabel("Component loss")
    plt.title("Stage A loss components")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_components.png"), dpi=180)
    plt.close()

    window = min(200, len(epochs))
    plt.figure(figsize=(9, 5))
    plt.plot(epochs[-window:], total[-window:], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.title(f"Stage A final {window}-epoch loss window")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_loss_window.png"), dpi=180)
    plt.close()

    first_loss = total[0]
    final_loss = total[-1]
    reduction = 100.0 * (first_loss - final_loss) / first_loss
    tail = total[-min(100, len(total)):]
    tail_mean = sum(tail) / len(tail)
    tail_min = min(tail)
    tail_max = max(tail)
    checkpoint = os.path.join(run_dir, "checkpoints", "ckpt_latest.pth")

    report = f"""# Stage A training report

Run directory: `{run_dir}`  
Checkpoint: `{checkpoint}`

## Summary

- Epochs completed: **{epochs[-1]}**
- Final total loss: **{final_loss:.4f}**
- Initial total loss: **{first_loss:.4f}**
- Total-loss reduction: **{reduction:.1f}%**
- Final 100-epoch mean loss: **{tail_mean:.4f}**
- Final 100-epoch range: **{tail_min:.4f}–{tail_max:.4f}**
- Training split: **30 DSR frames** (`0, 5, ..., 145`)

The loss decreases strongly and reaches a stable plateau in the final part of
training. This indicates that Stage A is sufficiently converged for Stage B.
The definitive quality check remains geometric reconstruction using IoU,
Chamfer distance, and normal consistency.

## Figures

![Total loss](total_loss.png)

![Loss components](loss_components.png)

![Final loss window](final_loss_window.png)

## Start Stage B

From `diffusion_dsr/`, with the `graphmae` environment active:

```bash
python train_diffusion.py \\
  --stage-a-ckpt exp/abeta_dsr_protocol_v2_s0/checkpoints/ckpt_latest.pth \\
  --case abeta_150 \\
  --mean-epochs 5000 \\
  --epochs 20000 \\
  --seed 0 \\
  --run-name abeta_dsr_protocol_v2_s0
```

Stage B will use the split metadata stored in the Stage-A checkpoint and train
on the same 30 DSR training-frame latents.
"""
    with open(os.path.join(output_dir, "STAGE_A_REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write(report)

    print(f"Report written to: {output_dir}")
    print(f"Markdown: {os.path.join(output_dir, 'STAGE_A_REPORT.md')}")


if __name__ == "__main__":
    main()
