#!/usr/bin/env python3
"""
Run a DSR reconstruction at a given resolution and organize the result under
reconstructions/<run_name>/{obj,previews}/, including a rendered GIF preview.

Usage:
    conda activate graphmae
    python scripts/reconstruct_and_preview.py --resolution 128 --run-name DSR_abeta_30_demo_res128

Assumes the standard demo checkpoint at
DSR/exp/<case>/igrt_debug_sphere/*/checkpoints/ckpt_latest.pth already exists
(i.e. --is_continue can find something to load).
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DSR_DIR = os.path.join(PROJECT_ROOT, "DSR")
RECON_ROOT = os.path.join(PROJECT_ROOT, "reconstructions")


def make_conf(base_conf, resolution, tag):
    with open(base_conf) as f:
        text = f.read()
    new_text = re.sub(
        r"(reconstruction\s*\{\s*\n\s*resolution\s*=\s*)\d+",
        rf"\g<1>{resolution}",
        text,
    )
    if new_text == text:
        raise RuntimeError("Could not find 'reconstruction { resolution = ... }' block to patch")
    out_path = os.path.join(DSR_DIR, "confs", f"_auto_{tag}.conf")
    with open(out_path, "w") as f:
        f.write(new_text)
    return out_path


def run_reconstruction(conf_path, case, gpu, log_path):
    cmd = [
        sys.executable, "-u", "runner_igr_t.py",
        "--conf", conf_path,
        "--mode", "reconstruction",
        "--gpu", str(gpu),
        "--case", case,
        "--is_continue",
    ]
    print(f"[run] {' '.join(cmd)}")
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=DSR_DIR, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    with open(log_path) as log:
        text = log.read()
    if proc.returncode != 0 or "Traceback" in text:
        print(text[-4000:])
        raise RuntimeError(f"Reconstruction failed (see {log_path})")
    n_done = text.count("it [") and re.findall(r"(\d+)it \[", text)
    print(f"[run] finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return elapsed


def find_output_dir(case):
    pattern = os.path.join(DSR_DIR, "exp", case, "*", "*", "reconstruction_plots", "seq_0_obj")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not matches:
        raise RuntimeError(f"No reconstruction_plots/seq_0_obj found for case {case}")
    return matches[-1]


def organize(src_dir, run_name):
    dest = os.path.join(RECON_ROOT, run_name)
    obj_dest = os.path.join(dest, "obj")
    preview_dest = os.path.join(dest, "previews")
    os.makedirs(obj_dest, exist_ok=True)
    os.makedirs(preview_dest, exist_ok=True)
    objs = sorted(glob.glob(os.path.join(src_dir, "*.obj")))
    for f in objs:
        os.rename(f, os.path.join(obj_dest, os.path.basename(f)))
    print(f"[organize] moved {len(objs)} .obj files -> {obj_dest}")
    return dest, obj_dest, preview_dest


def render_preview(obj_dir, preview_dest, fps=15, target_faces=6000):
    import numpy as np
    import trimesh
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import imageio.v2 as imageio

    files = sorted(
        glob.glob(os.path.join(obj_dir, "*.obj")),
        key=lambda f: int(re.search(r"frame_(\d+)_", f).group(1)),
    )
    print(f"[preview] {len(files)} frames to render")

    meshes = []
    all_verts = []
    for f in files:
        m = trimesh.load(f, process=False)
        if len(m.faces) > target_faces:
            try:
                m = m.simplify_quadric_decimation(face_count=target_faces)
            except Exception:
                pass  # fall back to full-res mesh if decimation isn't available
        meshes.append(m)
        all_verts.append(m.vertices)
    all_verts = np.concatenate(all_verts, axis=0)
    xmin, ymin, zmin = all_verts.min(axis=0)
    xmax, ymax, zmax = all_verts.max(axis=0)

    frames_img = []
    t0 = time.time()
    for i, (f, mesh) in enumerate(zip(files, meshes)):
        fig = plt.figure(figsize=(5, 5), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        verts, faces = mesh.vertices, mesh.faces
        normals = mesh.face_normals
        shade = 0.35 + 0.65 * np.clip(normals[:, 2], 0, 1)
        colors = plt.cm.viridis(shade)
        poly = Poly3DCollection(verts[faces], facecolor=colors, edgecolor="none")
        ax.add_collection3d(poly)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=20, azim=45)
        ax.axis("off")
        tag = os.path.basename(f).split("_t_")[1].split("_seq")[0]
        ax.set_title(f"frame {i:03d}  t={tag}", fontsize=10)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames_img.append(img)
        plt.close(fig)
        if i % 25 == 0:
            print(f"[preview] rendered {i}/{len(files)}  ({time.time()-t0:.1f}s)")

    out_path = os.path.join(preview_dest, "reconstruction.gif")
    imageio.mimsave(out_path, frames_img, fps=fps, loop=0)
    print(f"[preview] saved {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB, "
          f"rendered in {time.time()-t0:.1f}s)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, required=True)
    ap.add_argument("--run-name", type=str, required=True,
                     help="folder name under reconstructions/, e.g. DSR_abeta_30_demo_res128")
    ap.add_argument("--case", type=str, default="abeta_30")
    ap.add_argument("--base-conf", type=str, default=os.path.join(DSR_DIR, "confs", "abeta.conf"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--skip-preview", action="store_true")
    ap.add_argument("--target-faces", type=int, default=6000,
                     help="decimate meshes to this many faces before rendering the GIF (speed)")
    args = ap.parse_args()

    conf_path = make_conf(args.base_conf, args.resolution, args.run_name)
    log_path = f"/tmp/dsr_reconstruction_{args.run_name}.log"
    elapsed = run_reconstruction(conf_path, args.case, args.gpu, log_path)

    src_dir = find_output_dir(args.case)
    dest, obj_dir, preview_dir = organize(src_dir, args.run_name)

    if not args.skip_preview:
        render_preview(obj_dir, preview_dir, target_faces=args.target_faces)

    print(f"\n[done] resolution={args.resolution}  time={elapsed:.1f}s  -> {dest}")


if __name__ == "__main__":
    main()
