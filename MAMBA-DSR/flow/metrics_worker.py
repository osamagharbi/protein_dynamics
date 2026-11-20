"""Metric worker for MAMBA-Flow evaluation (light imports: runs in spawned processes).

Mirror of evaluate_protocol.evaluate_frame: the marching-cubes output is turned into a
trimesh exactly as DSR's get_mesh does, exported to OBJ and reloaded, then scored with
DSR's metrics in normalized coordinates against the normalized ground truth.
"""
import os
import sys
import tempfile

DSR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "DSR")


def init_worker():
    if DSR_DIR not in sys.path:
        sys.path.insert(0, DSR_DIR)


def metrics_job(job):
    """job = (gt_path, s, t, resolution, {label: (verts, faces, normals, values) or None})
    -> {label: {iou_mean, chamfer_mean, normal_consistency_mean} or None (no mesh)}."""
    init_worker()
    import trimesh
    from lib.metrics import compute_iou, compute_normal_consistency, compute_trimesh_chamfer
    gt_path, s, t, resolution, preds = job
    gt = trimesh.load(gt_path, process=False, force="mesh", skip_materials=True)
    gt.vertices = ((gt.vertices - t) / s) * 2.0 - 1.0
    out = {}
    with tempfile.TemporaryDirectory(prefix="flow_eval_") as tmp:
        for label, mc in preds.items():
            if mc is None:
                out[label] = None
                continue
            verts, faces, normals, values = mc
            path = os.path.join(tmp, f"{label}.obj")
            trimesh.Trimesh(verts, faces, normals, vertex_colors=values).export(path, "obj")
            pred = trimesh.load(path, process=False, force="mesh", skip_materials=True)
            out[label] = {"iou_mean": float(compute_iou(gt, pred, N=resolution)),
                          "chamfer_mean": float(compute_trimesh_chamfer(pred, gt)),
                          "normal_consistency_mean": float(compute_normal_consistency(gt, pred)),
                          "faces": int(len(pred.faces))}
    return out
