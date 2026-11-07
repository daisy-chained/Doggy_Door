"""Door angle vs the surrounding wall, from sensor depth (Madelief 2026-11-07). numpy only, deterministic per seed.
Wall = RANSAC vertical plane over the depth points in a band around the door box; the door panel's distance to that plane says
which way it opens (open door: panel in front of the wall = pull) and whether the closed door sits in a recess (push)."""
import json, pathlib
import numpy as np
from PIL import Image

from .data import frame_dir

_K_CACHE = {}


def _points(depth, K, mask):
    """Back-project the valid (> 5 cm) depth pixels under a boolean mask -> N x 3 camera points (x right, y down, z forward)."""
    v, u = np.nonzero(mask)
    z = depth[v, u]
    ok = z > 0.05
    u, v, z = u[ok], v[ok], z[ok]
    return np.stack([(u - K[2]) * z / K[0], (v - K[5]) * z / K[4], z], 1)


def _wall(P, rng, iters=300, tol=0.02):
    """RANSAC vertical plane (|normal y| <= 0.15, 2 cm inlier band), refit on its inliers -> (point on plane, unit normal toward the camera) or None."""
    best, bn, ba = -1, None, None
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if abs(n[1]) > 0.15:
            continue
        s = int((np.abs((P - a) @ n) < tol).sum())
        if s > best:
            best, bn, ba = s, n, a
    if bn is None:
        return None
    inl = np.abs((P - ba) @ bn) < tol
    m = P[inl].mean(0)
    n = np.linalg.svd(P[inl] - m, full_matrices=False)[2][2]
    return m, (-n if n @ m > 0 else n)


def geometry(depth_m, K, box_xywh, seed=0):
    """Door box vs the wall around it. depth_m: H x W metres, K: 9 intrinsics (row-major 3x3), box xywh in pixels.
    -> {"panel_off": median signed distance (+ = toward the camera) of box points 0.04..1.2 m off the wall (None if < 20),
        "near_wall": same for box points within 0.12 m of the wall (None if < 20)}, or None when no wall is found
    (< 100 band points or no plane). Band = same rows as the box, widened by its width each side, box left out; at most 4000 band points are used."""
    H, W = depth_m.shape
    x, y, w, h = [int(round(float(v))) for v in box_xywh]
    y0, y1, x0, x1 = max(0, y), min(H, y + h), max(0, x), min(W, x + w)
    band = np.zeros((H, W), bool)
    band[y0:y1, max(0, x - w):min(W, x + 2 * w)] = True
    box = np.zeros((H, W), bool)
    box[y0:y1, x0:x1] = True
    P = _points(depth_m, K, band & ~box)
    if len(P) < 100:
        return None
    rng = np.random.default_rng(seed)
    r = _wall(P[rng.choice(len(P), min(len(P), 4000), replace=False)], rng)
    if r is None:
        return None
    m, n = r
    d = (_points(depth_m, K, box) - m) @ n
    off, near = d[(np.abs(d) > 0.04) & (np.abs(d) < 1.2)], d[np.abs(d) < 0.12]
    return {"panel_off": float(np.median(off)) if len(off) >= 20 else None, "near_wall": float(np.median(near)) if len(near) >= 20 else None}


def depth_for(image_path):
    """Export image path -> (sensor depth in metres float32, K list) from <frame>/depth.png (mm) and <dataset>/dataset.json, or None without a depth.png."""
    frame = frame_dir(image_path)
    f = frame / "depth.png"
    if not f.exists():
        return None
    ds = frame.parent.parent
    if ds not in _K_CACHE:
        _K_CACHE[ds] = json.loads((ds / "dataset.json").read_text())["camera_intrinsics"]["K"]
    return np.asarray(Image.open(f), np.float32) / 1000.0, _K_CACHE[ds]
