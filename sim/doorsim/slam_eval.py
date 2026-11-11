"""Score a SLAM map against the true floor map (see sim/tests/test_slam_eval.py for the contract).

Grids are int8 HxW (1 occupied, 0 free, -1 unknown), row 0 = TOP, as ROS map yaml + pgm.
"""
import math
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None


def load(yaml_path):
    """Read a ROS map yaml + pgm -> {"grid", "res", "origin"}."""
    yaml_path = Path(yaml_path)
    cfg = yaml.safe_load(yaml_path.read_text())
    p = np.asarray(Image.open(yaml_path.parent / cfg["image"]).convert("L"), dtype=np.int32)
    if cfg.get("negate", 0):
        p = 255 - p
    grid = np.full(p.shape, -1, np.int8)
    grid[p <= 100] = 1
    grid[p >= 250] = 0
    return {"grid": grid, "res": float(cfg["resolution"]), "origin": (float(cfg["origin"][0]), float(cfg["origin"][1]))}


def save(m, stem):
    """Write stem.pgm + stem.yaml (ROS trinary: 0 occupied, 254 free, 205 unknown)."""
    stem = Path(stem)
    g = np.asarray(m["grid"])
    px = np.full(g.shape, 205, np.uint8)
    px[g == 1] = 0
    px[g == 0] = 254
    pgm = stem.with_name(stem.name + ".pgm")
    Image.fromarray(px, mode="L").save(pgm)
    cfg = {
        "image": pgm.name,
        "mode": "trinary",
        "resolution": float(m["res"]),
        "origin": [float(m["origin"][0]), float(m["origin"][1]), 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.25,
    }
    stem.with_name(stem.name + ".yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def _dilate(mask, n):
    """True where any cell within Chebyshev distance n is True."""
    if n <= 0:
        return mask.copy()
    if _ndi is not None:
        return _ndi.binary_dilation(mask, structure=np.ones((3, 3), bool), iterations=n)
    h, w = mask.shape
    pad = np.pad(mask, n, constant_values=False)
    out = np.zeros_like(mask)
    for dr in range(2 * n + 1):
        for dc in range(2 * n + 1):
            out |= pad[dr:dr + h, dc:dc + w]
    return out


def _place(grid, off_r, off_c, shape, fill):
    out = np.full(shape, fill, np.int8)
    h, w = grid.shape
    out[off_r:off_r + h, off_c:off_c + w] = grid
    return out


def _frac(num, den):
    return float(num) / float(den) if den else 0.0


def compare(true_m, slam_m, tol=0.1):
    res = float(true_m["res"])
    tg, sg = np.asarray(true_m["grid"]), np.asarray(slam_m["grid"])
    # common canvas in world coordinates (union of extents), rows from the top
    def ext(m, g):
        x0, y0 = m["origin"][0], m["origin"][1]
        return x0, x0 + g.shape[1] * res, y0, y0 + g.shape[0] * res
    tx0, tx1, ty0, ty1 = ext(true_m, tg)
    sx0, sx1, sy0, sy1 = ext(slam_m, sg)
    ux0, ux1, uy1 = min(tx0, sx0), max(tx1, sx1), max(ty1, sy1)
    uy0 = min(ty0, sy0)
    W = int(round((ux1 - ux0) / res))
    H = int(round((uy1 - uy0) / res))
    shape = (H, W)
    t_off = (int(round((uy1 - ty1) / res)), int(round((tx0 - ux0) / res)))
    s_off = (int(round((uy1 - sy1) / res)), int(round((sx0 - ux0) / res)))
    T = _place(tg, *t_off, shape, 1)    # outside true map counts as occupied
    S = _place(sg, *s_off, shape, -1)   # outside slam map counts as unknown
    region = np.zeros(shape, bool)
    region[t_off[0]:t_off[0] + tg.shape[0], t_off[1]:t_off[1] + tg.shape[1]] = True

    n = int(math.ceil(tol / res - 1e-9))
    t_free = (T == 0)
    t_occ = (T == 1)
    s_occ = (S == 1)
    s_free = (S == 0)

    # wall surface: true occupied with a free 4-neighbour (inside the true map)
    nb = np.zeros(shape, bool)
    nb[1:, :] |= t_free[:-1, :]
    nb[:-1, :] |= t_free[1:, :]
    nb[:, 1:] |= t_free[:, :-1]
    nb[:, :-1] |= t_free[:, 1:]
    wall = t_occ & nb & region

    n_free = int(t_free.sum())
    n_wall = int(wall.sum())
    n_slam_occ = int(s_occ.sum())
    coverage = _frac((t_free & s_free).sum(), n_free)
    walls_found = _frac((wall & _dilate(s_occ, n)).sum(), n_wall)
    false_walls = _frac((s_occ & ~_dilate(t_occ, n)).sum(), n_slam_occ)
    return {"coverage": coverage, "walls_found": walls_found, "false_walls": false_walls,
            "n_free": n_free, "n_wall": n_wall, "n_slam_occ": n_slam_occ}
