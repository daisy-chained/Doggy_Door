"""Fog of war for the run dashboard (Madelief 2026-11-11): what the robot camera has seen, on a coarse grid over the floor map.
Rules (view cone, range, walls block, row 0 = bottom) are in sim/tests/test_fog.py. numpy + math only."""
import math
import numpy as np


def new(m, cell=0.25):
    """Empty fog grid over map m: seen[j, i], row 0 = the bottom (lowest y)."""
    h, w = m["occ"].shape
    cols = int(math.ceil(round(w * m["res"] / cell, 6)))
    rows = int(math.ceil(round(h * m["res"] / cell, 6)))
    return {"cell": cell, "origin": tuple(m["origin"]), "seen": np.zeros((rows, cols), bool)}


def look(f, m, cam_xy, yaw, hfov=1.3, rng=4.0):
    """Cast rays over the view cone, mark the fog cells they cross (stop at the first occupied cell, which is marked).
    Returns the number of newly seen fog cells."""
    occ, res = m["occ"], m["res"]
    h, w = occ.shape
    mx0, my0 = m["origin"]
    n_rays = int(math.ceil(hfov * rng / res)) + 1
    n_steps = int(math.ceil(rng / (res / 2))) + 1
    ang = yaw + np.linspace(-hfov / 2, hfov / 2, n_rays)
    t = np.arange(n_steps) * (res / 2)
    t = t[t <= rng + 1e-9]
    px = cam_xy[0] + np.cos(ang)[:, None] * t[None, :]
    py = cam_xy[1] + np.sin(ang)[:, None] * t[None, :]
    col = np.floor((px - mx0) / res).astype(int)
    row = h - 1 - np.floor((py - my0) / res).astype(int)
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    hit = np.zeros(inside.shape, bool)
    hit[inside] = occ[row[inside], col[inside]]
    blocked = hit | ~inside
    first = np.where(blocked.any(1), blocked.argmax(1), blocked.shape[1])
    keep = np.arange(blocked.shape[1])[None, :] <= first[:, None]
    keep &= inside                                   # the occupied cell itself is kept, the outside of the map is not
    seen = f["seen"]
    fx0, fy0 = f["origin"]
    ci = np.floor((px[keep] - fx0) / f["cell"]).astype(int)
    cj = np.floor((py[keep] - fy0) / f["cell"]).astype(int)
    ok = (ci >= 0) & (ci < seen.shape[1]) & (cj >= 0) & (cj < seen.shape[0])
    ci, cj = ci[ok], cj[ok]
    before = int(seen.sum())
    seen[cj, ci] = True
    return int(seen.sum()) - before


def encode(f):
    """JSON-ready: rows[j] = "0"/"1" per column for seen row j (row 0 = bottom)."""
    return {"cell": f["cell"], "origin": [float(f["origin"][0]), float(f["origin"][1])],
            "rows": ["".join("1" if v else "0" for v in r) for r in f["seen"]]}
