"""Occupancy-grid doorway finder on radar points (milliMap-style, arXiv 1911.00398; Madelief 2026-11-04, "just do both").

Top-down grid: every return makes its cell occupied (counts). Free space is the sensor->return ray: a ray that crosses the
wall line (starts on one side, ends beyond the wall band on the other) marks the wall position where it crosses as free.
(Cell-wise free counting made a ray's last cells in front of the wall, which are inside the wall band, count as free; the
crossing test is the same information without that artefact.) Stacked views come from several sensor positions, so a doorway
one view sees through gets free rays from it. Wall line = RANSAC on the occupied cells weighted by their count. Along the
line a bin is "wall" when occupied clearly more than free (occ - LAMBDA * free > 0) and not just a few stray points
(WALL_FRAC of the median density, same rule as radar_doorways). A doorway = run of non-wall bins of door width with wall on
both sides. Score in [0, 1] = BLIND_SCORE + (1 - BLIND_SCORE) * share of the run's bins that rays went through, so a gap
nobody saw through is still a doorway, just a weaker one.

Same three-step API as radar_doorways (prepare -> fit_walls -> doorways) so tuning can cache the expensive steps.
find(points, sensors, cam_pos, params=None); params override the constants here and in radar_doorways by name, for that call.
"""
import pathlib, sys
import numpy as np

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import radar_doorways as rd

CELL_M = 0.05           # grid cell size (m)
MAX_CELLS = 3000        # cells per side; the cell grows if the extent is larger
LAMBDA = 0.3            # a bin is wall when occupied - LAMBDA * free > 0
BLIND_SCORE = 0.3       # score of a gap no ray went through (+ up to 0.7 for the share rays went through)
FREE_MIN = 1.0          # a bin counts as seen-through when its free count is at least this
BIN_M = 0.05            # bin size along the wall (m); sparse walls use the median cell spacing if larger


def _cfg(params):
    c = rd._cfg(None)
    c.update({k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float))})
    c.update(params or {})
    return c


def prepare(points, sensors, params=None):
    """Build the grid: occupied counts (nx, ny) + the rays (sensor xy -> hit xy) of the points in the height band."""
    c = _cfg(params)
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    sen = np.asarray(sensors, dtype=float).reshape(-1, 3) if sensors is not None else np.zeros_like(pts)
    ok = np.isfinite(pts).all(1) & np.isfinite(sen).all(1) & (pts[:, 2] >= c["Z_MIN"]) & (pts[:, 2] <= c["Z_MAX"])
    pxy, sxy = pts[ok, :2], sen[ok, :2]
    if len(pxy) == 0:
        return None
    lo = np.minimum(pxy.min(0), sxy.min(0)) - 0.2
    hi = np.maximum(pxy.max(0), sxy.max(0)) + 0.2
    cell = max(c["CELL_M"], float((hi - lo).max()) / c["MAX_CELLS"])
    nx, ny = (np.ceil((hi - lo) / cell).astype(int) + 1)
    pi = np.clip(((pxy - lo) / cell).astype(int), 0, [nx - 1, ny - 1])
    si = np.clip(((sxy - lo) / cell).astype(int), 0, [nx - 1, ny - 1])
    occ = np.bincount(pi[:, 0] * ny + pi[:, 1], minlength=nx * ny).astype(float)
    return {"lo": lo, "cell": cell, "nx": nx, "ny": ny, "occ": occ.reshape(nx, ny), "pxy": pxy, "sxy": sxy}


def _fit_line_w(xy, w, rng, c):
    """Weighted RANSAC line on cell centres -> (point, direction, inlier mask)."""
    tol = c["INLIER_TOL"] + c["CELL_M"] / 2
    if len(xy) > c["RANSAC_MAX_PTS"]:
        sel = rng.choice(len(xy), int(c["RANSAC_MAX_PTS"]), replace=False)
    else:
        sel = np.arange(len(xy))
    sub, sw = xy[sel], w[sel]
    best, best_n = None, 0.0
    for _ in range(int(c["RANSAC_ITERS"])):
        i, j = rng.choice(len(sub), 2, replace=False)
        d = sub[j] - sub[i]
        Ln = np.hypot(*d)
        if Ln < 0.3:
            continue
        d = d / Ln
        n = float(sw[np.abs((sub - sub[i]) @ np.array([-d[1], d[0]])) < tol].sum())
        if n > best_n:
            best, best_n = (sub[i], d), n
    if best is None:
        return None
    p, d = best
    mask = np.abs((xy - p) @ np.array([-d[1], d[0]])) < tol
    if mask.sum() >= 2:
        q, qw = xy[mask], w[mask]
        cc = (q * qw[:, None]).sum(0) / qw.sum()
        d = np.linalg.svd((q - cc) * np.sqrt(qw)[:, None], full_matrices=False)[2][0]
        p = cc
        mask = np.abs((xy - p) @ np.array([-d[1], d[0]])) < tol
    return p, d, mask


def fit_walls(grid, params=None):
    """Up to MAX_WALLS lines from the occupied cells -> [{"p","d","t","occ" (per occupied band cell),"tocc","tfree" (where rays cross the line)}]."""
    if grid is None:
        return []
    c = _cfg(params)
    rng = np.random.default_rng(0)
    cell, lo = grid["cell"], grid["lo"]
    ix, iy = np.nonzero(grid["occ"] > 0)
    cxy = lo + (np.stack([ix, iy], 1) + 0.5) * cell
    occ = grid["occ"][ix, iy]
    live = occ > 0
    S, H = grid["sxy"], grid["pxy"]
    out = []
    tol = c["INLIER_TOL"] + cell / 2
    for _ in range(int(c["MAX_WALLS"])):
        if occ[live].sum() < c["MIN_WALL_PTS"]:
            break
        idx = np.nonzero(live)[0]
        fit = _fit_line_w(cxy[idx], occ[idx], rng, c)
        if fit is None:
            break
        p, d, m = fit
        nrm = np.array([-d[1], d[0]])
        if occ[idx[m]].sum() < c["MIN_WALL_PTS"]:
            break
        t_all = (cxy - p) @ d
        band = np.abs((cxy - p) @ nrm) < tol
        occ_t = t_all[band & live]
        if occ_t.size and occ_t.max() - occ_t.min() >= c["MIN_WALL_LEN"]:
            # rays that start on one side and end beyond the band on the other cross the wall line: free there
            ps, ph = (S - p) @ nrm, (H - p) @ nrm
            cross = (ps * ph < 0) & (np.abs(ps) > tol) & (np.abs(ph) > tol)
            u = ps[cross] / (ps[cross] - ph[cross])
            tq = ((S[cross] + u[:, None] * (H[cross] - S[cross])) - p) @ d
            out.append({"p": p, "d": d, "t": t_all[band & live], "occ": occ[band & live], "tocc": np.sort(occ_t), "tfree": tq})
        live = live & ~band
    return out


def doorways(walls, cam_pos, params=None):
    """Gaps along the wall bins -> [{"center","width","normal","score"}]; normal towards cam_pos."""
    c = _cfg(params)
    cam = np.asarray(cam_pos, dtype=float)
    out = []
    for w in walls:
        p, d, t = w["p"], w["d"], w["t"]
        t0 = float(w["tocc"][0])
        ut = np.unique(np.round(w["tocc"], 4))
        bw = max(c["BIN_M"], float(np.median(np.diff(ut))) if len(ut) > 1 else c["BIN_M"])
        nb = int(np.ceil((w["tocc"][-1] - t0) / bw)) + 1
        bi = np.clip(np.floor((t - t0) / bw + 1e-9).astype(int), 0, nb - 1)
        occ_b = np.bincount(bi, weights=w["occ"], minlength=nb)
        fi = np.floor((w["tfree"] - t0) / bw).astype(int)
        fi = fi[(fi >= 0) & (fi < nb)]
        free_b = np.bincount(fi, minlength=nb).astype(float)
        med = np.median(occ_b[occ_b > 0])
        wall = (occ_b - c["LAMBDA"] * free_b > 0) & (occ_b >= max(c["WALL_FRAC"] * med, 1e-9))
        nrm = np.array([-d[1], d[0]])
        for a, b, i0, j0 in rd._runs_to_gaps(wall, t0, bw, c):
            ctr = p + d * (a + b) / 2
            n = nrm if (cam[:2] - ctr) @ nrm >= 0 else -nrm
            seen = float((free_b[i0:j0] >= c["FREE_MIN"]).mean())
            out.append({"center": [float(ctr[0]), float(ctr[1]), c["DOOR_Z"]], "width": float(b - a),
                        "normal": [float(n[0]), float(n[1]), 0.0], "score": c["BLIND_SCORE"] + (1 - c["BLIND_SCORE"]) * seen})
    return out


def find(points, sensors, cam_pos, params=None):
    """Doorways in world points (N, 3) seen from sensors (N, 3) (the radar position of each point; stacked views have
    several) -> [{"center","width","normal","score"}]."""
    return doorways(fit_walls(prepare(points, sensors, params), params), cam_pos, params)
