"""Doorway box: the frame opening of a door (wall face, width x height) projected into the image and cut to the part the
depth shows. Madelief 2026-11-04: a "doorway" class next to "door" (= the panel), so the detector also finds the opening
when the panel is open, glass or gone in smoke. Geometry comes from the v4 "approach" entry of the camera's side
(point on the wall face + horizontal direction to the camera); no depth return (0 / NaN) counts as see-through = visible,
a closed panel a few cm behind the face too (TOL). Depth = planar z-depth in metres. numpy only.
"""
import numpy as np

from doorsim import frames

TOL = 0.05     # m: depth this far in front of the wall face still counts as the opening being visible
MIN_PX = 4     # smaller boxes are dropped (as good as invisible)
NEAR = 0.05    # m: points closer than this to the camera plane are not projected
N_ALONG, N_UP = 41, 81


def own_ids(meta, door):
    """Segmentation ids of the door itself: its panel, parts (glass) and handles. They never hide its own doorway."""
    objs = meta["objects"]
    return [objs[door]["id"]] + [o["id"] for o in objs.values() if o.get("part_of") == door or o.get("door") == door]


def box(meta, door, K, depth, seg=None):
    """-> {"box": {x_min, y_min, x_max, y_max} inclusive pixel indices, "visible_fraction"} or None when not visible.
    seg (optional, same size): pixels of the door's own panel / glass / handles count as visible (a sliding panel hung on
    the camera's face sits in front of the wall face but is still in its doorway)."""
    o = meta["objects"][door]
    a = o["approach"][meta["side"]]
    p, d = np.array(a["point"], float), np.array(a["direction"], float)
    along = np.array([-d[1], d[0], 0.0])  # horizontal, perpendicular to the wall normal
    along /= np.linalg.norm(along)
    s, z = np.meshgrid(np.linspace(-o["width"] / 2, o["width"] / 2, N_ALONG), np.linspace(0, o["height"], N_UP))
    pts = np.stack([p[0] + s * along[0], p[1] + s * along[1], z], -1).reshape(-1, 3)
    cam = meta["camera"]
    body = (pts - np.array(cam["world_position"])) @ np.array(frames.rot(*cam["world_rpy"]))  # rows: (R^T v)^T
    x, y, zc = -body[:, 1], -body[:, 2], body[:, 0]  # body -> optical
    fx, cx, fy, cy = K[0], K[2], K[4], K[5]
    n = len(pts)
    H, W = depth.shape
    ok = zc > NEAR
    zs = np.where(ok, zc, 1.0)
    uf, vf = np.floor(fx * x / zs + cx), np.floor(fy * y / zs + cy)
    inside = ok & (uf >= 0) & (uf < W) & (vf >= 0) & (vf < H)
    # the grid is coarse near the image edge: points outside are clamped to the border pixel for the box (so a doorway
    # running out of the image reaches the edge), but only points really inside count for the fraction
    u, v, zc, inside = np.clip(uf, 0, W - 1)[ok].astype(int), np.clip(vf, 0, H - 1)[ok].astype(int), zc[ok], inside[ok]
    dv = depth[v, u]
    vis = ~np.isfinite(dv) | (dv <= 0) | (dv >= zc - TOL)
    if seg is not None:
        vis |= np.isin(seg[v, u], own_ids(meta, door))
    if not (vis & inside).any():
        return None
    u, v = u[vis], v[vis]
    if u.max() - u.min() + 1 < MIN_PX or v.max() - v.min() + 1 < MIN_PX:
        return None
    return {"box": {"x_min": int(u.min()), "y_min": int(v.min()), "x_max": int(u.max()), "y_max": int(v.max())},
            "visible_fraction": float((vis & inside).sum() / n)}
