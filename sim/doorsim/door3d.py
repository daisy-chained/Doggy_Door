"""A door's 3D position from its 2D box + depth: no extra model, just geometry (numpy only).

Why: the detector gives a 2D box; the approach planner needs where the doorway is and which way it faces. A plane fitted
(RANSAC, then least squares) to the depth inside the box gives centre, normal and size. Glass gives depth holes or the room
behind it, so then the plane of the wall/frame in a ring around the box is used instead.
Frames: ROS optical (x right, y down, z forward, metres); depth = planar z-depth; box = {x_min, y_min, x_max, y_max}
with inclusive pixel indices (so its edges in continuous pixel coordinates are x_min .. x_max + 1; pixel i has its centre at i + 0.5).

    python3 sim/doorsim/door3d.py DATASET [--out rows.jsonl]     # score GT boxes against the doorway ground truth
"""
import argparse, json, math, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from doorsim import frames

SHRINK = 0.1        # skip the frame edges: this fraction of the box size off each side
RING = 0.15         # ring margin around the box, as a fraction of the box size
MIN_VALID = 0.15    # box route needs at least this fraction of the shrunk box's pixels to have depth
MIN_INLIERS = 0.5   # ... and at least this fraction of those on one plane (glass with partial returns scores lower)
MIN_RING = 30       # fewest ring points worth fitting
THRESH = 0.025      # RANSAC inlier distance, m (sim depth is exact; real sensors are ~1-2 cm)
ITERS = 300
MAX_PTS = 3000      # RANSAC scores a random subset of this size (the refit uses all inliers)
BEHIND = 0.3        # box plane this far behind the ring's plane (along the box centre ray) = see-through glass


def _backproject(depth, K, ys, xs):
    z = depth[ys, xs].astype(np.float64)
    ok = np.isfinite(z) & (z > 0)
    ys, xs, z = ys[ok], xs[ok], z[ok]
    return np.stack([(xs + 0.5 - K[2]) / K[0] * z, (ys + 0.5 - K[5]) / K[4] * z, z], axis=1)


def _rect(box, depth_shape, grow):
    """(x0, y0, x1, y1) exclusive-end pixel rect of the box grown (grow > 0) or shrunk (grow < 0) by a fraction of its size, clipped."""
    x0, y0, x1, y1 = box["x_min"], box["y_min"], box["x_max"] + 1, box["y_max"] + 1
    dx, dy = grow * (x1 - x0), grow * (y1 - y0)
    h, w = depth_shape
    return (int(max(0, round(x0 - dx))), int(max(0, round(y0 - dy))), int(min(w, round(x1 + dx))), int(min(h, round(y1 + dy))))


def points(depth, K, box, shrink=SHRINK):
    """Valid depth pixels inside the box (shrunk by `shrink` of its size on each side) -> (N, 3) camera-frame points."""
    x0, y0, x1, y1 = _rect(box, depth.shape, -shrink)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return _backproject(depth, K, ys.ravel(), xs.ravel())


def ring_points(depth, K, box, margin=RING):
    """Valid depth pixels in the band just outside the box (the whole box excluded), clipped to the image -> (N, 3)."""
    x0, y0, x1, y1 = _rect(box, depth.shape, margin)
    mask = np.ones(depth.shape, bool)
    mask[:y0] = mask[y1:] = False
    mask[:, :x0] = mask[:, x1:] = False
    bx0, by0, bx1, by1 = _rect(box, depth.shape, 0.0)
    mask[by0:by1, bx0:bx1] = False
    ys, xs = np.nonzero(mask)
    return _backproject(depth, K, ys, xs)


def _plane3(p):
    """Planes (n, d) with n.x = d through each triple of points p (M, 3, 3); degenerate triples get n = 0."""
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return n, np.einsum("ij,ij->i", n, p[:, 0])


def plane_fit(pts, thresh=THRESH, iters=ITERS, seed=0):
    """Robust plane through (N, 3) points: RANSAC (fixed seed), then SVD least squares on the inliers.
    Returns (normal, d, inlier fraction) with normal . x = d, or None with < 3 points."""
    if len(pts) < 3:
        return None
    rng = np.random.default_rng(seed)
    sub = pts if len(pts) <= MAX_PTS else pts[rng.choice(len(pts), MAX_PTS, replace=False)]
    tri = sub[rng.integers(0, len(sub), (iters, 3))]
    n, d = _plane3(tri)
    count = (np.abs(sub @ n.T - d) < thresh).sum(axis=0)
    count[np.linalg.norm(n, axis=1) < 0.5] = 0
    n, d = n[count.argmax()], d[count.argmax()]
    for _ in range(2):  # refit on the inliers of the current plane, once more with the refitted plane
        inl = np.abs(pts @ n - d) < thresh
        if inl.sum() < 3:
            return None
        c = pts[inl].mean(axis=0)
        n = np.linalg.svd(pts[inl] - c, full_matrices=False)[2][-1]
        d = n @ c
    return n, d, float((np.abs(pts @ n - d) < thresh).mean())


def _ray(K, u, v):
    return np.array([(u - K[2]) / K[0], (v - K[5]) / K[4], 1.0])


def _hit(n, d, r):
    """Point where the ray t * r (r with z = 1) meets the plane n . x = d, or None when parallel / behind the camera."""
    den = n @ r
    t = d / den if abs(den) > 1e-9 else -1
    return t * r if t > 0 else None


def _geometry(plane, K, box):
    n, d, frac = plane
    x0, y0, x1, y1 = box["x_min"], box["y_min"], box["x_max"] + 1, box["y_max"] + 1
    um, vm = (x0 + x1) / 2, (y0 + y1) / 2
    c = _hit(n, d, _ray(K, um, vm))
    ends = [_hit(n, d, _ray(K, u, v)) for u, v in ((x0, vm), (x1, vm), (um, y0), (um, y1))]
    if c is None or any(e is None for e in ends):
        return None
    return c, (n if n @ c < 0 else -n), float(np.linalg.norm(ends[1] - ends[0])), float(np.linalg.norm(ends[3] - ends[2])), frac


def fit(depth, K, box):
    """Door pose from depth + 2D box -> {"center", "normal" (toward the camera), "width", "height", "source" "box" | "ring",
    "inliers" (fraction of the points used), "points"}, or None when there is no usable depth. Box route first; the ring
    around the box (glass: wall/frame) when the box has too little depth, a poor plane, or a plane clearly behind the ring's."""
    x0, y0, x1, y1 = _rect(box, depth.shape, -SHRINK)
    pts = points(depth, K, box)
    got = plane_fit(pts) if len(pts) >= MIN_VALID * max(1, (x1 - x0) * (y1 - y0)) else None
    if got is not None and got[2] < MIN_INLIERS:
        got = None
    rpts = ring_points(depth, K, box)
    ring = plane_fit(rpts) if len(rpts) >= MIN_RING else None
    use, source, npts = got, "box", len(pts)
    if got is None or (ring is not None and ring[2] >= MIN_INLIERS and _behind(got, ring, pts, K, box)):
        use, source, npts = ring, "ring", len(rpts)
    if use is None:
        return None
    g = _geometry(use, K, box)
    if g is None:
        return None
    c, n, w, h, frac = g
    return {"center": c.tolist(), "normal": n.tolist(), "width": w, "height": h, "source": source, "inliers": frac, "points": int(npts)}


def _behind(got, ring, pts, K, box):
    """True when the box's points lie > BEHIND beyond the ring plane along the box centre ray (see-through glass)."""
    r = _ray(K, (box["x_min"] + box["x_max"] + 1) / 2, (box["y_min"] + box["y_max"] + 1) / 2)
    hit = _hit(ring[0], ring[1], r)
    return hit is not None and float(np.median(pts[:, 2])) - hit[2] > BEHIND


def gt_doorway(meta, door):
    """Ground-truth doorway (not the swinging panel) in the frame's optical camera frame, from a door room v4 frame's meta:
    {"center", "normal" (toward the camera's side), "width", "height"}."""
    o = meta["objects"][door]
    a = o["approach"][meta["side"]]
    p = list(a["point"])
    p[2] = o["height"] / 2
    return {"center": frames.world_to_cam(p, meta["camera"], True), "normal": frames.world_to_cam(a["direction"], meta["camera"], False),
            "width": o["width"], "height": o["height"]}


def errors(est, gt):
    """Estimate vs ground truth: centre distance (m), normal angle (deg), width / height (est - gt, m), and how far the doorway is."""
    c, g = np.array(est["center"]), np.array(gt["center"])
    a, b = np.array(est["normal"]), np.array(gt["normal"])
    cos = float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
    return {"center_m": float(np.linalg.norm(c - g)), "normal_deg": math.degrees(math.acos(cos)),
            "width_m": est["width"] - gt["width"], "height_m": est["height"] - gt["height"], "distance_m": float(np.linalg.norm(g))}


def _iou(a, b):
    """IoU of two xyxy boxes (continuous coordinates)."""
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0, iw) * max(0, ih)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _xyxy(box):
    return [box["x_min"], box["y_min"], box["x_max"] + 1, box["y_max"] + 1]


def _from_xyxy(b):
    return {"x_min": int(math.floor(b[0])), "y_min": int(math.floor(b[1])), "x_max": int(math.ceil(b[2])) - 1, "y_max": int(math.ceil(b[3])) - 1}


def score_dataset(dataset_dir, boxes="gt", preds=None, iou=0.5, door_label=None):
    """One row per door per frame that has a 2D box: frame, door, side, state, glass, glass_depth_mode, visible_fraction,
    smoke_transmission (when present), "failed", and "errors" (None when failed). boxes="gt": the meta's boxes. preds
    ({frame: {"boxes" xyxy, "scores", "labels"}}): predicted boxes matched to GT doors by IoU >= `iou` (greedy, best score
    first, one box per door; `door_label` keeps only that label); a GT door with no match is failed with "missed": True."""
    root = pathlib.Path(dataset_dir)
    ds = json.loads((root / "dataset.json").read_text())
    K = ds["camera_intrinsics"]["K"]
    from PIL import Image
    rows = []
    for fdir in sorted((root / "frames").iterdir()):
        meta = json.loads((fdir / "meta.json").read_text())
        doors = {n: o for n, o in meta["objects"].items() if o.get("class") == 1 and o.get("box") and "approach" in o}
        if not doors:
            continue
        depth = np.asarray(Image.open(fdir / "depth.png"), dtype=np.float64) / 1000.0
        used = {}
        if preds is not None:
            p = preds.get(fdir.name, {"boxes": [], "scores": [], "labels": []})
            order = sorted((i for i in range(len(p["boxes"])) if door_label is None or p["labels"][i] == door_label), key=lambda i: -p["scores"][i])
            for i in order:
                cand = [(_iou(p["boxes"][i], _xyxy(o["box"])), n) for n, o in doors.items() if n not in used]
                if cand and max(cand)[0] >= iou:
                    used[max(cand)[1]] = _from_xyxy(p["boxes"][i])
        for name, o in doors.items():
            box = o["box"] if preds is None else used.get(name)
            row = {"frame": fdir.name, "door": name, "side": meta.get("side"), "state": o.get("state"), "glass": o.get("glass"),
                   "glass_depth_mode": meta.get("glass_depth_mode")}
            for k in ("visible_fraction", "smoke_transmission"):
                if k in o:
                    row[k] = o[k]
            est = fit(depth, K, box) if box else None
            row.update(failed=est is None, errors=None if est is None else errors(est, gt_doorway(meta, name)))
            if est:
                row["source"] = est["source"]
            if box is None:
                row["missed"] = True
            rows.append(row)
    return rows


def _stats(rows, key):
    v = np.array([r["errors"][key] for r in rows if r["errors"]])
    return f"{np.median(v):.3f} / {np.percentile(v, 90):.3f}" if len(v) else "-"


def summary(rows):
    """Short text table: median / 90th percentile centre (m) and normal (deg) error, overall and split."""
    groups = [("all", lambda r: True), ("glass", lambda r: r["glass"]), ("solid", lambda r: not r["glass"]),
              ("open", lambda r: r["state"] == "open"), ("closed", lambda r: r["state"] == "closed")]
    groups += [(f"{a}-{a + 1} m", lambda r, a=a: r["errors"] and a <= r["errors"]["distance_m"] < a + 1) for a in range(4)]
    out = [f"{'group':10} {'n':>5} {'failed':>7}  centre m (med / p90)  normal deg (med / p90)"]
    for name, f in groups:
        sel = [r for r in rows if f(r)]
        failed = sum(r["failed"] for r in sel)
        out.append(f"{name:10} {len(sel):5d} {failed / max(1, len(sel)):7.1%}  {_stats(sel, 'center_m'):>20}  {_stats(sel, 'normal_deg'):>21}")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--out", help="write the per-door rows here as JSON lines")
    a = ap.parse_args()
    rows = score_dataset(a.dataset)
    if a.out:
        pathlib.Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(summary(rows))
