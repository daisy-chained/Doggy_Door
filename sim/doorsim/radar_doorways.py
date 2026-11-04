"""Geometric doorway finder on radar points + scoring in metres (Madelief 2026-11-04).

Top-down map of the points in a height band, dominant wall line by RANSAC, gaps in the wall of door width (with wall on
both sides, so a wall end is no doorway). Scored against the doorway truth per distance and radar type; the centre error
is horizontal only (radar elevation is coarse).

usage:
  python3 sim/doorsim/radar_doorways.py DS [--models single imaging] [--split test] [--out runs/radar/<name>.jsonl]
         [--limit N] [--clean]
Reads per frame DS/frames/<f>/radar_depth.png, radar_seg.png, radar_depth_noglass.png and DS/radar/camera.json (written
by radar_render), runs doorsim.radar.simulate per model, finds + scores. --clean skips the radar and uses every 4th
pixel's world points of radar_depth.png directly (model "clean": the finder's upper bound / sanity check, no sensor).
Writes one JSON row per door (and per false positive) per model per frame; prints a summary.
"""
import argparse, json, pathlib, sys, zlib
import numpy as np

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))

Z_MIN, Z_MAX = 0.3, 2.2     # height band of the points used (m); radar elevation is coarse, so tolerant
MIN_W, MAX_W = 0.6, 1.3     # doorway width range (m)
MATCH_M = 0.5               # found doorway matches a GT door when the horizontal centre distance is below this
INLIER_TOL = 0.1            # RANSAC inlier distance to the wall line (m)
RANSAC_ITERS = 300
RANSAC_MAX_PTS = 3000       # fit on a subsample (inliers are counted on all points)
MIN_WALL_PTS = 25           # a wall needs at least this many inliers ...
MIN_WALL_LEN = 1.5          # ... and this length (m)
MAX_WALLS = 3               # walls tried per frame (the scenes have one door wall; the others rarely give gaps)
SUPPORT_M = 0.5             # wall support wanted within this distance on each side of a gap ...
MIN_SUPPORT_FRAC = 0.3      # ... = this fraction of the bins there must be wall bins (relative to the wall's own density)
BIN_M = 0.05                # bin size along the wall line for the density step (m); sparse walls use the median point spacing if larger
WALL_FRAC = 0.2             # a bin is wall when its count >= this fraction of the median count over occupied bins
DOOR_Z = 1.0                # z of a found doorway centre (guess: elevation is not measured well)


def _fit_line(xy, rng):
    """RANSAC dominant line -> (point, unit direction, inlier mask). Deterministic via rng."""
    sub = xy if len(xy) <= RANSAC_MAX_PTS else xy[rng.choice(len(xy), RANSAC_MAX_PTS, replace=False)]
    best, best_n = None, 0
    for _ in range(RANSAC_ITERS):
        i, j = rng.choice(len(sub), 2, replace=False)
        d = sub[j] - sub[i]
        L = np.hypot(*d)
        if L < 0.3:
            continue
        d = d / L
        n = int((np.abs((sub - sub[i]) @ np.array([-d[1], d[0]])) < INLIER_TOL).sum())
        if n > best_n:
            best, best_n = (sub[i], d), n
    if best is None:
        return None
    p, d = best
    nrm = np.array([-d[1], d[0]])
    mask = np.abs((xy - p) @ nrm) < INLIER_TOL
    if mask.sum() >= 2:  # refit by PCA on the inliers
        q = xy[mask]
        c = q.mean(0)
        d = np.linalg.svd(q - c, full_matrices=False)[2][0]
        p, nrm = c, np.array([-d[1], d[0]])
        mask = np.abs((xy - p) @ nrm) < INLIER_TOL
    return p, d, mask


def _density_gaps(t):
    """Gaps (a, b) along sorted wall positions t (m) found by density, not by the distance between neighbouring points.
    Histogram in bins of max(BIN_M, median point spacing) (so a sparse single-frame wall with 0.15 m spacing is not full of
    gaps). A bin is wall when its count >= WALL_FRAC * the median count over occupied bins (a few stray points inside a
    doorway in a dense stack stay below that; for a sparse wall, median ~1, any point counts). A gap is a run of non-wall bins
    between wall bins with width in [MIN_W, MAX_W]; it needs wall density on both sides: >= MIN_SUPPORT_FRAC of the bins
    within SUPPORT_M are wall bins (relative to the wall's own density, not a fixed point count)."""
    sp = np.diff(t)
    bw = max(BIN_M, float(np.median(sp)) if len(sp) else BIN_M)
    nb = int(np.ceil((t[-1] - t[0]) / bw)) + 1
    cnt = np.bincount(np.minimum(((t - t[0]) / bw).astype(int), nb - 1), minlength=nb)
    wall = cnt >= max(WALL_FRAC * np.median(cnt[cnt > 0]), 1e-9)
    sup = max(1, int(round(SUPPORT_M / bw)))
    idx = np.nonzero(wall)[0]
    out = []
    for i, j in zip(idx[:-1], idx[1:]):
        w = (j - i - 1) * bw
        if j - i < 2 or not (MIN_W <= w <= MAX_W):
            continue
        if wall[max(0, i - sup + 1):i + 1].mean() < MIN_SUPPORT_FRAC or wall[j:j + sup].mean() < MIN_SUPPORT_FRAC:
            continue
        out.append((t[0] + (i + 1) * bw, t[0] + j * bw))
    return out


def find(points, cam_pos):
    """Doorways in world points (N, 3) -> [{"center": [x, y, z], "width", "normal": [nx, ny, 0]}], normal towards cam_pos.
    z of the centre is DOOR_Z (not measured). Points behind a gap are not wall inliers, so they cannot close it."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(1)]
    pts = pts[(pts[:, 2] >= Z_MIN) & (pts[:, 2] <= Z_MAX)]
    xy = pts[:, :2]
    cam = np.asarray(cam_pos, dtype=float)
    rng = np.random.default_rng(0)
    out = []
    for _ in range(MAX_WALLS):
        if len(xy) < MIN_WALL_PTS:
            break
        fit = _fit_line(xy, rng)
        if fit is None:
            break
        p, d, mask = fit
        if mask.sum() < MIN_WALL_PTS:
            break
        t = (xy[mask] - p) @ d
        t.sort()
        if t[-1] - t[0] >= MIN_WALL_LEN:
            nrm = np.array([-d[1], d[0]])
            for a, b in _density_gaps(t):
                c = p + d * (a + b) / 2
                n = nrm if (cam[:2] - c) @ nrm >= 0 else -nrm
                out.append({"center": [float(c[0]), float(c[1]), DOOR_Z], "width": float(b - a), "normal": [float(n[0]), float(n[1]), 0.0]})
        xy = xy[~mask]
    return out


def ground_truth(meta):
    """{door: {"center": [x, y, height/2], "width", "normal", "state"}} of the class-1 objects with an approach on meta["side"]."""
    gt = {}
    for name, o in meta.get("objects", {}).items():
        a = (o.get("approach") or {}).get(meta.get("side"))
        if o.get("class") != 1 or not a:
            continue
        p = a["point"]
        gt[name] = {"center": [p[0], p[1], o["height"] / 2], "width": o["width"], "normal": list(a["direction"]), "state": o.get("state")}
    return gt


def score(found, gt, cam_pos):
    """Greedy one-to-one nearest matching (horizontal centre distance < MATCH_M). Rows per GT door
    {"door", "found", "center_m", "width_m" (est - gt), "distance_m" (cam to door, horizontal), "state"}, then
    {"door": None, "false_positive": True, ...} per unmatched found doorway."""
    cam = np.asarray(cam_pos, dtype=float)[:2]
    pairs = sorted((float(np.hypot(*(np.array(f["center"][:2]) - np.array(g["center"][:2])))), i, name)
                   for i, f in enumerate(found) for name, g in gt.items())
    match, used = {}, set()
    for dist, i, name in pairs:
        if dist < MATCH_M and i not in used and name not in match:
            match[name] = (i, dist)
            used.add(i)
    rows = []
    for name, g in gt.items():
        row = {"door": name, "found": name in match, "center_m": None, "width_m": None,
               "distance_m": float(np.hypot(*(np.array(g["center"][:2]) - cam))), "state": g["state"]}
        if name in match:
            i, dist = match[name]
            row["center_m"], row["width_m"] = dist, float(found[i]["width"] - g["width"])
        rows.append(row)
    for i, f in enumerate(found):
        if i not in used:
            rows.append({"door": None, "false_positive": True, "found": True, "center": f["center"], "width": f["width"]})
    return rows


OCCLUDE_M = 0.3             # a doorway is hidden when the patch depth at it is more than this nearer than the doorway
PATCH_PX = 3                # half-size of the depth patch around the projected doorway centre (px), capped by the projected half-width


def in_view(door_gt, camera, K, depth):
    """Could the radar see this doorway: its centre projects inside the image (z > 0) and the depth patch around it is no
    return (see-through) or at least z - OCCLUDE_M (the doorway itself or the room behind), not something in front."""
    from doorsim.frames import world_to_cam
    x, y, z = world_to_cam(door_gt["center"], camera, True)
    if z <= 0:
        return False
    depth = np.asarray(depth, dtype=float)
    H, W = depth.shape
    u, v = K[0] * x / z + K[2], K[4] * y / z + K[5]
    if not (0 <= u < W and 0 <= v < H):
        return False
    r = int(max(1, min(PATCH_PX, K[0] * door_gt["width"] / 2 / z)))
    ui, vi = int(u), int(v)
    patch = depth[max(0, vi - r):vi + r + 1, max(0, ui - r):ui + r + 1]
    patch = np.where(np.isfinite(patch), patch, 0.0)
    med = float(np.median(patch))
    return med <= 0 or med >= z - OCCLUDE_M


def _mark_in_view(rows, gt, camera, K, depth):
    for r in rows:
        if r["door"] is not None:
            r["in_view"] = bool(in_view(gt[r["door"]], camera, K, depth))


def view_windows(metas, ns):
    """{n: [(target_frame, [frames of the n views ending at the target, in view order])]}. Groups = (world, scene, side):
    a robot looking around stays in its own room (a scene has views from both sides of the wall; mixing them would put the
    other room's walls in the doorway). Views ordered by meta["view"]. Targets = frames with at least max(ns) - 1 earlier
    views in their group, so every n is scored on exactly the same target frames."""
    scenes = {}
    for f, m in metas.items():
        scenes.setdefault((m.get("world"), m.get("scene"), m.get("side")), []).append(f)
    need = max(ns) - 1
    out = {n: [] for n in ns}
    for key in sorted(scenes, key=str):
        fr = sorted(scenes[key], key=lambda f: (metas[f]["view"], f))
        for i in range(need, len(fr)):
            for n in ns:
                out[n].append((fr[i], fr[i - n + 1:i + 1]))
    return out


# --- CLI ---
BINS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 1e9)]


def _bin_name(lo, hi):
    return f"{lo}-{hi} m" if hi < 1e9 else f"{lo}+ m"


def _stats(rows):
    doors = [r for r in rows if r["door"] is not None]
    if not doors:
        return "no doors"
    e = np.array([r["center_m"] for r in doors if r["found"]])
    s = f"n={len(doors)} found={100 * len(e) / len(doors):.0f}%"
    if len(e):
        s += f" centre err median={np.median(e):.2f} p90={np.percentile(e, 90):.2f} m"
    return s


def summarize(rows_by_model, n_frames_by_model):
    for model, rows in rows_by_model.items():
        fp = sum(1 for r in rows if r["door"] is None)
        print(f"\n== {model}: {_stats(rows)}; false positives/frame={fp / max(1, n_frames_by_model[model]):.2f}")
        iv = [r for r in rows if r["door"] is not None and r.get("in_view")]
        ivo = [r for r in iv if str(r["state"]) == "open"]
        def _nf(x):
            return f"n={len(x)} found={100 * sum(1 for r in x if r['found']) / len(x):.0f}%" if x else "n=0 found=-"
        print(f"   in view: {_nf(iv)}  (open: {_nf(ivo)})")
        for lo, hi in BINS:
            sel = [r for r in rows if r["door"] is not None and lo <= r["distance_m"] < hi]
            if sel:
                print(f"   {_bin_name(lo, hi):>8}: {_stats(sel)}")
        for st in sorted({str(r["state"]) for r in rows if r["door"] is not None}):
            sel = [r for r in rows if r["door"] is not None and str(r["state"]) == st]
            print(f"   state {st:>7}: {_stats(sel)}")


def view_table(rows_by_model, models, ns, in_view_only=False):
    """Compact table: rows = model, columns = views, cells = open-door found rate / median centre error."""
    def cell(rows):
        doors = [r for r in rows if r["door"] is not None and str(r["state"]) == "open" and (not in_view_only or r.get("in_view"))]
        if not doors:
            return "-"
        e = [r["center_m"] for r in doors if r["found"]]
        return f"{100 * len(e) / len(doors):.0f}% / " + (f"{np.median(e):.2f} m" if e else "-") + f" (n={len(doors)})"
    print("\nOpen doors in view: found rate / median centre error, by number of views" if in_view_only
          else "\nOpen doors: found rate / median centre error, by number of views")
    w = 24
    print(f"{'model':<10}" + "".join(f"{n:>{w}}" for n in ns))
    for m in models:
        print(f"{m:<10}" + "".join(f"{cell(rows_by_model.get(f'{m}@{n}', [])):>{w}}" for n in ns))


def _camera_K(root):
    k = json.loads((root / "radar" / "camera.json").read_text())
    return k["K"] if isinstance(k, dict) else k


def _run_views(args, ds, root, metas, names, K, models, radar, mats_cache):
    from PIL import Image
    from doorsim import effects
    ns = sorted(set(args.views))
    windows = view_windows({n: metas[n] for n in names}, ns)
    cache = {}  # (frame, model) -> points; each frame's points are computed once per model

    def points(name, m):
        if (name, m) in cache:
            return cache[name, m]
        meta, fd = metas[name], root / "frames" / name
        cam = meta["camera"]
        depth = effects.depth_m(fd / "radar_depth.png")
        if m == "clean":
            pts = effects.world_points(depth, K, cam)[::4, ::4].reshape(-1, 3)
            pts = pts[np.isfinite(pts).all(1)]
        else:
            seg = np.asarray(Image.open(fd / "radar_seg.png"))
            noglass = effects.depth_m(fd / "radar_depth_noglass.png")
            wf = effects.world_file(ds, meta)
            if wf not in mats_cache:
                mats_cache[wf] = radar.materials(json.loads((wf.parent / f"{wf.stem}.attributes.json").read_text()))
            rng = np.random.default_rng(zlib.crc32(f"{name}".encode()))
            res = radar.simulate(depth, seg, noglass, K, cam, radar.MODELS[m], mats_cache[wf], rng)
            pts = res["points"] if isinstance(res, dict) else res
        cache[name, m] = pts
        return pts

    def rendered(name):
        return (root / "frames" / name / "radar_depth_noglass.png").exists()

    keys = [f"{m}@{n}" for m in models for n in ns]
    rows_by_model = {k: [] for k in keys}
    nfr = {k: 0 for k in keys}
    out = None
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out = open(args.out, "w")
    targets = [t for t, _ in windows[ns[0]]]
    used = 0
    for k, target in enumerate(targets):
        if not all(rendered(f) for n in ns for tt, fr in [windows[n][k]] for f in fr):
            continue  # a window frame is not rendered (yet): skip the target for every n, so all n stay on the same targets
        used += 1
        cam_pos = metas[target]["camera"]["world_position"]
        gt = ground_truth(metas[target])
        tdepth = effects.depth_m(root / "frames" / target / "radar_depth.png")
        for m in models:
            for n in ns:
                fr = windows[n][k][1]
                pts = np.vstack([points(f, m) for f in fr])
                rows = score(find(pts, cam_pos), gt, cam_pos)
                _mark_in_view(rows, gt, metas[target]["camera"], K, tdepth)
                key = f"{m}@{n}"
                nfr[key] += 1
                rows_by_model[key] += rows
                if out:
                    for r in rows:
                        out.write(json.dumps({"frame": target, "model": m, "views": n, **r}) + "\n")
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(targets)} targets", flush=True)
    if out:
        out.close()
    print(f"{used}/{len(targets)} target windows complete (all views rendered)")
    summarize(rows_by_model, nfr)
    view_table(rows_by_model, models, ns)
    view_table(rows_by_model, models, ns, in_view_only=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ds")
    ap.add_argument("--models", nargs="+", default=None, help="default single imaging (none with --clean alone)")
    ap.add_argument("--split", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--clean", action="store_true", help="also/only the clean upper bound: every 4th pixel's world points, no radar")
    ap.add_argument("--export-seed", type=int, default=0)
    ap.add_argument("--views", nargs="+", type=int, default=None,
                    help="also stack the points of the last n views of the same scene (world coordinates), e.g. 1 3 10; "
                         "all n are scored on the same target frames; models are reported as name@n")
    args = ap.parse_args()

    from PIL import Image
    from doorsim import effects
    root = pathlib.Path(args.ds)
    ds = json.loads((root / "dataset.json").read_text())
    metas = {d.name: json.loads((d / "meta.json").read_text()) for d in sorted((root / "frames").iterdir()) if (d / "meta.json").exists()}
    metas, _ = effects.split_frames(ds, metas, args.split, (0.7, 0.15, 0.15), args.export_seed)
    names = sorted(metas)[:args.limit]
    K = _camera_K(root)
    models = list(args.models) if args.models else ([] if args.clean else ["single", "imaging"])
    radar = None
    if models:
        from doorsim import radar
    if args.clean:
        models.append("clean")
    mats_cache = {}
    if args.views:
        _run_views(args, ds, root, metas, names, K, models, radar, mats_cache)
        return
    rows_by_model = {m: [] for m in models}
    nfr = {m: 0 for m in models}
    out = None
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out = open(args.out, "w")
    for k, name in enumerate(names):
        meta, fd = metas[name], root / "frames" / name
        if not (fd / "radar_depth_noglass.png").exists():  # not rendered (yet)
            continue
        gt = ground_truth(meta)
        cam = meta["camera"]
        depth = effects.depth_m(fd / "radar_depth.png")
        if radar is not None:
            seg = np.asarray(Image.open(fd / "radar_seg.png"))
            noglass = effects.depth_m(fd / "radar_depth_noglass.png")
            wf = effects.world_file(ds, meta)
            if wf not in mats_cache:
                mats_cache[wf] = radar.materials(json.loads((wf.parent / f"{wf.stem}.attributes.json").read_text()))
        for m in models:
            if m == "clean":
                pts = effects.world_points(depth, K, cam)[::4, ::4].reshape(-1, 3)
                pts = pts[np.isfinite(pts).all(1)]
            else:
                rng = np.random.default_rng(zlib.crc32(f"{name}".encode()))
                res = radar.simulate(depth, seg, noglass, K, cam, radar.MODELS[m], mats_cache[wf], rng)
                pts = res["points"] if isinstance(res, dict) else res
            rows = score(find(pts, cam["world_position"]), gt, cam["world_position"])
            _mark_in_view(rows, gt, cam, K, depth)
            nfr[m] += 1
            rows_by_model[m] += rows
            if out:
                for r in rows:
                    out.write(json.dumps({"frame": name, "model": m, **r}) + "\n")
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(names)} frames", flush=True)
    if out:
        out.close()
    summarize(rows_by_model, nfr)


if __name__ == "__main__":
    main()
