"""Patrol (run 3 "no markers", Madelief 2026-11-11): the robot gets the floor map but no door positions; it patrols every
reachable space, looks around, turns detector "door" boxes into door candidates on the map and runs the door step on each.
Technical rules = Claude decisions, dev log 2026-11-11 reactive door step. Pure helpers, numpy + math only."""
import math
from collections import deque
import numpy as np

from . import hinge_look


def _cell(m, x, y):
    """(row, col) of the world point (x, y); may lie outside the map."""
    h = m["occ"].shape[0]
    return h - 1 - int(math.floor((y - m["origin"][1]) / m["res"])), int(math.floor((x - m["origin"][0]) / m["res"]))


def reachable(m, start, radius):
    """bool HxW: cells where the robot fits (free square of half ceil(radius / res), inside the map) and that connect to start (4-neighbour)."""
    occ = m["occ"]
    h, w = occ.shape
    half = int(math.ceil(radius / m["res"]))
    k = 2 * half + 1
    out = np.zeros((h, w), bool)
    if h < k or w < k:
        return out
    ii = np.zeros((h + 1, w + 1), np.int32)
    ii[1:, 1:] = occ.astype(np.int32).cumsum(0).cumsum(1)
    cnt = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]      # occupied cells in each k x k window
    fit = np.zeros((h, w), bool)
    fit[half:h - half, half:w - half] = cnt == 0
    r0, c0 = _cell(m, *start)
    if not (0 <= r0 < h and 0 <= c0 < w) or not fit[r0, c0]:
        return out
    f = fit.tolist()
    seen = [[False] * w for _ in range(h)]
    seen[r0][c0] = True
    q = deque([(r0, c0)])
    while q:
        r, c = q.popleft()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < h and 0 <= cc < w and f[rr][cc] and not seen[rr][cc]:
                seen[rr][cc] = True
                q.append((rr, cc))
    return np.array(seen, bool)


def viewpoints(m, start, radius, spacing=1.5, pass_radius=None, anchor=None):
    """Grid points (x0 + spacing * (i + 0.5), y0 + spacing * (j + 0.5)) on reachable cells (flood fill with pass_radius, default
    radius) that are free with radius (room to turn), ordered greedily nearest-first from start. anchor = (ax, ay): the grid is
    fixed in the world instead, x = ax + spacing * i, y = ay + spacing * j for any integers i, j (a growing SLAM map)."""
    reach = reachable(m, start, radius if pass_radius is None else pass_radius)
    h, w = reach.shape
    x0, y0 = m["origin"]
    if anchor is None:
        js, is_, ox, oy = range(int(h * m["res"] / spacing) + 1), range(int(w * m["res"] / spacing) + 1), x0 + spacing * 0.5, y0 + spacing * 0.5
    else:
        ox, oy = anchor
        js = range(int(math.ceil((y0 - oy) / spacing)), int(math.floor((y0 + h * m["res"] - oy) / spacing)) + 1)
        is_ = range(int(math.ceil((x0 - ox) / spacing)), int(math.floor((x0 + w * m["res"] - ox) / spacing)) + 1)
    pts = []
    for j in js:
        y = oy + spacing * j if anchor is not None else y0 + spacing * (j + 0.5)
        for i in is_:
            x = ox + spacing * i if anchor is not None else x0 + spacing * (i + 0.5)
            r, c = _cell(m, x, y)
            if 0 <= r < h and 0 <= c < w and reach[r, c] and (pass_radius is None or hinge_look.free(m, x, y, radius)):
                pts.append((x, y))
    out, cur = [], tuple(start)
    while pts:
        k = min(range(len(pts)), key=lambda i: math.hypot(pts[i][0] - cur[0], pts[i][1] - cur[1]))
        cur = pts.pop(k)
        out.append(cur)
    return out


def _box(a, n):
    """Horizontal dilation of bool array a by n cells each side (sliding window via cumsum)."""
    h, w = a.shape
    cs = np.zeros((h, w + 1), np.int32)
    cs[:, 1:] = a.astype(np.int32).cumsum(1)
    lo, hi = np.clip(np.arange(w) - n, 0, w), np.clip(np.arange(w) + n + 1, 0, w)
    return cs[:, hi] - cs[:, lo] > 0


def frontiers(m, start, pass_radius, radius, min_len=0.5, reach_out=1.0, sectors=1, min_dist=0.0):
    """[{"x", "y", "size"}] sorted nearest-first from start. Frontier cells: known-free cells 4-connected to start through known-free
    cells with an unknown cell among their 8 neighbours; clusters = 8-connected groups, dropped when under min_len / res cells. Goal per
    cluster: of the reachable(pass_radius) cells that are free(radius) and within reach_out m of a cluster cell, the one nearest the
    cluster centroid (none -> skipped). size = cells * res (m). sectors > 1: every cluster is first split by direction from start
    (sector of a cell centre = floor((atan2(dy, dx) mod 2pi) / (2pi / sectors))) into pieces = 8-connected groups of its cells in
    one sector; each piece then goes through min_len / goal / size like a cluster. Goals under min_dist m (straight line) from
    start are dropped. Defaults give the old output."""
    occ, unk, res = m["occ"], m["unknown"], m["res"]
    h, w = occ.shape
    r0, c0 = _cell(m, *start)
    if not (0 <= r0 < h and 0 <= c0 < w) or occ[r0, c0]:
        return []
    f = (~occ).tolist()
    seen = [[False] * w for _ in range(h)]
    seen[r0][c0] = True
    q = deque([(r0, c0)])
    while q:
        r, c = q.popleft()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < h and 0 <= cc < w and f[rr][cc] and not seen[rr][cc]:
                seen[rr][cc] = True
                q.append((rr, cc))
    conn = np.array(seen, bool)
    up = np.pad(unk, 1)
    near = np.zeros((h, w), bool)
    for dr in range(3):
        for dc in range(3):
            if (dr, dc) != (1, 1):
                near |= up[dr:dr + h, dc:dc + w]
    fr = conn & near
    if not fr.any():
        return []
    reach = reachable(m, start, pass_radius)
    half = int(math.ceil(radius / res))
    k = 2 * half + 1
    fit = np.zeros((h, w), bool)
    if h >= k and w >= k:
        ii = np.zeros((h + 1, w + 1), np.int32)
        ii[1:, 1:] = occ.astype(np.int32).cumsum(0).cumsum(1)
        fit[half:h - half, half:w - half] = (ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]) == 0
    cand = reach & fit
    n = int(round(reach_out / res))
    cl = deque()
    left = fr.copy()
    for r, c in zip(*np.nonzero(fr)):
        if not left[r, c]:
            continue
        left[r, c] = False
        cells, q = [(r, c)], deque([(r, c)])
        while q:
            a, b = q.popleft()
            for rr in range(max(a - 1, 0), min(a + 2, h)):
                for cc in range(max(b - 1, 0), min(b + 2, w)):
                    if left[rr, cc]:
                        left[rr, cc] = False
                        cells.append((rr, cc))
                        q.append((rr, cc))
        cl.append(cells)
    if sectors > 1:
        pieces = deque()
        for cells in cl:
            sec = {}
            for a, b in cells:
                ang = math.atan2(m["origin"][1] + (h - 1 - a + 0.5) * res - start[1], m["origin"][0] + (b + 0.5) * res - start[0])
                sec[(a, b)] = int(math.floor((ang % (2 * math.pi)) / (2 * math.pi / sectors))) % sectors
            left_p = set(cells)
            for cell in cells:
                if cell not in left_p:
                    continue
                left_p.discard(cell)
                grp, q = [cell], deque([cell])
                while q:
                    a, b = q.popleft()
                    for rr in range(a - 1, a + 2):
                        for cc in range(b - 1, b + 2):
                            if (rr, cc) in left_p and sec[(rr, cc)] == sec[cell]:
                                left_p.discard((rr, cc))
                                grp.append((rr, cc))
                                q.append((rr, cc))
                pieces.append(grp)
        cl = pieces
    out = []
    for cells in cl:
        if len(cells) < min_len / res:
            continue
        rs, cs_ = np.array([p[0] for p in cells]), np.array([p[1] for p in cells])
        ra, rb, ca, cb = max(rs.min() - n, 0), min(rs.max() + n + 1, h), max(cs_.min() - n, 0), min(cs_.max() + n + 1, w)
        sub = np.zeros((rb - ra, cb - ca), bool)
        sub[rs - ra, cs_ - ca] = True
        hd, near_c = {}, np.zeros_like(sub)
        for dy in range(-n, n + 1):
            dx = int(math.floor(math.sqrt(max(n * n - dy * dy, 0)) + 1e-9))
            if dx not in hd:
                hd[dx] = _box(sub, dx)
            s = hd[dx]
            if dy > 0:
                s = np.vstack((np.zeros((dy, s.shape[1]), bool), s[:-dy]))
            elif dy < 0:
                s = np.vstack((s[-dy:], np.zeros((-dy, s.shape[1]), bool)))
            near_c |= s
        ok = near_c & cand[ra:rb, ca:cb]
        gr, gc = np.nonzero(ok)
        if not gr.size:
            continue
        d = (gr + ra - rs.mean()) ** 2 + (gc + ca - cs_.mean()) ** 2
        i = int(np.argmin(d))
        r, c = int(gr[i] + ra), int(gc[i] + ca)
        g = {"x": m["origin"][0] + (c + 0.5) * res, "y": m["origin"][1] + (h - 1 - r + 0.5) * res, "size": len(cells) * res}
        if min_dist > 0 and math.hypot(g["x"] - start[0], g["y"] - start[1]) < min_dist:
            continue
        out.append(g)
    return sorted(out, key=lambda g: math.hypot(g["x"] - start[0], g["y"] - start[1]))


def view_yaws(n=5, start_yaw=0.0):
    """n yaws evenly around the circle (HFOV 74.5 deg: 5 views cover 360)."""
    return [start_yaw + k * 2 * math.pi / n for k in range(n)]


def _med_valid(depth, ra, rb, c0, c1):
    """Valid depth pixels (finite, > 0) in rows ra:rb, columns c0..c1 inclusive (clipped to the image)."""
    a = depth[ra:rb, max(0, c0):min(depth.shape[1], c1 + 1)]
    return a[np.isfinite(a) & (a > 0)]


def _wall_median(box, depth, strip=6):
    """Median valid depth in the two strips just outside the box, over the middle half of its rows (the wall); None when there is none."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    ra, rb = max(0, y1 + (y2 - y1) // 4), min(depth.shape[0], y1 + 3 * (y2 - y1) // 4 + 1)
    v = np.concatenate((_med_valid(depth, ra, rb, x1 - strip, x1), _med_valid(depth, ra, rb, x2, x2 + strip)))
    return float(np.median(v)) if v.size else None


def door_estimate(box, depth, cam, fx, cx, strip=6):
    """{"x", "y", "width", "depth"} of a door box. Depth = median of the valid depth in strips just outside the box (the wall, not the
    room behind an open door); else inside the box; else None."""
    d = _wall_median(box, depth, strip)
    if d is None:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        ra, rb = max(0, y1 + (y2 - y1) // 4), min(depth.shape[0], y1 + 3 * (y2 - y1) // 4 + 1)
        v = _med_valid(depth, ra, rb, x1, x2)
        if not v.size:
            return None
        d = float(np.median(v))
    u = (box[0] + box[2]) / 2
    left = d * (cx - u) / fx
    camx, camy, yaw = cam
    return {"x": camx + d * math.cos(yaw) - left * math.sin(yaw), "y": camy + d * math.sin(yaw) + left * math.cos(yaw),
            "width": (box[2] - box[0]) * d / fx, "depth": d}


def snap_to_wall(m, x, y, cam_xy, search=0.4):
    """Walls run along x or y. Per direction look at the lines 0.6..1.0 m to both sides of (x, y) along it, for occupied cells within
    +-search along the normal; both sides occupied -> snap to their mean normal coordinate. Returns (x, y, yaw towards cam_xy) or None."""
    occ, res = m["occ"], m["res"]
    h, w = occ.shape
    x0, y0 = m["origin"]
    offs = np.arange(0.6, 1.0 + 1e-9, res)
    steps = np.arange(-search, search + 1e-9, res)
    best = None
    for axis in (0, 1):                      # 0: wall along x (normal y); 1: wall along y (normal x)
        along, normal = (x, y) if axis == 0 else (y, x)
        sides = []
        for sgn in (-1, 1):
            hits = []
            for o in offs:
                for s in steps:
                    px, py = (along + sgn * o, normal + s) if axis == 0 else (normal + s, along + sgn * o)
                    r, c = _cell(m, px, py)
                    if 0 <= r < h and 0 <= c < w and occ[r, c]:
                        hits.append(normal + s)
            sides.append(hits)
        if sides[0] and sides[1]:
            mean = float(np.mean(sides[0] + sides[1]))
            if best is None or abs(mean - normal) < best[0]:
                best = (abs(mean - normal), axis, mean)
    if best is None:
        return None
    _, axis, mean = best
    if axis == 0:
        return x, mean, (math.pi / 2 if cam_xy[1] >= mean else -math.pi / 2)
    return mean, y, (0.0 if cam_xy[0] >= mean else math.pi)


def snap_to_gap(m, x, y, yaw, search=1.0, gap=(0.3, 1.4)):
    """gap_centre(...) if there is one, else (x, y)."""
    return gap_centre(m, x, y, yaw, search, gap) or (x, y)


def gap_centre(m, x, y, yaw, search=1.0, gap=(0.3, 1.4), line_search=0.3):
    """The wall line first: of the lines parallel to the wall within +-line_search, the one with the most occupied cells over the sampled
    stretch (ties: nearest); then _gap_on_line there. Returns (x, y) on that line, or None."""
    occ, res = m["occ"], m["res"]
    h, w = occ.shape
    along_x = abs(math.cos(yaw)) < 0.5
    k = int(round((search + gap[1]) / res))
    best_o, best_n = 0.0, -1
    for j in sorted(range(-int(round(line_search / res)), int(round(line_search / res)) + 1), key=abs):
        o = j * res
        n = 0
        for i in range(-k, k + 1):
            px, py = (x + i * res, y + o) if along_x else (x + o, y + i * res)
            r, c = _cell(m, px, py)
            n += (not (0 <= r < h and 0 <= c < w)) or bool(occ[r, c])
        if n > best_n:
            best_o, best_n = o, n
    lx, ly = (x, y + best_o) if along_x else (x + best_o, y)
    return _gap_on_line(m, lx, ly, yaw, search, gap)


def _gap_on_line(m, x, y, yaw, search=1.0, gap=(0.3, 1.4)):
    """Open doors are gaps in the wall on the map. The wall line through (x, y) runs across yaw (a multiple of pi / 2). Sample it every res
    over +-(search + gap[1]); a sample is open when all cells within +-0.15 m across the wall are free (outside the map = occupied). Gaps =
    runs of open samples not touching either end, with length within gap and centre within search of (x, y). Returns the nearest centre, else (x, y)."""
    occ, res = m["occ"], m["res"]
    h, w = occ.shape
    along_x = abs(math.cos(yaw)) < 0.5       # normal along y -> wall along x
    k = int(round((search + gap[1]) / res))
    ks = np.arange(-k, k + 1)
    half = int(round(0.15 / res))
    across = np.arange(-half, half + 1) * res
    is_open = []
    for i in ks:
        free = True
        for a in across:
            px, py = (x + i * res, y + a) if along_x else (x + a, y + i * res)
            r, c = _cell(m, px, py)
            if not (0 <= r < h and 0 <= c < w) or occ[r, c]:
                free = False
                break
        is_open.append(free)
    best = None
    n = len(ks)
    i = 0
    while i < n:
        if not is_open[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and is_open[j + 1]:
            j += 1
        length = (j - i + 1) * res
        if i > 0 and j < n - 1 and gap[0] - 1e-9 <= length <= gap[1] + 1e-9:
            centre = (ks[i] + ks[j]) / 2 * res
            if abs(centre) <= search + 1e-9 and (best is None or abs(centre) < abs(best)):
                best = centre
        i = j + 1
    if best is None:
        return None
    return (x + best, y) if along_x else (x, y + best)


def _gap_runs(b, res, gap, nw, d):
    """Runs on the lines (rows) of the bool array b (True = occupied; outside = occupied): free runs of length within gap with at least nw
    occupied cells beyond both ends and free cells d lines above and below every run cell. Yields (line, first pos, last pos)."""
    pad = nw + d
    p = np.pad(b, pad, constant_values=True)
    free = ~p
    f = np.zeros((free.shape[0], free.shape[1] + 2), np.int8)
    f[:, 1:-1] = free
    dif = np.diff(f, axis=1)
    ls, s = np.nonzero(dif == 1)
    le, e = np.nonzero(dif == -1)                       # same order: runs are [s, e) on line ls
    n = e - s
    keep = (n * res >= gap[0] - 1e-9) & (n * res <= gap[1] + 1e-9) & (ls >= pad) & (ls < p.shape[0] - pad)
    for line, a, z in zip(ls[keep], s[keep], e[keep]):
        if p[line, a - nw:a].all() and p[line, z:z + nw].all() and not p[line - d, a:z].any() and not p[line + d, a:z].any():
            yield int(line) - pad, int(a) - pad, int(z) - 1 - pad


def _slit(b, g, nl, d):
    """True when, with the runs of g [(.., .., .., line, first, last)] blocked, the free cells d lines either side of the centre connect
    (4-neighbour) inside a +-nl cell window: not the only way between the two sides."""
    mean_line = sum(r[3] for r in g) / len(g)
    r0 = min(g, key=lambda r: abs(r[3] - mean_line))
    cl, cp = r0[3], (r0[4] + r0[5]) // 2
    pad = nl + d + 1
    win = np.pad(b, pad, constant_values=True)[cl + pad - nl:cl + pad + nl + 1, cp + pad - nl:cp + pad + nl + 1].copy()
    for r in g:
        win[r[3] - cl + nl, max(r[4] - cp + nl, 0):max(r[5] - cp + nl + 1, 0)] = True
    free = ~win
    start, goal = (nl - d, nl), (nl + d, nl)
    if not (free[start] and free[goal]):
        return False
    seen = np.zeros_like(free)
    seen[start] = True
    q = deque([start])
    hh, ww = free.shape
    while q:
        i, j = q.popleft()
        if (i, j) == goal:
            return True
        for a, c in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < hh and 0 <= c < ww and free[a, c] and not seen[a, c]:
                seen[a, c] = True
                q.append((a, c))
    return False


def map_gaps(m, gap=(0.3, 1.4), wall=0.3, clear=0.4, local=2.5):
    """Doorways on the map: free runs of length within gap along a map row (axis "x") or column (axis "y") with wall (>= wall m occupied) on
    both ends and open floor at +-clear across the line. Runs on neighbouring lines whose centres are within 0.3 m are merged (mean centre,
    width of the longest run). A doorway must be the only way between its two sides nearby: with the merged runs blocked, the free cells at
    +-clear across the centre must not connect (4-neighbour) inside a +-local m window (else a slit between furniture). Returns [{"x", "y", "axis", "width"}] sorted by (axis, x, y)."""
    occ, res = m["occ"], m["res"]
    h = occ.shape[0]
    x0, y0 = m["origin"]
    nw, d = max(1, int(round(wall / res))), int(round(clear / res))
    nl = int(round(local / res))
    out = []
    for axis, b in (("x", occ), ("y", occ.T)):
        runs = []                                        # (cx, cy, width)
        for line, a, z in _gap_runs(b, res, gap, nw, d):
            width = (z - a + 1) * res
            mid = (a + z) / 2 + 0.5
            if axis == "x":
                runs.append((x0 + mid * res, y0 + (h - 1 - line + 0.5) * res, width, line, a, z))
            else:
                runs.append((x0 + (line + 0.5) * res, y0 + (h - mid) * res, width, line, a, z))
        parent = list(range(len(runs)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i in range(len(runs)):
            for j in range(i):
                if math.hypot(runs[i][0] - runs[j][0], runs[i][1] - runs[j][1]) <= 0.3 + 1e-9:
                    parent[find(i)] = find(j)
        groups = {}
        for i, r in enumerate(runs):
            groups.setdefault(find(i), []).append(r)
        for g in groups.values():
            if _slit(b, g, nl, d):
                continue
            out.append({"x": sum(r[0] for r in g) / len(g), "y": sum(r[1] for r in g) / len(g), "axis": axis, "width": max(r[2] for r in g)})
    return sorted(out, key=lambda g: (g["axis"], g["x"], g["y"]))


def _adiff(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def add_candidate(cands, est, tol=0.6):
    """Merge est ({"x", "y", "yaw", "width"}) into the nearest candidate within tol and normal within 45 deg (running mean of x, y, width;
    "n" counts merges), else append it with n = 1. Returns the index."""
    best, best_d = None, None
    for i, c in enumerate(cands):
        d = math.hypot(c["x"] - est["x"], c["y"] - est["y"])
        if d <= tol and _adiff(c["yaw"], est["yaw"]) < math.pi / 4 and (best_d is None or d < best_d):
            best, best_d = i, d
    if best is None:
        cands.append({**est, "n": 1})
        return len(cands) - 1
    c = cands[best]
    n = c["n"]
    for k in ("x", "y", "width"):
        c[k] = (c[k] * n + est[k]) / (n + 1)
    c["n"] = n + 1
    return best


def match(est, doors, tol=0.5):
    """(door id, "front" | "back") of the real door nearest to est within tol, else None. The door yaw points front -> back; est yaw
    points towards the robot, so cos(est yaw - door yaw) < 0 -> "front"."""
    best, best_d = None, None
    for d in doors:
        dist = math.hypot(d["pos"][0] - est["x"], d["pos"][1] - est["y"])
        if dist <= tol and (best_d is None or dist < best_d):
            best, best_d = d, dist
    if best is None:
        return None
    return best["id"], ("front" if math.cos(est["yaw"] - best["yaw"]) < 0 else "back")


def approach(est, dists=(2.0, 1.0)):
    """Poses on the normal, dist out from (x, y), facing the door."""
    return [{"x": est["x"] + d * math.cos(est["yaw"]), "y": est["y"] + d * math.sin(est["yaw"]),
             "yaw": est["yaw"] + math.pi, "dist": d} for d in dists]


def near_opened(est, opened, margin=0.4):
    """True when est ({"x", "y", "yaw"}) is on the same wall as a door the robot opened (opened = [{"x", "y", "yaw", "width"}]): normals
    parallel (|cos| > 0.7), across-wall offset < 0.3 m, along-wall distance <= 1.5 * width + margin (the panel lies beside the doorway)."""
    for o in opened:
        if abs(math.cos(est["yaw"] - o["yaw"])) <= 0.7:
            continue
        dx, dy = est["x"] - o["x"], est["y"] - o["y"]
        across = dx * math.cos(o["yaw"]) + dy * math.sin(o["yaw"])
        along = -dx * math.sin(o["yaw"]) + dy * math.cos(o["yaw"])
        if abs(across) < 0.3 and abs(along) <= 1.5 * o["width"] + margin:
            return True
    return False


def spans_frame(box, width, margin=8):
    """True when a box reaches both side edges of the image (within margin px): no door edge in view, so it can't place a door."""
    return box[0] <= margin and box[2] >= width - 1 - margin


def in_front_of_wall(box, depth, strip=6, tol=0.25):
    """True when the middle half of the box (rows and columns) is nearer than the wall strips beside it by more than tol: something
    stands in front of the wall (furniture, a swung panel). No valid inside or wall pixel -> False."""
    wall = _wall_median(box, depth, strip)
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    ra, rb = max(0, y1 + (y2 - y1) // 4), min(depth.shape[0], y1 + 3 * (y2 - y1) // 4 + 1)
    v = _med_valid(depth, ra, rb, x1 + (x2 - x1) // 4, x1 + 3 * (x2 - x1) // 4)
    return wall is not None and v.size > 0 and float(np.median(v)) < wall - tol
