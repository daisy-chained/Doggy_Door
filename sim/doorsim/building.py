"""Building generator (robot run phase 2e): one floor, a corridor with rooms on both sides, doors from the door-room v5 sampler.

Frames: world x east, y north, z up, metres. A "space" is the corridor ("corridor") or a room ("r0", "r1", ...); rects are
[x0, y0, x1, y1] from wall centre to wall centre, walls are WALL_T thick and centred on the shared edges.
Doors are built exactly as in the v5 world (doors.py) in their own local frame (wall centre plane at x = WALL_X, door centre at
y = SLOT_Y["a"]) and moved into the building by a 2D rigid transform: rotate by the door's yaw about z, then translate so the local
door centre lands on the door's pos. Local +x = the normal from connects[0] (front space) into connects[1] (back space), so "front"
(camera at x < WALL_X) is connects[0]. Room <-> corridor doors have the corridor as connects[0]; neighbour doors have the west room
as connects[0] and the east room as connects[1].
Open doors: the door's joint controller gets <initial_position> = open_frac * the v5 open value (hinged: 1.2 rad about the hinge,
sliding: 0.9 of the travel); closed doors stay at 0. Label ids: door id i uses base = 10 * i; base + 1 panel, + 2 handle (front),
+ 3 glass, + 4 handle (back), + 5..7 hinge knuckles (as v5 with k = i), so door ids stay < 200 for i <= MAX_DOORS (10 * 19 + 7 = 197).
Clutter (cfg "clutter", optional): items of clutter.POOL, label 200 + (pool id - 100) = 200..247; floor items stand in the spaces
(occupied in the map), wall items hang flat on a wall face. Randomness: numpy Generator.
Houses (cfg kind "house", round 1 of the Dutch-house world): sample_house / place_house_doors / make_building(..., floor); a hub
("corridor" = hall / landing strip) plus rooms, a planned door per room pair, and a "stairs" obstacle box (see sample_house)."""
import json, math, pathlib, re
import numpy as np

from doorsim import clutter, doors

WALL_T = 0.1
MAX_DOORS = 19  # base 10 * id + 7 < 200 (clutter labels start at 200) -> id <= 19
SLOT = "a"      # every door is built in the v5 slot "a" frame (door centre at y = SLOT_Y["a"])


def _u(rng, lo_hi):
    return float(rng.uniform(*lo_hi))


# --- layout ---
def _cuts(rng, length, lo, hi):
    """Widths in [lo, hi] that sum to `length`: a random room count that can fit it, then random widths that keep the rest feasible."""
    counts = [n for n in range(1, int(length / lo) + 2) if n * lo <= length + 1e-9 and length <= n * hi + 1e-9]
    n = int(rng.choice(counts))
    out, rem = [], length
    for k in range(n, 1, -1):
        out.append(float(rng.uniform(max(lo, rem - (k - 1) * hi), min(hi, rem - (k - 1) * lo))))
        rem -= out[-1]
    return out + [rem]


def sample_layout(rng, cfg):
    """{"bounds", "corridor", "rooms": [{"id", "rect", "side"}]}: corridor along x, rooms south (below) and north (above) of it,
    cut at random widths (cfg room_width), one random depth per side (cfg room_depth)."""
    length, cw = _u(rng, cfg["corridor_length"]), _u(rng, cfg["corridor_width"])
    depth = {"south": _u(rng, cfg["room_depth"]), "north": _u(rng, cfg["room_depth"])}
    y0 = depth["south"]
    corridor = [0.0, y0, length, y0 + cw]
    rooms, n = [], 0
    for side in ("south", "north"):
        x = 0.0
        for w in _cuts(rng, length, *cfg["room_width"]):
            x1 = length if abs(x + w - length) < 1e-9 else x + w
            rect = [x, corridor[3], x1, corridor[3] + depth[side]] if side == "north" else [x, 0.0, x1, y0]
            rooms.append({"id": f"r{n}", "rect": [float(v) for v in rect], "side": side})
            x, n = x1, n + 1
    return {"bounds": [0.0, 0.0, length, y0 + cw + depth["north"]], "corridor": corridor, "rooms": rooms}


def _spaces(layout):
    return {"corridor": layout["corridor"], **{r["id"]: r["rect"] for r in layout["rooms"]}}


# --- doors ---
def _rot(yaw, x, y):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * x - s * y, s * x + c * y


def _world(door, lx, ly):
    """World (x, y) of the local door-frame point (lx, ly)."""
    dx, dy = _rot(door["yaw"], lx - doors.WALL_X, ly - doors.SLOT_Y[SLOT])
    return door["pos"][0] + dx, door["pos"][1] + dy


def hinge_xy(d):
    """World (x, y) of a hinged door's hinge axis; None for a sliding door."""
    if d["spec"]["door_type"] == "sliding":
        return None
    g = doors._layout(d["spec"], SLOT)
    return _world(d, g["hinge"][0], g["hinge"][1])


def open_position(d):
    """Joint value of the door fully open as the sim opens it (rad hinged, m sliding)."""
    return doors._layout(d["spec"], SLOT)["open"]


def door_topic(d):
    """gz topic of the door's joint position controller."""
    return f"/{_names(d['id'])['door']}/cmd_pos"


def _extent(spec, sign):
    """(lo, hi) of opening + trim (+ the slide path of a sliding door) along the wall, relative to the door centre, where `sign` = +1
    if the local y axis points along the wall's +coordinate, else -1."""
    g = doors._layout(spec, SLOT)
    half = g["w"] / 2 + g["tw"]
    lo, hi = -half, half
    if spec["door_type"] == "sliding":
        lo, hi = min(lo, g["path"][0] - g["y"]), max(hi, g["path"][1] - g["y"])
    return (lo, hi) if sign > 0 else (-hi, -lo)


def _walls_between(layout):
    """(front, back, axis, coord, lo, hi, yaw) of every wall a door may sit on: room <-> corridor (front = corridor) and
    west room -> east room on the same side."""
    sp, out = _spaces(layout), []
    for r in layout["rooms"]:
        c, x0, x1 = layout["corridor"], r["rect"][0], r["rect"][2]
        if r["side"] == "south":
            out.append(("corridor", r["id"], "x", c[1], x0, x1, -math.pi / 2))
        else:
            out.append(("corridor", r["id"], "x", c[3], x0, x1, math.pi / 2))
    rooms = layout["rooms"]
    for a in rooms:
        for b in rooms:
            if a["side"] == b["side"] and abs(a["rect"][2] - b["rect"][0]) < 1e-9:
                out.append((a["id"], b["id"], "y", a["rect"][2], max(a["rect"][1], b["rect"][1]), min(a["rect"][3], b["rect"][3]), 0.0))
    return out


def _sample_spec(rng, cfg, split):
    combos = cfg["heldout"]
    spec = doors.sample_door(rng, cfg["door"], SLOT) if split == "test" else doors._draw(rng, cfg["door"], SLOT, combos, False)
    spec["door_stop"] = bool(cfg.get("door_stop", False))
    if spec["door_type"] == "sliding":
        spec["slide_y"] = int(rng.choice([-1, 1]))
    return spec


def _try_place(rng, spec, wall, taken, margin):
    """Along-wall coordinate of the door centre (random), or None if the door (+ trim + slide path) does not fit between the
    wall ends (margin) and the doors already on that wall (`taken`: list of (lo, hi))."""
    _, _, axis, coord, lo, hi, yaw = wall
    # local y points along world +coordinate of the wall when the rotated (0, 1) has a positive component on it
    vx, vy = _rot(yaw, 0.0, 1.0)
    sign = (vx if axis == "x" else vy)
    e_lo, e_hi = _extent(spec, 1 if sign > 0 else -1)
    c_lo, c_hi = lo + margin - e_lo, hi - margin - e_hi
    if c_hi < c_lo:
        return None
    for _ in range(100):
        c = float(rng.uniform(c_lo, c_hi))
        if all(c + e_hi <= t0 + 1e-9 or c + e_lo >= t1 - 1e-9 for t0, t1 in taken):
            return c, (c + e_lo, c + e_hi)
    return None


def place_doors(rng, layout, cfg, split="train"):
    """Doors of a building: every room one door to the corridor (a second one with cfg p_second_door), and with cfg p_neighbour_door
    one to its east neighbour; door dicts as in the tests (id, spec, connects, pos, yaw, state, open_frac, heldout)."""
    walls = _walls_between(layout)
    corr = {w[1]: [w] for w in walls if w[0] == "corridor"}
    neigh = {w[0]: w for w in walls if w[0] != "corridor"}
    out, taken = [], {}
    rooms = [r["id"] for r in layout["rooms"]]

    def add(wall, max_tries=30):
        if len(out) >= MAX_DOORS:
            return False
        key = (wall[2], round(wall[3], 6))
        for _ in range(max_tries):
            spec = _sample_spec(rng, cfg, split)
            hit = _try_place(rng, spec, wall, taken.setdefault(key, []), cfg["corner_margin"])
            if hit is None:
                continue
            c, span = hit
            taken[key].append(span)
            closed = bool(rng.random() < cfg["p_closed"])
            frac = 0.0 if closed else round(float(rng.uniform(cfg["min_open"], 1.0)), 3)
            pos = [c, wall[3]] if wall[2] == "x" else [wall[3], c]
            out.append({"id": len(out) + 1, "spec": spec, "connects": [wall[0], wall[1]], "pos": [float(pos[0]), float(pos[1])],
                        "yaw": float(wall[6]), "state": "closed" if closed else "open", "open_frac": frac,
                        "heldout": bool(doors.is_heldout(spec, cfg["heldout"]))})
            return True
        return False

    for i, rid in enumerate(rooms):
        if not add(corr[rid][0], 200):
            raise ValueError(f"no door fits room {rid}: check room_width / corner_margin / door width")
        left = len(rooms) - i - 1  # primary doors still to come must keep their slot
        if rng.random() < cfg["p_second_door"] and len(out) + left < MAX_DOORS:
            add(corr[rid][0])
        if rid in neigh and rng.random() < cfg["p_neighbour_door"] and len(out) + left < MAX_DOORS:
            add(neigh[rid])
    return out


# --- approach poses ---
def door_gap(d):
    """Clear width (m) the robot can pass with the panel where it stands; 0.0 unless open. Hinged: the panel swings out of the wall
    about a hinge at one jamb, so the squeeze is the far jamb to the panel = width * sin(th). Matches doors._layout (hinged open
    angle 1.2 rad, sliding panel width+0.08, travel width+0.15, open = 0.9 * travel)."""
    if d["state"] != "open":
        return 0.0
    w, f = d["spec"]["width"], d["open_frac"]
    if d["spec"]["door_type"] == "sliding":
        return float(min(max(0.9 * (w + 0.15) * f - 0.04, 0.0), w))
    return float(w * math.sin(1.2 * f))


def approach_poses(layout, doors_, cfg, clutter=None):
    """One pose per door side and distance in cfg approach.distances: on the door's normal line, `dist` from the door centre into
    that space (clamped to stay robot_radius + WALL_T / 2 inside it), facing the door centre; reachable = space reachable from the
    corridor through open doors with a clear gap (door_gap) of at least cfg min_door_gap (missing = any open door). clear = no
    occupied map cell (occupancy() at cfg map_resolution) within robot_radius: a blocked pose (e.g. on an open door panel or floor clutter) is moved further back along the normal in 0.1 m steps (staying inside the space), then closer down to
    0.5 m; dist = the distance used; clear False only if nothing fits."""
    sp, m = _spaces(layout), cfg["robot_radius"] + WALL_T / 2
    res = float(cfg.get("map_resolution", 0.05))
    grid, origin = occupancy(layout, doors_, res, clutter)
    rc = int(math.ceil(cfg["robot_radius"] / res))

    def free(x, y):
        c, row = int((x - origin[0]) / res), grid.shape[0] - 1 - int((y - origin[1]) / res)
        return not (grid[max(0, row - rc):row + rc + 1, max(0, c - rc):c + rc + 1] == 0).any()

    reach, grow = {"corridor"}, True
    while grow:
        grow = False
        for d in doors_:
            a, b = d["connects"]
            if d["state"] == "open" and door_gap(d) >= cfg.get("min_door_gap", 0.0) and (a in reach) != (b in reach):
                reach |= {a, b}
                grow = True
    out = []
    for d in doors_:
        nx, ny = round(math.cos(d["yaw"])), round(math.sin(d["yaw"]))
        for side, k, s in (("front", 0, -1), ("back", 1, 1)):
            space = d["connects"][k]
            r = sp[space]
            ux, uy = s * nx, s * ny
            room = (r[2] - m - d["pos"][0]) * ux if ux > 0 else (d["pos"][0] - (r[0] + m)) * -ux if ux < 0 else math.inf
            roomy = (r[3] - m - d["pos"][1]) * uy if uy > 0 else (d["pos"][1] - (r[1] + m)) * -uy if uy < 0 else math.inf
            dmax = min(room, roomy)
            for dist in cfg["approach"]["distances"]:
                t0 = min(dist, dmax)
                tries = [t0] + [t0 + 0.1 * k for k in range(1, 100) if t0 + 0.1 * k <= dmax + 1e-9] + \
                        [t0 - 0.1 * k for k in range(1, 100) if t0 - 0.1 * k >= 0.5 - 1e-9]
                t = next((u for u in tries if free(d["pos"][0] + ux * u, d["pos"][1] + uy * u)), None)
                ok = t is not None
                t = t0 if t is None else t
                x, y = d["pos"][0] + ux * t, d["pos"][1] + uy * t
                out.append({"door": d["id"], "space": space, "side": side, "dist": float(round(t, 6)), "x": float(x), "y": float(y),
                            "yaw": float(math.atan2(d["pos"][1] - y, d["pos"][0] - x)), "reachable": space in reach, "clear": ok})
    return out


# --- walls ---
def _door_wall(d):
    """(axis, rounded coord, along) of the wall a door sits on: axis "x" = wall along x at y = coord."""
    if abs(math.cos(d["yaw"])) > 0.5:
        return "y", round(d["pos"][0], 6), d["pos"][1]
    return "x", round(d["pos"][1], 6), d["pos"][0]


def _wall_pieces(layout, doors_):
    """(axis, coord, lo, hi) wall pieces (centre line coord, along-wall span) of the whole building, with a gap as wide as the
    door at every door."""
    groups = {}
    for r in list(layout["rooms"]) + [{"id": "corridor", "rect": layout["corridor"]}]:
        x0, y0, x1, y1 = r["rect"]
        for axis, coord, lo, hi in (("x", y0, x0, x1), ("x", y1, x0, x1), ("y", x0, y0, y1), ("y", x1, y0, y1)):
            groups.setdefault((axis, round(coord, 6)), []).append((lo, hi))
    pieces = []
    for (axis, coord), spans in groups.items():
        spans.sort()
        merged = [list(spans[0])]
        for lo, hi in spans[1:]:
            if lo <= merged[-1][1] + 1e-9:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        gaps = sorted(((_door_wall(d)[2], d) for d in doors_ if _door_wall(d)[:2] == (axis, round(coord, 6))), key=lambda g: g[0])
        for lo, hi in merged:
            if axis == "x":
                lo, hi = lo - WALL_T / 2, hi + WALL_T / 2  # x walls cover the corners
            cur = lo
            for along, d in gaps:
                a, b = along - d["spec"]["width"] / 2, along + d["spec"]["width"] / 2
                if lo <= along <= hi:
                    pieces.append((axis, coord, cur, a))
                    cur = b
            pieces.append((axis, coord, cur, hi))
    return pieces


def _rect_of(piece):
    axis, coord, lo, hi = piece
    return (lo, hi, coord - WALL_T / 2, coord + WALL_T / 2) if axis == "x" else (coord - WALL_T / 2, coord + WALL_T / 2, lo, hi)


def _opening_rect(d):
    """(x0, x1, y0, y1) of a door's opening in the wall plane (width of the door, WALL_T thick)."""
    w = d["spec"]["width"] / 2
    if abs(math.cos(d["yaw"])) > 0.5:  # normal along x -> wall along y
        return d["pos"][0] - WALL_T / 2, d["pos"][0] + WALL_T / 2, d["pos"][1] - w, d["pos"][1] + w
    return d["pos"][0] - w, d["pos"][0] + w, d["pos"][1] - WALL_T / 2, d["pos"][1] + WALL_T / 2


def _wall_boxes(layout, doors_):
    """Wall + lintel boxes (name, rgb, x0, x1, y0, y1, z0, z1) of the whole building."""
    boxes = []
    for i, p in enumerate(_wall_pieces(layout, doors_)):
        x0, x1, y0, y1 = _rect_of(p)
        if x1 - x0 > 1e-6 and y1 - y0 > 1e-6:
            boxes.append((f"wall_{i}", doors.WALL_RGB, x0, x1, y0, y1, 0.0, doors.ROOM_HEIGHT))
    for d in doors_:
        x0, x1, y0, y1 = _opening_rect(d)
        boxes.append((f"lintel_{d['id']}", doors.WALL_RGB, x0, x1, y0, y1, d["spec"]["height"], doors.ROOM_HEIGHT))
    return boxes


_POOL = {p["name"]: p for p in clutter.POOL}


# --- occupancy map ---
def occupancy(layout, doors_, res, clutter=None):
    """(grid uint8 H x W, origin [x, y]): 0 = occupied, 254 = free, row 0 = north edge, origin = world xy of the bottom-left cell.
    Free = inside a space; occupied = outside, the walls, the doorway of a closed door and the panel of an open one where it stands
    (hinged: the panel's centre line turned by open_frac * the v5 open angle about its hinge; sliding: shifted along its path)
    and the footprints of the floor items in `clutter` (b["clutter"])."""
    bx, pad = layout["bounds"], 0.3
    ox, oy = bx[0] - pad, bx[1] - pad
    w, h = int(math.ceil((bx[2] - bx[0] + 2 * pad) / res)), int(math.ceil((bx[3] - bx[1] + 2 * pad) / res))
    cx = ox + (np.arange(w) + 0.5) * res
    cy = oy + (np.arange(h)[::-1] + 0.5) * res  # row 0 = north
    free = np.zeros((h, w), bool)

    def fill(mask_rect, value):
        x0, x1, y0, y1 = mask_rect
        m = ((cx >= x0 - 1e-9) & (cx <= x1 + 1e-9))[None, :] & ((cy >= y0 - 1e-9) & (cy <= y1 + 1e-9))[:, None]
        free[m] = value

    for r in _spaces(layout).values():
        fill((r[0], r[2], r[1], r[3]), True)
    for p in _wall_pieces(layout, doors_):
        fill(_rect_of(p), False)
    if "stairs" in layout:
        st = layout["stairs"]
        fill((st[0], st[2], st[1], st[3]), False)

    def mark(x, y):
        col, row = int((x - ox) / res), h - 1 - int((y - oy) / res)
        if 0 <= col < w and 0 <= row < h:
            free[row, col] = False

    for d in doors_:
        if d["state"] == "closed":
            fill(_opening_rect(d), False)
            continue
        spec, g = d["spec"], doors._layout(d["spec"], SLOT)
        if doors.is_hinged_like(spec):
            th = g["open"] * d["open_frac"]
            hy, hx = spec["hinge_y"], g["hinge"][0]
            leaves = [(hy, th, g["hinge"][1])] + ([(-hy, -th, g["hinge_b"][1])] if doors.is_double(spec) else [])  # double: both leaves swing to swing_x
            for hy_, th_, y0 in leaves:
                for t in np.linspace(0, g["pw"], int(g["pw"] / res * 4) + 2):  # centre line of the panel, hinge -> lock edge
                    mark(*_world(d, hx + hy_ * t * math.sin(th_), y0 - hy_ * t * math.cos(th_)))
        else:
            s, shift = spec["slide_y"], g["open"] * d["open_frac"]
            for t in np.linspace(-g["pw"] / 2, g["pw"] / 2, int(g["pw"] / res * 2) + 2):
                for dx in (-0.02, 0.0, 0.02):
                    mark(*_world(d, g["xc"] + dx, g["y"] + s * (t + shift)))
    for c in clutter or []:
        if c["place"] != "floor":
            continue
        p, (x, y) = _POOL[c["item"]], c["position"][:2]
        co, si = math.cos(c["yaw"]), math.sin(c["yaw"])
        dx, dy = cx[None, :] - x, cy[:, None] - y
        lx, ly = co * dx + si * dy, -si * dx + co * dy
        free[(np.abs(lx) <= p["size"][0] / 2 + 1e-9) & (np.abs(ly) <= p["size"][1] / 2 + 1e-9)] = False
        mark(x, y)
    grid = np.where(free, 254, 0).astype(np.uint8)
    return grid, [float(ox), float(oy)]


# --- clutter ---
_EPS = 1e-4
_WALL_YAW = {"x0": math.pi, "x1": 0.0, "y0": -math.pi / 2, "y1": math.pi / 2}  # item back (local +x) on that wall of the space


def _door_line_span(d, margin):
    """(wall axis "x" | "y", wall line coord, lo, hi) of a door's opening + trim (+ slide path, both directions) grown by margin."""
    lo, hi = _extent(d["spec"], 1)
    lo, hi = min(lo, -hi) - margin, max(hi, -lo) + margin
    if abs(math.cos(d["yaw"])) > 0.5:
        return "y", d["pos"][0], d["pos"][1] + lo, d["pos"][1] + hi
    return "x", d["pos"][1], d["pos"][0] + lo, d["pos"][0] + hi


def _keep_outs(ds, space, rect, cfg):
    """Keep-out strips [x0, y0, x1, y1] inside the space for every door touching it: along the wall the door's span grown by
    robot_radius, into the space max(approach distances) + robot_radius from the wall centre line."""
    deep, out = max(cfg["approach"]["distances"]) + cfg["robot_radius"], []
    for d in ds:
        if space not in d["connects"]:
            continue
        axis, line, lo, hi = _door_line_span(d, cfg["robot_radius"])
        if axis == "x":
            out.append([lo, line, hi, line + deep] if rect[1] >= line - 1e-6 else [lo, line - deep, hi, line])
        else:
            out.append([line, lo, line + deep, hi] if rect[0] >= line - 1e-6 else [line - deep, lo, line, hi])
    return out


def _box_hit(a, b, grow=0.0):
    """True if the [x0, y0, x1, y1] boxes a (grown by `grow`) and b overlap."""
    return min(a[2] + grow, b[2]) - max(a[0] - grow, b[0]) > 1e-6 and min(a[3] + grow, b[3]) - max(a[1] - grow, b[1]) > 1e-6


def _floor_spot(rng, p, rect, keeps, taken, gap, against, tries=50):
    """(x, y, yaw) of a floor item inside the rect (shrunk by WALL_T / 2), off the keep-out strips and `taken` boxes (gap), or None."""
    e = WALL_T / 2 + _EPS
    for _ in range(tries):
        if rng.random() < against:
            side = ("x0", "x1", "y0", "y1")[int(rng.integers(4))]
            yaw = _WALL_YAW[side]
        else:
            side, yaw = None, float(rng.uniform(-math.pi, math.pi))
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        hx, hy = c * p["size"][0] / 2 + s * p["size"][1] / 2, s * p["size"][0] / 2 + c * p["size"][1] / 2
        xr, yr = (rect[0] + e + hx, rect[2] - e - hx), (rect[1] + e + hy, rect[3] - e - hy)
        if xr[1] < xr[0] or yr[1] < yr[0]:
            continue
        x, y = float(rng.uniform(*xr)), float(rng.uniform(*yr))
        if side:
            x = {"x0": xr[0], "x1": xr[1]}.get(side, x)
            y = {"y0": yr[0], "y1": yr[1]}.get(side, y)
        bb = [x - hx, y - hy, x + hx, y + hy]
        if any(_box_hit(bb, k) for k in keeps) or any(_box_hit(bb, t, gap) for t in taken):
            continue
        return x, y, yaw
    return None


def _wall_spot(rng, p, rect, ds, space, taken, cc, tries=50):
    """(x, y, z, yaw, wall) of a wall item hanging flat on a wall face of the space: inside wall_z and the wall segment, off the
    door spans (grown by door_margin) on that wall line and the other wall items there (`taken`: {wall: [(lo, hi)]}), or None."""
    d, wd, h = p["size"]
    z0, z1 = cc["wall_z"]
    if h > z1 - z0:
        return None
    for _ in range(tries):
        side = ("x0", "x1", "y0", "y1")[int(rng.integers(4))]
        axis = "y" if side[0] == "x" else "x"        # the wall runs along this axis
        line = {"x0": rect[0], "x1": rect[2], "y0": rect[1], "y1": rect[3]}[side]
        lo, hi = (rect[1], rect[3]) if axis == "y" else (rect[0], rect[2])
        lo, hi = lo + WALL_T / 2, hi - WALL_T / 2              # off the corner walls
        if hi - lo < wd:
            continue
        a, z = float(rng.uniform(lo + wd / 2, hi - wd / 2)), float(rng.uniform(z0 + h / 2, z1 - h / 2))
        span = (a - wd / 2, a + wd / 2)
        doors_on = [_door_line_span(q, cc["door_margin"]) for q in ds]
        if any(ax == ("x" if axis == "x" else "y") and abs(ln - line) < 1e-6 and span[0] < hi_ and lo_ < span[1] for ax, ln, lo_, hi_ in doors_on):
            continue
        if any(span[0] < t1 and t0 < span[1] for t0, t1 in taken.get((space, side), [])):
            continue
        off = WALL_T / 2 + d / 2
        n = {"x0": line + off, "x1": line - off, "y0": line + off, "y1": line - off}[side]
        return (n, a, z) if axis == "y" else (a, n, z), _WALL_YAW[side], (space, side), span
    return None


def _fit_labels(grid, res, radius):
    """Connected components of the cells where a robot of `radius` fits (no occupied cell within it)."""
    from scipy import ndimage
    r = int(math.ceil(radius / res))
    return ndimage.label(~ndimage.maximum_filter(grid == 0, size=2 * r + 1, mode="constant", cval=0))[0]


def place_clutter(rng, layout, ds, cfg, keeps=None):
    """Clutter entries (see the module docstring) for cfg["clutter"]: per space (corridor first) a random number of floor items,
    then of wall items, each pool item at most once per space and place; an item that finds no spot (50 tries) is skipped, and
    a floor item that cuts apart approach poses of its space that were connected before is dropped. Houses: the stairs box and the extra
    `keeps` boxes ([x0, y0, x1, y1], e.g. the start spot) are off limits to floor items, the stairs also to wall items."""
    cc, out, taken_wall = cfg["clutter"], [], {}
    res = float(cfg.get("map_resolution", 0.05))
    lab0 = _fit_labels(occupancy(layout, ds, res)[0], res, cfg["robot_radius"])
    for space, rect in _spaces(layout).items():
        counts = cc["corridor" if space == "corridor" else "rooms"]
        kos = _keep_outs(ds, space, rect, cfg) + ([layout["stairs"]] if "stairs" in layout else []) + list(keeps or [])
        for place_kind in ("floor", "wall"):
            group = [q for q in clutter.POOL if q["place"] == place_kind]
            lo, hi = counts[f"{place_kind}_count"]
            want, n = int(rng.integers(lo, hi + 1)), 0
            for i in rng.permutation(len(group)):
                if n >= want:
                    break
                p = group[i]
                if place_kind == "floor":
                    taken = [_aabb_of(c) for c in out if c["space"] == space and c["place"] == "floor"]
                    spot = _floor_spot(rng, p, rect, kos, taken, cc["gap"], cc["p_against_wall"])
                    if spot is None:
                        continue
                    pos, yaw = [spot[0], spot[1], p["size"][2] / 2], spot[2]
                else:
                    spot = _wall_spot(rng, p, rect, ds, space, taken_wall, cc)
                    if spot is None:
                        continue
                    if "stairs" in layout and _box_hit(_aabb_of({"item": p["name"], "position": list(spot[0]), "yaw": spot[1]}), layout["stairs"], 0.02):
                        continue
                    pos, yaw = list(spot[0]), spot[1]
                colour = [float(v) for v in rng.uniform(0, 1, 3)] if p["recolour"] else None
                entry = {"name": f"clutter_{len(out)}", "item": p["name"], "label": 200 + p["id"] - 100, "space": space, "place": place_kind,
                         "position": [float(v) for v in pos], "yaw": float(yaw), "colour": colour}
                if place_kind == "floor" and not _keeps_connected(layout, ds, cfg, out + [entry], space, lab0):
                    continue
                if place_kind == "wall":
                    taken_wall.setdefault(spot[2], []).append(spot[3])
                out.append(entry)
                n += 1
    return out


def _aabb_of(c):
    """[x0, y0, x1, y1] of a floor clutter entry's footprint."""
    xs, ys = zip(*clutter.footprint(c, _POOL[c["item"]]))
    return [min(xs), min(ys), max(xs), max(ys)]


def _keeps_connected(layout, ds, cfg, items, space, lab0):
    """True if, with `items` in, the clear approach poses of `space` that are in one fit component without clutter (lab0) still are."""
    res = float(cfg.get("map_resolution", 0.05))
    grid, origin = occupancy(layout, ds, res, items)
    lab = _fit_labels(grid, res, cfg["robot_radius"])
    cell = lambda p: (grid.shape[0] - 1 - int((p["y"] - origin[1]) / res), int((p["x"] - origin[0]) / res))
    clear = [cell(p) for p in approach_poses(layout, ds, cfg, items) if p["space"] == space and p["clear"]]
    return all(lab[a] == lab[b] for i, a in enumerate(clear) for b in clear[i + 1:] if lab0[a] == lab0[b] != 0)


# --- the world file ---
def _names(i):
    return {"door": f"door_{i}_panel", "handle_front": f"door_{i}_handle_front", "handle_back": f"door_{i}_handle_back",
            "glass": f"door_{i}_glass", "hinge": f"door_{i}_hinge_{{}}"}


def door_labels(d):
    """{part: label id} of one door (see the module docstring)."""
    base, spec = 10 * d["id"], d["spec"]
    out = {"panel": base + 1, "handle_front": base + 2, "handle_back": base + 4}
    if doors.has_pane(spec):
        out["glass"] = base + 3
    if doors._has_hinges(spec):
        out.update({f"hinge_{i}": base + 5 + i for i in range(3)})
    return out


def _place(text, d):
    """Move a v5 door-frame model (its first <pose> = x y z roll pitch yaw in the local frame) into the building."""
    m = re.search(r"<pose>([^<]*)</pose>", text)
    x, y, z, r, p, yaw = [float(v) for v in m.group(1).split()]
    wx, wy = _world(d, x, y)
    new = doors._pose(wx, wy, z, r, p, yaw + d["yaw"])
    return text[:m.start(1)] + new + text[m.end(1):]


def _door_models(d):
    """SDF of one door's models (panel, handles, glass, hinges, wall-frame trim / stop / rail), placed at d["pos"] / d["yaw"]."""
    spec, i = d["spec"], d["id"]
    g, nm = doors._layout(spec, SLOT), _names(i)
    start = g["open"] * d["open_frac"] if d["state"] == "open" else 0.0
    parts = [_place(doors._door_model(SLOT, i, spec, g, nm, initial=start), d)]
    if doors.has_pane(spec):
        parts.append(_place(doors._glass_model(SLOT, i, g, nm, spec), d))
    parts += [_place(doors._handle_model(SLOT, i, spec, g, f, nm), d) for f in ("front", "back")]
    if doors._has_hinges(spec):
        parts += [_place(doors._hinge_model(SLOT, i, spec, g, k, nm), d) for k in range(3)]
    vis = doors._door_wall_visuals(spec, g, f"{i}")
    if vis:  # origin on the wall centre plane at y = 0 of the local frame
        pose = doors._pose(*_world(d, doors.WALL_X, 0.0), 0, 0, 0, d["yaw"])
        parts.append(f'    <model name="door_{i}_frame">\n      <static>true</static>\n      <pose>{pose}</pose>\n      <link name="link">\n'
                     + "".join("        " + v + "\n" for v in vis) + "      </link>\n    </model>\n")
    return "".join(parts)


def _clutter_model(c):
    """SDF of one clutter entry: a static model at its pose with a visual and a collision per part (layout of clutter._model) and its label."""
    p = _POOL[c["item"]]
    h, parts = p["size"][2], []
    for i, (shape, (x, y, z), dims, rgb) in enumerate(p["parts"]):
        col = " ".join(f"{v:.3f}" for v in (c["colour"] if p["recolour"] else rgb))
        pose, geo = f"{x} {y} {z - h / 2:.4f} 0 0 0", clutter._geometry(shape, dims)
        parts.append(f'        <visual name="v{i}"><pose>{pose}</pose><geometry>{geo}</geometry>'
                     f'<material><diffuse>{col} 1</diffuse><ambient>{col} 1</ambient></material></visual>\n'
                     f'        <collision name="c{i}"><pose>{pose}</pose><geometry>{geo}</geometry></collision>\n')
    x, y, z = c["position"]
    return (f'    <model name="{c["name"]}">\n      <static>true</static>\n      <pose>{x!r} {y!r} {z!r} 0 0 {c["yaw"]!r}</pose>\n'
            '      <link name="link">\n' + "".join(parts) + "      </link>\n"
            f'      <plugin filename="gz-sim-label-system" name="gz::sim::systems::Label"><label>{c["label"]}</label></plugin>\n    </model>\n')


def building_sdf(b, name):
    """Full SDF text: plugins / scene / sun of door_room_v3.sdf, floor, ceiling, walls with door openings, one ceiling light per space,
    every door's models, one static model per clutter entry (visuals + collisions + label), the v3 camera rig (at the corridor centre)."""
    v3 = (doors.ROOM / "door_room_v3.sdf").read_text()
    head = v3[v3.index("<sdf "):v3.index('    <light type="point" name="ceiling_light">')]
    head = head.replace('<world name="door_room_v3">', f'<world name="{name}">')
    lay = b["layout"]
    bx = lay["bounds"]
    lights = "".join(f'    <light type="point" name="ceiling_light_{sid}">\n      <pose>{doors._pose((r[0] + r[2]) / 2, (r[1] + r[3]) / 2, 2.5)} 0 0 0</pose>\n'
                     '      <diffuse>0.8 0.8 0.75 1</diffuse>\n      <attenuation><range>8</range><constant>0.5</constant><linear>0.1</linear>'
                     '<quadratic>0.02</quadratic></attenuation>\n    </light>\n' for sid, r in _spaces(lay).items())
    e = WALL_T / 2
    floor = doors._boxes_model("floor", "floor over the whole building", [("floor", (0.45, 0.45, 0.45), bx[0] - e, bx[2] + e, bx[1] - e, bx[3] + e, -0.1, 0.0)])
    ceiling = doors._boxes_model("ceiling", "ceiling over the whole building", [("ceiling", doors.CEILING_RGB, bx[0] - e, bx[2] + e, bx[1] - e, bx[3] + e, doors.ROOM_HEIGHT, doors.ROOM_HEIGHT + 0.1)])
    walls = doors._boxes_model("walls", f"walls ({WALL_T} thick) with a gap + lintel per door", _wall_boxes(lay, b["doors"]))
    stairs = ""
    if "stairs" in lay:  # houses: a solid box (ground floor: the staircase up to the ceiling; upper floor: the railing round the void)
        s = lay["stairs"]
        stairs = "\n" + doors._boxes_model("stairs", "stairs: an obstacle box (no label)", [("stairs", (0.55, 0.42, 0.3) if lay.get("floor", 0) == 0 else (0.5, 0.5, 0.55),
                                                                                           s[0], s[2], s[1], s[3], 0.0, lay["stairs_height"])])
    rig = v3[v3.index("    <!-- Fixed sensor rig"):v3.index("    <!-- clutter pool")]
    c = lay["corridor"]
    rig = rig.replace("<pose>0 0 1.2 0 0 0</pose>", f"<pose>{doors._pose((c[0] + c[2]) / 2, (c[1] + c[3]) / 2, 1.2)} 0 0 0</pose>", 1)
    parts = [f'<?xml version="1.0" ?>\n<!-- Building {name} (seed {b["seed"]}, split {b["split"]}): generated by sim/building/make_buildings.py -->\n', head, lights, "\n",
             floor, ceiling, walls, stairs] + ["\n" + _door_models(d) for d in b["doors"]] + ["\n" + _clutter_model(c) for c in b.get("clutter", [])]
    return "".join(parts) + "\n" + rig + "  </world>\n</sdf>\n"


# --- whole building ---
def make_building(cfg, seed, split="train", floor=0):
    """{"seed", "split", "layout", "doors", "clutter", "poses", "map_resolution"}: deterministic for a seed. cfg kind "house":
    sample_house (floor 0 / 1 of the same footprint) + one door per planned pair + "start" in the hub; `floor` is ignored otherwise."""
    rng = np.random.default_rng(seed)
    if cfg.get("kind") == "house":
        return _make_house(cfg, rng, seed, split, floor)
    layout = sample_layout(rng, cfg)
    ds = place_doors(rng, layout, cfg, split)
    cl = place_clutter(rng, layout, ds, cfg) if "clutter" in cfg else []  # after all door draws: doors / layout stay as without clutter
    return {"seed": int(seed), "split": split, "layout": layout, "doors": ds, "clutter": cl, "poses": approach_poses(layout, ds, cfg, cl),
            "map_resolution": float(cfg.get("map_resolution", 0.05))}


# --- houses ---
def _shared_wall(ra, rb):
    """(axis, coord, lo, hi, yaw) of the wall rect ra shares with rect rb (yaw = normal from ra into rb), or None."""
    ov = lambda i: (max(ra[i], rb[i]), min(ra[i + 2], rb[i + 2]))
    for e, o, axis, coord, i, yaw in ((ra[2], rb[0], "y", ra[2], 1, 0.0), (ra[0], rb[2], "y", ra[0], 1, math.pi),
                                      (ra[3], rb[1], "x", ra[3], 0, math.pi / 2), (ra[1], rb[3], "x", ra[1], 0, -math.pi / 2)):
        if abs(e - o) < 1e-9:
            lo, hi = ov(i)
            return (axis, coord, lo, hi, yaw) if hi - lo > 1e-9 else None
    return None


_STAIR_GAP = 0.1  # m between a door's trim / slide path and the stairs (and the wall end) on a wall the stairs touch


def house_walls(layout, margin):
    """{"a|b": (front, back, axis, coord, lo, hi, yaw)} of every planned pair, as _walls_between. A wall whose hub-side strip (0.5 m
    deep) touches the stairs is cut to its larger free piece, with _STAIR_GAP clear at both ends instead of `margin` (the door
    must not stand against the stairs); a pair without a shared wall is left out."""
    sp, st, out = _spaces(layout), layout.get("stairs"), {}
    for a, b in layout["plan"]:
        w = _shared_wall(sp[a], sp[b])
        if w is None:
            continue
        axis, coord, lo, hi, yaw = w
        if st is not None and a == "corridor":
            n = math.sin(yaw) if axis == "x" else math.cos(yaw)
            d0, d1 = (coord - 0.5, coord) if n > 0 else (coord, coord + 0.5)
            strip = [lo, d0, hi, d1] if axis == "x" else [d0, lo, d1, hi]
            if _box_hit(strip, st):
                b0, b1 = (st[0], st[2]) if axis == "x" else (st[1], st[3])
                lo, hi = max((lo, min(hi, b0)), (max(lo, b1), hi), key=lambda p: p[1] - p[0])
                lo, hi = lo + _STAIR_GAP - margin, hi - _STAIR_GAP + margin
        out[f"{a}|{b}"] = (a, b, axis, coord, lo, hi, yaw)
    return out


def _house_rects(rng_vals, floor):
    """Layout dict of one floor from the drawn numbers v (see sample_house)."""
    v = rng_vals
    W, D, sw, stw = v["W"], v["D"], v["sw"], v["stw"]
    hx, mx = ((0.0, sw), (sw, W)) if v["west"] else ((W - sw, W), (0.0, W - sw))
    sx = (0.0, stw) if v["west"] else (W - stw, W)
    R = lambda x, y0, y1: [float(x[0]), float(y0), float(x[1]), float(y1)]
    top = v["gap"] + v["length"]
    if floor == 0:
        h, t = v["hall_d"], v["toilet_d"]
        hub, rooms = R(hx, 0.0, h), [("toilet", R(hx, h, h + t)), ("kitchen", R(hx, h + t, D))]
        if v["ensuite"]:
            f = v["fe"] * D
            rooms = [("living_front", R(mx, 0.0, f)), ("living_back", R(mx, f, D))] + rooms
            plan = [["corridor", "living_front"], ["corridor", "toilet"], ["living_back", "kitchen"], ["living_front", "living_back"]]
            types = {"corridor|toilet": "hinged", "living_front|living_back": "sliding"}
        else:
            rooms = [("living", R(mx, 0.0, D))] + rooms
            plan, types = [["corridor", "living"], ["corridor", "toilet"], ["living", "kitchen"]], {"corridor|toilet": "hinged"}
        stairs = [sx[0], v["gap"], sx[1], top]
    else:
        b3, bd = v["bed3_d"], v["bath_d"]
        hub = R(hx, b3, D - bd)
        f = v["f1"] * D
        rooms = [("bed3", R(hx, 0.0, b3)), ("bath", R(hx, D - bd, D)), ("bed1", R(mx, 0.0, f)), ("bed2", R(mx, f, D))]
        plan, types = [["corridor", r] for r in ("bed1", "bed2", "bed3", "bath")], {}
        stairs = [sx[0], max(v["gap"], b3), sx[1], top]
    return {"bounds": [0.0, 0.0, float(W), float(D)], "corridor": hub, "rooms": [{"id": i, "rect": r, "side": None} for i, r in rooms],
            "kind": "house", "floor": floor, "strip_side": "west" if v["west"] else "east", "stairs": [float(x) for x in stairs],
            "plan": plan, "door_types": types}


def _house_ok(lay, v, h, cm, wmax):
    """True if the floor layout keeps every rule of sample_house (stairs, room sizes, shared walls, doors that fit)."""
    sp, st, hub = _spaces(lay), lay["stairs"], lay["corridor"]
    if lay["floor"] == 0:
        if hub[3] - st[3] < h["ground"]["hall_free"] - 1e-9 or sp["kitchen"][3] - sp["kitchen"][1] < h["ground"]["kitchen_min"] - 1e-9:
            return False
        t = sp["toilet"]
        w, d = sorted([t[2] - t[0], t[3] - t[1]])
        if w < 0.85 or d < 1.2:
            return False
    elif hub[3] - st[3] < h["upper"]["step_off"] - 1e-9 or st[3] - st[1] < 0.5:
        return False
    walls = house_walls(lay, cm)
    for a, b in lay["plan"]:
        w = walls.get(f"{a}|{b}")
        s = _shared_wall(sp[a], sp[b])
        if w is None or s[3] - s[2] < h["min_shared"] - 1e-9 or w[5] - w[4] - 2 * cm < wmax:  # a door of the widest kind must fit
            return False
    return True


def sample_house(rng, cfg, floor=0):
    """Layout of one floor of a Dutch terraced house (cfg["house"]; the full spec is in sim/tests/test_houses.py): footprint first
    (W, D, strip side + width, stairs), then both floors' own numbers, in a fixed order, so floors 0 and 1 of one rng share the
    footprint; redrawn until every rule of BOTH floors holds (<= 2000 tries). Returns the layout of `floor`."""
    h = cfg["house"]
    g0, g1, s, cm, wmax = h["ground"], h["upper"], h["stairs"], cfg.get("corner_margin", 0.3), cfg["door"]["width"][1]
    for _ in range(2000):
        v = {"W": _u(rng, h["width"]), "D": _u(rng, h["depth"]), "west": bool(rng.integers(2)), "sw": _u(rng, h["strip_width"]),
             "stw": _u(rng, s["width"]), "length": _u(rng, s["length"]), "gap": _u(rng, s["front_gap"])}
        v.update(toilet_d=_u(rng, g0["toilet_depth"]), hall_f=float(rng.random()), ensuite=bool(rng.random() < g0["p_ensuite"]),
                 fe=_u(rng, g0["ensuite_front"]), bed3_d=_u(rng, g1["bed3_depth"]), bath_d=_u(rng, g1["bath_depth"]), f1=_u(rng, g1["bed1_front"]))
        lo, hi = v["gap"] + v["length"] + g0["hall_free"], v["D"] - v["toilet_d"] - g0["kitchen_min"]
        if hi < lo or v["sw"] - v["stw"] < 0.9:
            continue
        v["hall_d"] = lo + v["hall_f"] * (hi - lo)
        lays = [_house_rects(v, f) for f in (0, 1)]
        if all(_house_ok(lay, v, h, cm, wmax) for lay in lays):
            lay = lays[floor]
            lay["stairs_height"] = float(doors.ROOM_HEIGHT if floor == 0 else h["rail_height"])
            return lay
    raise ValueError("sample_house: no layout in 2000 draws, check the house numbers in the config")


def _passable(layout, d, cfg):
    """True if, with door d fully open (and every other door a plain wall), a robot of robot_radius fits through it into the main part
    of the other space: the biggest fit component (_fit_labels) of each of the door's two spaces is the same one (a narrow
    hinged door swung open can leave < 2 radii, or its panel can wall off the pocket behind it)."""
    res, sp = float(cfg.get("map_resolution", 0.05)), _spaces(layout)
    grid, (ox, oy) = occupancy(layout, [dict(d, state="open", open_frac=1.0)], res)
    lab, main = _fit_labels(grid, res, cfg["robot_radius"]), []
    for sid in d["connects"]:
        r = sp[sid]
        part = lab[grid.shape[0] - 1 - int((r[3] - oy) / res):grid.shape[0] - 1 - int((r[1] - oy) / res) + 1, int((r[0] - ox) / res):int((r[2] - ox) / res) + 1]
        cnt = np.bincount(part[part > 0])
        main.append(int(cnt.argmax()) if cnt.size else 0)
    return main[0] > 0 and main[0] == main[1]


def place_house_doors(rng, layout, cfg, split="train"):
    """Door dicts as place_doors: exactly one per planned pair, on their shared wall (house_walls), forced types from
    layout["door_types"] (spec resampled until it matches, fits and the open door lets the robot through, _passable). A hinged hub
    door always swings into the room (an open panel must not cut the narrow hall / landing beside the stairs)."""
    walls, out = house_walls(layout, cfg["corner_margin"]), []
    for a, b in layout["plan"]:
        wall, forced = walls[f"{a}|{b}"], layout["door_types"].get(f"{a}|{b}")
        axis, coord, yaw = wall[2], wall[3], wall[6]
        for _ in range(1000):
            spec = _sample_spec(rng, cfg, split)
            hit = None if forced and spec["door_type"] != forced else _try_place(rng, spec, wall, [], cfg["corner_margin"])
            if not hit:
                continue
            c, span = hit
            if a == "corridor" and doors.is_hinged_like(spec):
                spec["swing_x"] = 1
            pos = [c, coord] if axis == "x" else [coord, c]
            d = {"id": len(out) + 1, "spec": spec, "connects": [a, b], "pos": [float(pos[0]), float(pos[1])], "yaw": float(yaw)}
            if _passable(layout, d, cfg):
                break
        else:
            raise ValueError(f"no door fits {a}|{b}")
        closed = bool(rng.random() < cfg["p_closed"])
        frac = 0.0 if closed else round(float(rng.uniform(cfg["min_open"], 1.0)), 3)
        out.append(dict(d, state="closed" if closed else "open", open_frac=frac, heldout=bool(doors.is_heldout(spec, cfg["heldout"]))))
    return out


def _house_start(layout, ds, cfg):
    """[x, y] of the robot start: in the hub past the stairs' top end, robot_radius off the hub walls and the stairs, where the
    distance to anything occupied is largest with every door open."""
    from scipy import ndimage
    res, rad, hub, st = float(cfg.get("map_resolution", 0.05)), cfg["robot_radius"], layout["corridor"], layout["stairs"]
    grid, (ox, oy) = occupancy(layout, [dict(d, state="open", open_frac=1.0) for d in ds], res)
    dist = ndimage.distance_transform_edt(grid != 0) * res
    lab = _fit_labels(grid, res, rad)
    part = lab[grid.shape[0] - 1 - int((hub[3] - oy) / res):grid.shape[0] - 1 - int((hub[1] - oy) / res) + 1, int((hub[0] - ox) / res):int((hub[2] - ox) / res) + 1]
    main = int(np.bincount(part[part > 0]).argmax())
    xs, ys = ox + (np.arange(grid.shape[1]) + 0.5) * res, oy + (np.arange(grid.shape[0])[::-1] + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    y0 = max(hub[1] + rad, st[3] + rad)
    ok = (X >= hub[0] + rad) & (X <= hub[2] - rad) & (Y >= y0) & (Y <= hub[3] - rad) & (dist >= rad) & (lab == main)
    if not ok.any():
        raise ValueError("no start spot in the hub")
    score = np.where(ok, dist - 1e-3 * np.hypot(X - (hub[0] + hub[2]) / 2, Y - (y0 + hub[3] - rad) / 2), -np.inf)
    r, c = np.unravel_index(np.argmax(score), score.shape)
    return [float(X[r, c]), float(Y[r, c])]


def _make_house(cfg, rng, seed, split, floor):
    layout = sample_house(rng, cfg, floor)
    ds = place_house_doors(rng, layout, cfg, split)
    start = _house_start(layout, ds, cfg)
    r = cfg["robot_radius"] + 0.2
    cl = place_clutter(rng, layout, ds, cfg, [[start[0] - r, start[1] - r, start[0] + r, start[1] + r]]) if "clutter" in cfg else []
    return {"seed": int(seed), "split": split, "layout": layout, "doors": ds, "clutter": cl, "poses": approach_poses(layout, ds, cfg, cl),
            "map_resolution": float(cfg.get("map_resolution", 0.05)), "start": start}


def after_opening(b, opened, cfg):
    """Copy of building b with the doors in `opened` fully open (state open, open_frac 1.0, as the door step opens them) and the
    approach poses recomputed, so rooms behind them become reachable (round 2 of the tour, Madelief 2026-11-11)."""
    ids = set(opened)
    ds = [dict(d, state="open", open_frac=1.0) if d["id"] in ids else d for d in b["doors"]]
    return dict(b, doors=ds, poses=approach_poses(b["layout"], ds, cfg, b.get("clutter")))


def write_map(b, stem):
    """stem.map.pgm (P5) + stem.map.yaml (Nav2 map_server) of building b, at b["map_resolution"] (default 0.05)."""
    stem = pathlib.Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    res = b.get("map_resolution", 0.05)
    grid, origin = occupancy(b["layout"], b["doors"], res, b.get("clutter"))
    pathlib.Path(f"{stem}.map.pgm").write_bytes(f"P5\n{grid.shape[1]} {grid.shape[0]}\n255\n".encode() + grid.tobytes())
    pathlib.Path(f"{stem}.map.yaml").write_text(f"image: {stem.name}.map.pgm\nresolution: {res}\norigin: [{origin[0]:.4f}, {origin[1]:.4f}, 0.0]\n"
                                                "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\nmode: trinary\n")


def write(b, out_dir, name):
    """Write name.sdf, name.building.json (b + labels per door), name.map.pgm (P5) and name.map.yaml (Nav2 map_server) to out_dir.
    The map resolution is b["map_resolution"] (default 0.05)."""
    res = b.get("map_resolution", 0.05)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.sdf").write_text(building_sdf(b, name))
    meta = dict(b, doors=[dict(d, labels=door_labels(d)) for d in b["doors"]])
    (out / f"{name}.building.json").write_text(json.dumps(meta, indent=2) + "\n")
    write_map(b, out / name)
