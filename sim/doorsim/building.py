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
+ 3 glass, + 4 handle (back), + 5..7 hinge knuckles (as v5 with k = i), so ids stay < 256 for i <= MAX_DOORS. The v3 clutter pool
(ids 100+) is left out. Randomness: numpy Generator."""
import json, math, pathlib, re
import numpy as np

from doorsim import doors

WALL_T = 0.1
MAX_DOORS = 24  # base 10 * id + 7 < 256 -> id <= 24
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
def approach_poses(layout, doors_, cfg):
    """One pose per door side and distance in cfg approach.distances: on the door's normal line, `dist` from the door centre into
    that space (clamped to stay robot_radius + WALL_T / 2 inside it), facing the door centre; reachable = space reachable from the
    corridor through open doors. clear = no occupied map cell (occupancy() at cfg map_resolution) within robot_radius: a blocked pose
    (e.g. on an open door panel) is moved further back along the normal in 0.1 m steps (staying inside the space), then closer down to
    0.5 m; dist = the distance used; clear False only if nothing fits."""
    sp, m = _spaces(layout), cfg["robot_radius"] + WALL_T / 2
    res = float(cfg.get("map_resolution", 0.05))
    grid, origin = occupancy(layout, doors_, res)
    rc = int(math.ceil(cfg["robot_radius"] / res))

    def free(x, y):
        c, row = int((x - origin[0]) / res), grid.shape[0] - 1 - int((y - origin[1]) / res)
        return not (grid[max(0, row - rc):row + rc + 1, max(0, c - rc):c + rc + 1] == 0).any()

    reach, grow = {"corridor"}, True
    while grow:
        grow = False
        for d in doors_:
            a, b = d["connects"]
            if d["state"] == "open" and (a in reach) != (b in reach):
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


# --- occupancy map ---
def occupancy(layout, doors_, res):
    """(grid uint8 H x W, origin [x, y]): 0 = occupied, 254 = free, row 0 = north edge, origin = world xy of the bottom-left cell.
    Free = inside a space; occupied = outside, the walls, the doorway of a closed door and the panel of an open one where it stands
    (hinged: the panel's centre line turned by open_frac * the v5 open angle about its hinge; sliding: shifted along its path)."""
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

    def mark(x, y):
        col, row = int((x - ox) / res), h - 1 - int((y - oy) / res)
        if 0 <= col < w and 0 <= row < h:
            free[row, col] = False

    for d in doors_:
        if d["state"] == "closed":
            fill(_opening_rect(d), False)
            continue
        spec, g = d["spec"], doors._layout(d["spec"], SLOT)
        if spec["door_type"] == "hinged":
            th = g["open"] * d["open_frac"]
            hy, hx = spec["hinge_y"], g["hinge"][0]
            y0 = g["hinge"][1]
            for t in np.linspace(0, g["pw"], int(g["pw"] / res * 4) + 2):  # centre line of the panel, hinge -> lock edge
                mark(*_world(d, hx + hy * t * math.sin(th), y0 - hy * t * math.cos(th)))
        else:
            s, shift = spec["slide_y"], g["open"] * d["open_frac"]
            for t in np.linspace(-g["pw"] / 2, g["pw"] / 2, int(g["pw"] / res * 2) + 2):
                for dx in (-0.02, 0.0, 0.02):
                    mark(*_world(d, g["xc"] + dx, g["y"] + s * (t + shift)))
    grid = np.where(free, 254, 0).astype(np.uint8)
    return grid, [float(ox), float(oy)]


# --- the world file ---
def _names(i):
    return {"door": f"door_{i}_panel", "handle_front": f"door_{i}_handle_front", "handle_back": f"door_{i}_handle_back",
            "glass": f"door_{i}_glass", "hinge": f"door_{i}_hinge_{{}}"}


def door_labels(d):
    """{part: label id} of one door (see the module docstring)."""
    base, spec = 10 * d["id"], d["spec"]
    out = {"panel": base + 1, "handle_front": base + 2, "handle_back": base + 4}
    if spec["glass"]:
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
    if spec["glass"]:
        parts.append(_place(doors._glass_model(SLOT, i, g, nm), d))
    parts += [_place(doors._handle_model(SLOT, i, spec, g, f, nm), d) for f in ("front", "back")]
    if doors._has_hinges(spec):
        parts += [_place(doors._hinge_model(SLOT, i, spec, g, k, nm), d) for k in range(3)]
    vis = doors._door_wall_visuals(spec, g, f"{i}")
    if vis:  # origin on the wall centre plane at y = 0 of the local frame
        pose = doors._pose(*_world(d, doors.WALL_X, 0.0), 0, 0, 0, d["yaw"])
        parts.append(f'    <model name="door_{i}_frame">\n      <static>true</static>\n      <pose>{pose}</pose>\n      <link name="link">\n'
                     + "".join("        " + v + "\n" for v in vis) + "      </link>\n    </model>\n")
    return "".join(parts)


def building_sdf(b, name):
    """Full SDF text: plugins / scene / sun of door_room_v3.sdf, floor, ceiling, walls with door openings, one ceiling light per space,
    every door's models, the v3 camera rig (at the corridor centre; the clutter pool is left out: its ids clash with the doors')."""
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
    rig = v3[v3.index("    <!-- Fixed sensor rig"):v3.index("    <!-- clutter pool")]
    c = lay["corridor"]
    rig = rig.replace("<pose>0 0 1.2 0 0 0</pose>", f"<pose>{doors._pose((c[0] + c[2]) / 2, (c[1] + c[3]) / 2, 1.2)} 0 0 0</pose>", 1)
    parts = [f'<?xml version="1.0" ?>\n<!-- Building {name} (seed {b["seed"]}, split {b["split"]}): generated by sim/building/make_buildings.py -->\n', head, lights, "\n",
             floor, ceiling, walls] + ["\n" + _door_models(d) for d in b["doors"]]
    return "".join(parts) + "\n" + rig + "  </world>\n</sdf>\n"


# --- whole building ---
def make_building(cfg, seed, split="train"):
    """{"seed", "split", "layout", "doors", "poses", "map_resolution"}: deterministic for a seed."""
    rng = np.random.default_rng(seed)
    layout = sample_layout(rng, cfg)
    ds = place_doors(rng, layout, cfg, split)
    return {"seed": int(seed), "split": split, "layout": layout, "doors": ds, "poses": approach_poses(layout, ds, cfg),
            "map_resolution": float(cfg.get("map_resolution", 0.05))}


def write(b, out_dir, name):
    """Write name.sdf, name.building.json (b + labels per door), name.map.pgm (P5) and name.map.yaml (Nav2 map_server) to out_dir.
    The map resolution is b["map_resolution"] (default 0.05)."""
    res = b.get("map_resolution", 0.05)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.sdf").write_text(building_sdf(b, name))
    meta = dict(b, doors=[dict(d, labels=door_labels(d)) for d in b["doors"]])
    (out / f"{name}.building.json").write_text(json.dumps(meta, indent=2) + "\n")
    grid, origin = occupancy(b["layout"], b["doors"], res)
    (out / f"{name}.map.pgm").write_bytes(f"P5\n{grid.shape[1]} {grid.shape[0]}\n255\n".encode() + grid.tobytes())
    (out / f"{name}.map.yaml").write_text(f"image: {name}.map.pgm\nresolution: {res}\norigin: [{origin[0]:.4f}, {origin[1]:.4f}, 0.0]\n"
                                          "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\nmode: trinary\n")
