"""Clutter and look-alikes for door room v2: a pool of visual-only static models (one label id each),
parked under the floor until a scene needs them, then moved with set_pose. Randomness: numpy Generator.

Object frame: origin = centre of the item's bounding box, yaw 0 = front facing -x (toward the camera),
size = [depth_x, width_y, height_z]. Floor items stand on the floor (z = height / 2); wall items hang
with their back flat on the wall face."""
import math
import numpy as np

PARK = [0.0, 0.0, -20.0]  # unused items wait here, under the floor


def box(pos, size, rgb):
    return ("box", pos, size, rgb)


def cyl(pos, r, length, rgb):
    return ("cylinder", pos, (r, length), rgb)


def ball(pos, r, rgb):
    return ("sphere", pos, (r,), rgb)


def item(name, id, role, kind, place, size, parts, recolour=False):
    """parts: (shape, centre, dims, rgb) with centre = (x, y, z), z measured up from the item's bottom."""
    return {"name": name, "id": id, "role": role, "kind": kind, "place": place, "size": size,
            "recolour": recolour, "parts": parts}


GREY = (0.6, 0.6, 0.6)
WOOD, DARK, LIGHT, METAL = (0.55, 0.38, 0.22), (0.25, 0.17, 0.1), (0.8, 0.72, 0.6), (0.7, 0.7, 0.72)


def shape_item(name, id, kind, size):
    d, w, h = size
    dims = {"box": tuple(size), "cylinder": (d / 2, h), "sphere": (d / 2,)}[kind]
    return item(name, id, "clutter", kind, "floor", size, [(kind, (0, 0, h / 2), dims, GREY)], recolour=True)


def table():
    parts = [box((0, 0, 0.74), (0.7, 1.2, 0.04), WOOD)]
    parts += [box((sx * 0.3, sy * 0.55, 0.36), (0.05, 0.05, 0.72), DARK) for sx in (-1, 1) for sy in (-1, 1)]
    return item("table", 120, "clutter", "table", "floor", [0.7, 1.2, 0.76], parts)


def chair(name, id, s, rgb):
    """Seat, 4 legs and a backrest at the back (+x). s = scale of the 0.45 m base."""
    d, h = 0.45 * s, 0.9 * s
    seat = 0.45 * s
    parts = [box((0, 0, seat), (d, d, 0.04), rgb)]
    parts += [box((sx * (d / 2 - 0.025), sy * (d / 2 - 0.025), seat / 2 - 0.02), (0.04, 0.04, seat - 0.04), DARK)
              for sx in (-1, 1) for sy in (-1, 1)]
    parts += [box((d / 2 - 0.02, 0, (seat + h) / 2), (0.04, d, h - seat), rgb)]
    return item(name, id, "clutter", "chair", "floor", [d, d, h], parts)


def plant():
    return item("plant", 123, "clutter", "plant", "floor", [0.5, 0.5, 0.75],
                [cyl((0, 0, 0.15), 0.15, 0.3, (0.7, 0.35, 0.2)), ball((0, 0, 0.5), 0.25, (0.15, 0.5, 0.18))])


def bookcase():
    d, w, h = 0.3, 0.8, 1.8
    parts = [box((0, sy * (w / 2 - 0.01), h / 2), (d, 0.02, h), WOOD) for sy in (-1, 1)]
    parts += [box((d / 2 - 0.005, 0, h / 2), (0.01, w, h), DARK)]
    parts += [box((0, 0, z), (d, w - 0.04, 0.02), WOOD) for z in (0.01, 0.45, 0.9, 1.35, 1.79)]
    return item("bookcase", 124, "clutter", "bookcase", "floor", [d, w, h], parts)


def box_stack():
    cardboard = (0.7, 0.55, 0.35)
    parts = [box((0, 0, 0.15), (0.6, 0.6, 0.3), cardboard), box((0.02, 0.03, 0.45), (0.5, 0.5, 0.3), (0.65, 0.5, 0.3)),
             box((-0.02, 0, 0.75), (0.4, 0.4, 0.3), (0.75, 0.6, 0.4))]
    return item("box_stack", 125, "clutter", "box_stack", "floor", [0.6, 0.6, 0.9], parts)


def two_door_unit(name, id, kind, size, body_rgb, door_rgb, door_h, door_z, handle):
    """Carcass + 2 doors on the front (-x). handle(side, front_x, z) -> parts for that door."""
    d, w, h = size
    parts = [box((0.02, 0, h / 2), (d - 0.04, w, h), body_rgb)]
    fx = -d / 2 + 0.02
    for s in (-1, 1):
        parts.append(box((fx + 0.01, s * (w / 4 + 0.002), door_z), (0.02, w / 2 - 0.01, door_h), door_rgb))
        parts += handle(s, fx, door_z)
    return item(name, id, "lookalike", kind, "floor", size, parts)


def cabinet():
    def knob(s, fx, z):
        return [ball((fx - 0.0, s * 0.05, z + 0.28), 0.02, METAL)]
    return two_door_unit("cabinet", 140, "cabinet", [0.4, 1.0, 0.9], LIGHT, (0.85, 0.8, 0.7), 0.76, 0.45, knob)


def wardrobe():
    def bar(s, fx, z):
        return [box((fx - 0.012, s * 0.05, z), (0.025, 0.02, 0.3), METAL)]
    return two_door_unit("wardrobe", 141, "wardrobe", [0.55, 1.0, 1.9], WOOD, (0.6, 0.43, 0.27), 1.8, 0.95, bar)


def drawers():
    d, w, h, n = 0.45, 0.8, 0.9, 4
    parts = [box((0.02, 0, h / 2), (d - 0.04, w, h), LIGHT)]
    fh = (h - 0.06) / n
    for i in range(n):
        z = 0.03 + fh * (i + 0.5)
        parts.append(box((-d / 2 + 0.03, 0, z), (0.02, w - 0.06, fh - 0.015), (0.85, 0.8, 0.7)))
        parts.append(box((-d / 2 + 0.008, 0, z + 0.03), (0.016, 0.25, 0.02), METAL))
    return item("drawers", 142, "lookalike", "drawers", "floor", [d, w, h], parts)


def picture_frame(name, id, w, h, picture):
    parts = [box((0.005, 0, h / 2), (0.02, w, h), DARK), box((-0.01, 0, h / 2), (0.01, w - 0.08, h - 0.08), picture)]
    return item(name, id, "lookalike", "frame", "wall", [0.03, w, h], parts)


def whiteboard():
    parts = [box((0.005, 0, 0.35), (0.02, 1.0, 0.7), (0.5, 0.5, 0.52)), box((-0.01, 0, 0.35), (0.01, 0.95, 0.65), (0.95, 0.95, 0.95))]
    return item("whiteboard", 147, "lookalike", "whiteboard", "wall", [0.03, 1.0, 0.7], parts)


BOXES = [(0.15, 0.15, 0.15), (0.3, 0.3, 0.3), (0.5, 0.4, 0.3), (0.25, 0.6, 0.2), (0.8, 0.5, 0.4), (0.4, 0.4, 0.8), (0.2, 0.2, 0.5), (0.6, 0.6, 0.15)]
CYLS = [(0.2, 0.2, 0.4), (0.4, 0.4, 0.3), (0.3, 0.3, 0.8), (0.15, 0.15, 0.25)]
SPHERES = [0.2, 0.35, 0.5]

POOL = ([shape_item(f"box_{i + 1}", 100 + i, "box", list(s)) for i, s in enumerate(BOXES)]
        + [shape_item(f"cylinder_{i + 1}", 110 + i, "cylinder", list(s)) for i, s in enumerate(CYLS)]
        + [shape_item(f"sphere_{i + 1}", 115 + i, "sphere", [d, d, d]) for i, d in enumerate(SPHERES)]
        + [table(), chair("chair_1", 121, 1.0, (0.3, 0.45, 0.7)), chair("chair_2", 122, 1.15, (0.7, 0.3, 0.25)), plant(), bookcase(), box_stack(),
           cabinet(), wardrobe(), drawers(),
           picture_frame("frame_1", 143, 0.4, 0.5, (0.35, 0.55, 0.7)), picture_frame("frame_2", 144, 0.7, 0.5, (0.7, 0.6, 0.35)),
           picture_frame("frame_3", 145, 0.3, 0.4, (0.5, 0.7, 0.5)), whiteboard()])


def _geometry(shape, dims):
    if shape == "box":
        return f"<box><size>{dims[0]} {dims[1]} {dims[2]}</size></box>"
    if shape == "cylinder":
        return f"<cylinder><radius>{dims[0]}</radius><length>{dims[1]}</length></cylinder>"
    return f"<sphere><radius>{dims[0]}</radius></sphere>"


def _model(p):
    h = p["size"][2]
    visuals = []
    for i, (shape, (x, y, z), dims, rgb) in enumerate(p["parts"]):
        name = "v" if p["recolour"] else f"p{i}"
        c = " ".join(f"{v:.3f}" for v in rgb)
        visuals.append(f'      <visual name="{name}"><pose>{x} {y} {z - h / 2:.4f} 0 0 0</pose><geometry>{_geometry(shape, dims)}</geometry>'
                       f'<material><diffuse>{c} 1</diffuse><ambient>{c} 1</ambient></material></visual>')
    px, py, pz = PARK
    return (f'    <model name="{p["name"]}">\n      <static>true</static>\n      <pose>{px} {py} {pz} 0 0 0</pose>\n      <link name="link">\n'
            + "\n".join(visuals) + "\n      </link>\n"
            f'      <plugin filename="gz-sim-label-system" name="gz::sim::systems::Label"><label>{p["id"]}</label></plugin>\n    </model>')


def world_sdf(pool):
    """SDF <model> elements for every pool item: static, visuals only, parked, one label id each."""
    return "\n".join(_model(p) for p in pool) + "\n"


def footprint(placement, p):
    """The 4 (x, y) corners of the item's depth x width rectangle, rotated by yaw around its position."""
    x, y = placement["position"][:2]
    c, s = math.cos(placement["yaw"]), math.sin(placement["yaw"])
    hd, hw = p["size"][0] / 2, p["size"][1] / 2
    return [(x + c * dx - s * dy, y + s * dx + c * dy) for dx, dy in ((-hd, -hw), (hd, -hw), (hd, hw), (-hd, hw))]


def _aabb(corners):
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    return min(xs), max(xs), min(ys), max(ys)


def _overlap(a, b, grow=0.0):
    return a[0] < b[1] + grow and b[0] - grow < a[1] and a[2] < b[3] + grow and b[2] - grow < a[3]


def _floor_spot(rng, p, cfg, taken, tries=50):
    """(x, y, yaw) for a floor item inside the zone, clear of keep_out and of `taken` AABBs, or None."""
    zx, zy = cfg["zone"]["x"], cfg["zone"]["y"]
    keep = [(k["x"][0], k["x"][1], k["y"][0], k["y"][1]) for k in cfg["keep_out"]]
    d, w = p["size"][0], p["size"][1]
    for _ in range(tries):
        if rng.random() < cfg["p_against_wall"]:
            yaw, x, hy = 0.0, zx[1] - d / 2, w / 2
            hx = d / 2
        else:
            yaw = float(rng.uniform(-math.pi, math.pi))
            c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
            hx, hy = c * d / 2 + s * w / 2, s * d / 2 + c * w / 2
            x = None
        if hy > (zy[1] - zy[0]) / 2 or hx > zx[1] - zx[0]:
            continue
        if x is None:
            x = float(rng.uniform(zx[0] + hx, zx[1] - hx))
        y = float(rng.uniform(zy[0] + hy, zy[1] - hy))
        bb = _aabb(footprint({"position": [x, y], "yaw": yaw}, p))
        if any(_overlap(bb, k) for k in keep) or any(_overlap(bb, t, cfg["gap"]) for t in taken):
            continue
        return x, y, yaw
    return None


def _wall_spot(rng, p, cfg, taken, tries=50):
    """(x, y, z) for a wall item within the wall's y / z range, off the keep_out_y spans and other wall items."""
    w = cfg["wall"]
    d, wd, h = p["size"]
    if wd > w["y"][1] - w["y"][0] or h > w["z"][1] - w["z"][0]:
        return None
    for _ in range(tries):
        y = float(rng.uniform(w["y"][0] + wd / 2, w["y"][1] - wd / 2))
        z = float(rng.uniform(w["z"][0] + h / 2, w["z"][1] - h / 2))
        lo, hi = y - wd / 2, y + wd / 2
        if any(lo < b and a < hi for a, b in w["keep_out_y"]) or any(lo < b and a < hi for a, b in taken):
            continue
        return w["x"] - d / 2, y, z
    return None


def place(rng, pool, cfg):
    """Pick items for one scene and place them. Returns {name: {"position", "yaw", "colour", "in_scene"}} for every
    pool item; items not used stay at PARK. A picked item that finds no spot is skipped for the next one."""
    out = {p["name"]: {"position": list(PARK), "yaw": 0.0, "colour": None, "in_scene": False} for p in pool}
    for place_kind, key in (("floor", "floor_count"), ("wall", "wall_count")):
        group = [p for p in pool if p["place"] == place_kind]
        want = int(rng.integers(cfg[key][0], cfg[key][1] + 1))
        taken, n = [], 0
        for i in rng.permutation(len(group)):
            if n >= want:
                break
            p = group[i]
            if place_kind == "floor":
                spot = _floor_spot(rng, p, cfg, taken)
                if spot is None:
                    continue
                x, y, yaw = spot
                pos = [x, y, p["size"][2] / 2]
                taken.append(_aabb(footprint({"position": pos, "yaw": yaw}, p)))
            else:
                spot = _wall_spot(rng, p, cfg, taken)
                if spot is None:
                    continue
                pos, yaw = list(spot), 0.0
                taken.append((pos[1] - p["size"][1] / 2, pos[1] + p["size"][1] / 2))
            colour = [float(c) for c in rng.uniform(0, 1, 3)] if p["recolour"] else None
            out[p["name"]] = {"position": [float(v) for v in pos], "yaw": float(yaw), "colour": colour, "in_scene": True}
            n += 1
    return out


def camera_blocked(position, placements, pool, margin):
    """True if the camera is inside an in-scene item (footprint grown by margin, height range grown too)."""
    for p in pool:
        pl = placements[p["name"]]
        if not pl["in_scene"]:
            continue
        dx, dy = position[0] - pl["position"][0], position[1] - pl["position"][1]
        c, s = math.cos(pl["yaw"]), math.sin(pl["yaw"])
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        half_h = p["size"][2] / 2
        if (abs(lx) <= p["size"][0] / 2 + margin and abs(ly) <= p["size"][1] / 2 + margin
                and abs(position[2] - pl["position"][2]) <= half_h + margin):
            return True
    return False


def visible_fraction(seg, seg_clear, attrs):
    """{object: share of its pixels (its id + ids of its parts) still visible with the clutter in} for every object
    with a class. seg_clear = the same view with all clutter parked. None when it isn't in the clear view."""
    objs = attrs["objects"]
    out = {}
    for name, o in objs.items():
        if o.get("class") is None:
            continue
        ids = [o["id"]] + [q["id"] for q in objs.values() if q.get("part_of") == name]
        clear = int(np.isin(seg_clear, ids).sum())
        out[name] = None if clear == 0 else float(min(1.0, max(0.0, np.isin(seg, ids).sum() / clear)))
    return out
