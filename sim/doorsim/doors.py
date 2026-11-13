"""Door variety (door room v4): random door specs, worlds of two doors each, ground-truth attributes and the world file.

Doors are fixed per world file (Gazebo can't change a joint while running), so variety = many worlds, each with its own two
doors in the same wall. Frames: world x toward the wall, y left, z up. "front" = camera at x < WALL_X looking +x (right = -y);
"back" = camera at x > WALL_X looking -x (right = +y). Randomness: numpy Generator."""
import math
import pathlib
import numpy as np

from doorsim import clutter, export

ROOM = pathlib.Path(__file__).resolve().parents[1] / "door_room"
WALL_X = 3.0  # wall centre plane, 0.1 thick: faces at 2.95 / 3.05
SLOTS = ("a", "b")
SLOT_Y = {"a": -1.0, "b": 1.0}
ROOM_HEIGHT = 2.7
TRIM = {"none": (0.0, 0.0), "thin": (0.07, 0.015), "thick": (0.12, 0.03)}  # frame_style -> (width, proud) in m
WALL_RGB, CEILING_RGB, METAL_RGB = (0.85, 0.83, 0.78), (0.9, 0.9, 0.88), (0.15, 0.15, 0.15)


# --- random doors and worlds ---
def _pick(rng, options):
    """Key of a {key: {"p": weight, ...}} or {key: weight} dict, drawn by weight."""
    keys = list(options)
    p = np.array([o["p"] if isinstance(o, dict) else o for o in options.values()], float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


def _shade(rng, family):
    lo, hi = family["rgb"]
    return [round(float(c), 3) for c in rng.uniform(lo, hi)]


def sample_door(rng, cfg, slot, force=None):
    """Spec of one door (plain Python values, JSON-able). `force` = optional {glass, door_type, handle_type} set instead of
    drawn (same random draws either way, hinged handle_type only). Hinged: hinge_y = which edge the hinge is on (-1 / +1),
    swing_x = which way it swings (-1 toward -x, +1 toward +x). Sliding: hangs_on a wall face, slide_y away from the other doorway."""
    sliding = bool(rng.random() < cfg["p_sliding"])
    glass = bool(rng.random() < cfg["p_glass"])
    force = force or {}
    if "door_type" in force:
        sliding = force["door_type"] == "sliding"
    if "glass" in force:
        glass = bool(force["glass"])
    if sliding:
        handle = _pick(rng, cfg["sliding_handle"]) if "sliding_handle" in cfg else "flush_pull"  # new key: draws only when present
    else:
        handle = _pick(rng, cfg["handle"])
    if not sliding and "handle_type" in force:
        handle = force["handle_type"]
    spec = {"slot": slot, "door_type": "sliding" if sliding else "hinged", "glass": glass, "handle_type": handle,
            "width": round(float(rng.uniform(*cfg["width"])), 3), "height": round(float(rng.uniform(*cfg["height"])), 3),
            "frame_style": _pick(rng, cfg["frame_style"])}
    colour = _shade(rng, cfg["colours"][_pick(rng, cfg["colours"])])
    spec["colour"] = colour
    spec["frame_colour"] = _shade(rng, cfg["glass_frame"][_pick(rng, cfg["glass_frame"])]) if glass else list(colour)
    trim = cfg["trim"][_pick(rng, cfg["trim"])]
    spec["trim_colour"] = _shade(rng, trim) if "rgb" in trim else list(colour)  # "match" = the door's colour
    if sliding:
        spec["hangs_on"] = "back" if rng.random() < cfg["p_hangs_back"] else "front"
        spec["slide_y"] = -1 if SLOT_Y[slot] < 0 else 1
    else:
        spec["hinge_y"], spec["swing_x"] = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
    if "glazing" in cfg:  # v8: after every other draw; glass doors always "none" (the draw is still made)
        gl = _pick(rng, cfg["glazing"])
        spec["glazing"] = "none" if glass else gl
    return spec


def face_handle(spec, face):
    """Handle type on the `face` ("front" | "back"): push_plate doors have a plate on the push face and a pull bar on the pull face."""
    if spec["handle_type"] == "push_plate":
        return "push_plate" if seen_from(spec, face)["opening"] == "push" else "pull_bar"
    return spec["handle_type"]


def has_pane(spec):
    """True if the door has a glass pane model: a glass door or a partly glazed (strip / panel) door."""
    return bool(spec["glass"]) or spec.get("glazing", "none") != "none"


def glass_rect(spec, g):
    """Glass pane in the panel's link frame: (dy, dz, w, h) = centre offset along y and z, size. Deterministic from the spec."""
    pw, ph = g["pw"], g["ph"]
    kind = "none" if spec["glass"] else spec.get("glazing", "none")
    frac = (round(spec["width"] * 1000) % 9) / 8  # 0..1, a stable pseudo-random number per door (no rng draw)
    if kind == "strip":
        w, h = 0.12 + 0.08 * frac, 0.6 * ph
        lock = 1 if g["lock_y"] > g["y"] else -1
        return (lock * (pw / 2 - 0.1 - w / 2), 0.05, w, h)
    if kind == "panel":
        h = (0.5 + 0.4 * frac) * ph / 2
        return (0.0, ph / 2 - 0.1 - h / 2, pw - 0.3, h)
    return (0.0, 0.02, pw - 0.2, ph - 0.24)


def seen_from(spec, side):
    """What the camera on `side` sees of the door's operation (push / pull and left / right flip with the side)."""
    flip = side == "back"
    if spec["door_type"] == "hinged":
        push = (spec["swing_x"] > 0) != flip
        right = (spec["hinge_y"] < 0) != flip
        return {"opening": "push" if push else "pull", "hinge_side": "right" if right else "left"}
    return {"opening": "slide", "slide_direction": "left" if (spec["slide_y"] > 0) != flip else "right"}


def is_heldout(spec, combos):
    """True if the spec matches every key / value of any held-out combo."""
    return any(all(spec.get(k) == v for k, v in c.items()) for c in combos)


def _draw(rng, cfg, slot, combos, heldout, force=None):
    for _ in range(1000):
        spec = sample_door(rng, cfg, slot, force)
        if is_heldout(spec, combos) == heldout:
            return spec
    raise ValueError(f"no door with heldout={heldout} in 1000 draws: check the heldout combos")


def sample_world(rng, wcfg, heldout=False):
    """{"doors": {slot: spec}, "doors_heldout": {slot: bool}}. A normal world has no held-out door; a held-out world has
    one door (random slot) that is a held-out combo, the other door is normal."""
    combos = wcfg["heldout"]
    must = SLOTS[int(rng.integers(len(SLOTS)))] if heldout else None
    doors = {slot: _draw(rng, wcfg["door"], slot, combos, slot == must) for slot in SLOTS}
    return {"doors": doors, "doors_heldout": {slot: is_heldout(d, combos) for slot, d in doors.items()}}


def _balanced_world(rng, wcfg, held, deal, sp, j):
    """Like sample_world, but door types come from `deal` (normal doors) or balance.heldout_cells[j % len] (held-out door)."""
    combos = wcfg["heldout"]
    must = SLOTS[int(rng.integers(len(SLOTS)))] if held else None
    doors = {}
    for slot in SLOTS:
        if slot == must:
            hc = wcfg["balance"]["heldout_cells"]
            force = hc[j % len(hc)]
        else:
            force = deal(sp)
        doors[slot] = _draw(rng, wcfg["door"], slot, combos, slot == must, force)
    return {"doors": doors, "doors_heldout": {slot: is_heldout(d, combos) for slot, d in doors.items()}}


def make_worlds(wcfg):
    """All worlds of a config, deterministic: `count` normal worlds (split by world), then `heldout_count` held-out worlds (test).
    Config keys `out` (version in the world names, default v4) and `door_stop` (push / pull cues on hinged doors, default false)
    change no random draw: the same seed gives the same doors."""
    out_name, stop = wcfg.get("out", "v4"), bool(wcfg.get("door_stop", False))
    rng = np.random.default_rng(wcfg["seed"])
    split = {i: sp for sp, idx in export.split_scenes(range(wcfg["count"]), tuple(wcfg["split_ratios"]), wcfg["seed"]).items() for i in idx}
    bal = wcfg.get("balance")
    decks = {}  # split -> door types still to deal (balance)

    def deal(sp):
        if not decks.get(sp):
            cycle = [c for c in bal["cells"] for _ in range(int(c["w"]))]
            decks[sp] = [cycle[j] for j in rng.permutation(len(cycle))]
        return {k: v for k, v in decks[sp].pop(0).items() if k != "w"}

    out = []
    for i in range(wcfg["count"] + wcfg["heldout_count"]):
        held = i >= wcfg["count"]
        sp = "test" if held else split[i]
        if bal:
            world = _balanced_world(rng, wcfg, held, deal, sp, i - wcfg["count"])
        else:
            world = sample_world(rng, wcfg, heldout=held)
        for d in world["doors"].values():
            d["door_stop"] = stop
        world = {"name": f"door_room_{out_name}_{i:03d}", "split": sp, "heldout": held, **world}
        if wcfg.get("window"):  # own rng stream: the doors stay the same as without a window
            world["window"] = sample_window(np.random.default_rng([wcfg["seed"], 7000 + i]), wcfg["window"], world)
        out.append(world)
    return out


# --- windows (door room v7) ---
def _free_stretches(world, margin):
    """Free y stretches of the door wall: [-3 + margin, 3 - margin] minus every door opening + trim + margin and sliding path + margin."""
    blocked = []
    for slot, spec in world["doors"].items():
        g = _layout(spec, slot)
        blocked.append((g["y"] - g["w"] / 2 - g["tw"] - margin, g["y"] + g["w"] / 2 + g["tw"] + margin))
        if spec["door_type"] == "sliding":
            blocked.append((g["path"][0] - margin, g["path"][1] + margin))
    out, at = [], -3.0 + margin
    for a, b in sorted(blocked):
        if a > at:
            out.append((at, min(a, 3.0 - margin)))
        at = max(at, b)
    if at < 3.0 - margin:
        out.append((at, 3.0 - margin))
    return [(a, b) for a, b in out if b > a]


def sample_window(rng, wcfg_window, world):
    """Window spec (plain Python values, JSON-able) in a free stretch of the door wall, or None (p, or no stretch fits min_width).
    The span width + 2 * frame_width stays 0.002 m inside the stretch, so the 3-decimal rounding never touches a blocked edge."""
    c = wcfg_window
    has = rng.random() < c.get("p", 1.0)
    fw = round(float(rng.uniform(*c["frame_width"])), 3)
    stretches = [(a, b) for a, b in _free_stretches(world, c["margin"]) if b - a - 2 * fw - 0.002 >= c["min_width"]]
    if not has or not stretches:
        return None
    a, b = stretches[int(rng.integers(len(stretches)))]
    width = math.floor(min(float(rng.uniform(*c["width"])), b - a - 2 * fw - 0.002) * 1000) / 1000
    lo, hi = a + width / 2 + fw, b - width / 2 - fw
    y = min(max(round(float(rng.uniform(lo, hi)), 3), math.ceil(lo * 1000) / 1000), math.floor(hi * 1000) / 1000)
    low = bool(rng.random() < c["sill"]["low"]["p"])
    sill = round(float(rng.uniform(*c["sill"]["low" if low else "normal"]["z"])), 3)
    top = round(min(float(rng.uniform(*c["top"])), ROOM_HEIGHT - 0.1), 3)
    return {"y": y, "width": width, "sill": sill, "top": top, "frame_width": fw, "frame_colour": _shade(rng, c["frame"][_pick(rng, c["frame"])]),
            "mullions": _pick(rng, c["mullions"]),
            "handle": (("front", "back")[int(rng.integers(2))] if rng.random() < c["p_handle"] else None), "low": low}


def _window_wall(parts, win):
    """Split the wall box that holds the window into boxes around the hole (outer frame size): left, right, below the sill, above the top."""
    y0, y1 = win["y"] - win["width"] / 2 - win["frame_width"], win["y"] + win["width"] / 2 + win["frame_width"]
    z0, z1 = max(0.0, win["sill"] - win["frame_width"]), win["top"] + win["frame_width"]
    out = []
    for part in parts:
        n, rgb, bx0, bx1, by0, by1, bz0, bz1 = part
        if not (n.startswith("wall_") and by0 <= y0 and y1 <= by1):
            out.append(part)
            continue
        out += [(n, rgb, bx0, bx1, by0, y0, bz0, bz1), (n + "_win_r", rgb, bx0, bx1, y1, by1, bz0, bz1),
                (n + "_win_above", rgb, bx0, bx1, y0, y1, z1, bz1)]
        if z0 > 0:
            out.append((n + "_win_below", rgb, bx0, bx1, y0, y1, bz0, z0))
    return out


def _window_models(win, name_label=(31, 32, 33)):
    """SDF of the window frame model, the free glass model (moved aside by set_pose, no DetachableJoint) and the optional handle model."""
    y, w, fw, sill, top = win["y"], win["width"], win["frame_width"], win["sill"], win["top"]
    zm, h, t = (sill + top) / 2, top - sill, 0.06
    f, spec_f = win["frame_colour"], "0.5 0.5 0.5 1"
    # frame bars in the frame link (origin at WALL_X, y, 0): fill the ring from the hole edge to the glass edge
    zlo, zhi = max(0.0, sill - fw), top + fw
    bars = [("left", (0, -(w / 2 + fw / 2), (zlo + zhi) / 2), (t, fw, zhi - zlo)), ("right", (0, w / 2 + fw / 2, (zlo + zhi) / 2), (t, fw, zhi - zlo)),
            ("top", (0, 0, top + fw / 2), (t, w, fw)), ("bottom", (0, 0, (zlo + sill) / 2), (t, w, sill - zlo))]
    bars = [b for b in bars if b[2][1] > 1e-6 and b[2][2] > 1e-6]
    if win["mullions"] in ("vertical", "cross"):
        bars.append(("mullion_v", (0, 0, zm), (t, 0.04, h)))
    if win["mullions"] == "cross":
        bars.append(("mullion_h", (0, 0, zm), (t, w, 0.04)))
    body = "\n".join(f'        <collision name="c_{n}"><pose>{_pose(*pos)} 0 0 0</pose>{_box(*size)}</collision>\n        '
                     + _vis(n, pos, _box(*size), f, spec_f) for n, pos, size in bars)
    text = f"""    <model name="window">
      <pose>{_pose(WALL_X, y, 0)} 0 0 0</pose>
      <link name="frame_anchor"><inertial><mass>1</mass></inertial></link>
      <joint name="anchor" type="fixed"><parent>world</parent><child>frame_anchor</child></joint>
      <link name="frame">
        <gravity>false</gravity>
        {_inertial(10, t, w + 2 * fw, h + 2 * fw)}
{body}
      </link>
      <joint name="frame_fix" type="fixed"><parent>frame_anchor</parent><child>frame</child></joint>
      {_label(31)}
    </model>
    <model name="window_glass">
      <pose>{_pose(WALL_X, y, zm)} 0 0 0</pose>
      <link name="link">
        <gravity>false</gravity>
        {_inertial(0.5, 0.01, w, h)}
        <visual name="glass">
          {_box(0.01, w, h)}
          <material><diffuse>0.75 0.88 0.92 1</diffuse><specular>0.9 0.9 0.9 1</specular></material>
          <transparency>0.85</transparency>
          <visibility_flags>2</visibility_flags>
        </visual>
      </link>
      {_label(33)}
    </model>
"""
    if win["handle"]:
        front = win["handle"] == "front"
        vis = "\n".join("        " + v for v in _handle_visuals("lever", 1))
        text += f"""    <model name="window_handle">
      <pose>{_pose(WALL_X + (-0.03 if front else 0.03), y + w / 2 - 0.06, min(zm, 1.4))} 0 0 {0 if front else 3.14159}</pose>
      <link name="link">
        <gravity>false</gravity>
        {_inertial(0.2, 0.05, 0.05, 0.05)}
{vis}
      </link>
      {_label(32)}
    </model>
"""
    return text


# --- geometry shared by the attributes and the world file ---
def _layout(spec, slot):
    """Derived numbers of one door: panel size, where its link sits (xc, zc), trim, joint, handle positions. Hinged panels sit
    centred in the wall (v4); with spec["door_stop"] they sit flush with the wall face on the swing (pull) side, hinge axis on that face."""
    w, h, y = spec["width"], spec["height"], SLOT_Y[slot]
    tw, proud = TRIM[spec["frame_style"]]
    g = {"y": y, "w": w, "h": h, "tw": tw, "proud": proud}
    if spec["door_type"] == "hinged":
        stop = bool(spec.get("door_stop"))
        sx = spec["swing_x"]
        g.update(pw=w - 0.02, ph=h - 0.02, xc=WALL_X + 0.03 * sx if stop else WALL_X, zc=h / 2)  # faces at WALL_X + 0.01 s .. + 0.05 s
        g["open"] = 1.2 * spec["swing_x"] * spec["hinge_y"]  # revolute about +z: the lock edge moves to -lock*sin(theta) in x
        g["range"] = [-1.6, 0.0] if g["open"] < 0 else [0.0, 1.6]
        g["hinge"] = [WALL_X + (0.05 if stop else 0.02) * sx, y + spec["hinge_y"] * g["pw"] / 2, g["zc"]]
        g["lock_y"] = y - spec["hinge_y"] * (g["pw"] / 2 - 0.08)  # handle: 0.08 in from the lock edge
        g["bar"] = spec["hinge_y"]  # lever bar points toward the hinge (world y sign)
    else:
        sgn = 1 if spec["hangs_on"] == "back" else -1
        g.update(pw=w + 0.08, ph=h + 0.04, sgn=sgn, travel=round(w + 0.15, 3))
        g["xc"], g["zc"] = WALL_X + sgn * (0.09 + proud), g["ph"] / 2  # panel 0.02 clear of the trim, 0.04 thick
        g["open"], g["range"] = round(0.9 * g["travel"], 3), [0.0, g["travel"]]
        g["lock_y"] = y - spec["slide_y"] * (g["pw"] / 2 - 0.1)  # handle: 0.1 in from the edge opposite the slide direction
        g["bar"] = 1
        e1, e2 = y - spec["slide_y"] * g["pw"] / 2, y + spec["slide_y"] * (g["pw"] / 2 + g["travel"])
        g["path"] = (min(e1, e2), max(e1, e2))  # y span of the panel plus where it slides to
    return g


def _keep_out(spec, g):
    """Floor rects to keep clutter off, per camera side: a strip in front of the opening, the swing arc, the panel + slide path."""
    y, w = g["y"], g["w"]
    def rect(side, depth, y0, y1):
        x0, x1 = (WALL_X - 0.05 - depth, WALL_X - 0.05) if side == "front" else (WALL_X + 0.05, WALL_X + 0.05 + depth)
        return {"x": [round(x0, 3), round(x1, 3)], "y": [round(y0, 3), round(y1, 3)]}
    strip = lambda side: rect(side, 0.3, y - w / 2 - g["tw"] - 0.1, y + w / 2 + g["tw"] + 0.1)  # opening + trim + handle margin
    out = {"front": [strip("front")], "back": [strip("back")]}
    if spec["door_type"] == "hinged":
        side = "back" if spec["swing_x"] > 0 else "front"
        out[side].append(rect(side, w + 0.05, y - w / 2 - 0.15, y + w / 2 + 0.15))
    else:
        out[spec["hangs_on"]].insert(0, rect(spec["hangs_on"], 0.25, g["path"][0] - 0.05, g["path"][1] + 0.05))
    return out


def attributes(world, name):
    """Ground truth of one world (same layout as door_room_v3.attributes.json) plus per-door spec, both camera sides, approach
    points and keep-out rects for the generator."""
    objs = {}
    for k, slot in enumerate(SLOTS, 1):
        spec, g = world["doors"][slot], _layout(world["doors"][slot], slot)
        held, sliding = world["doors_heldout"][slot], spec["door_type"] == "sliding"
        d = {"id": 10 * k + 1, "class": 1, "door_type": spec["door_type"], "glass": spec["glass"], "heldout": held,
             "width": spec["width"], "height": spec["height"], "frame_style": spec["frame_style"]}
        if "glazing" in spec:
            d["glazing"] = spec["glazing"]
        if sliding:
            d["hangs_on"] = spec["hangs_on"]
        d["seen_from"] = {s: seen_from(spec, s) for s in ("front", "back")}
        d.update(d["seen_from"]["front"])  # default = the front camera's view; the generator sets the frame's side
        if sliding:
            d["joint"] = {"name": "slide", "type": "prismatic", "unit": "m", "open": g["open"], "range": g["range"]}
            d["slide_world"] = {"point": [round(g["xc"], 4), g["y"], round(g["zc"], 4)], "direction": [0, spec["slide_y"], 0]}
        else:
            d["joint"] = {"name": "hinge", "type": "revolute", "unit": "rad", "open": g["open"], "range": g["range"]}
            d["hinge_world"] = {"point": [round(v, 4) for v in g["hinge"]], "direction": [0, 0, 1]}
        # approach, not *_world: frames.build_frame converts every *_world key into the camera frame
        d["approach"] = {"front": {"point": [WALL_X - 0.05, g["y"], 1.05], "direction": [-1, 0, 0]},
                         "back": {"point": [WALL_X + 0.05, g["y"], 1.05], "direction": [1, 0, 0]}}
        d["keep_out"] = _keep_out(spec, g)
        objs[f"door_{slot}"] = d
        for suffix, face, hid in (("", "front", 10 * k + 2), ("_back", "back", 10 * k + 4)):
            objs[f"handle_{slot}{suffix}"] = {"id": hid, "class": 2, "handle_type": face_handle(spec, face), "door": f"door_{slot}",
                                              "face": face, "heldout": held}
        if _has_hinges(spec):
            for i in range(3):
                objs[f"hinge_{slot}_{i}"] = {"id": 10 * k + 5 + i, "class": 3, "door": f"door_{slot}", "heldout": held}
        if has_pane(spec):
            objs[f"glass_{slot}"] = {"id": 10 * k + 3, "class": None, "part": "glass_pane", "part_of": f"door_{slot}",
                                     "detach_topic": f"/glass_{slot}/detach", "attach_topic": f"/glass_{slot}/attach"}
    for p in clutter.POOL:
        objs[p["name"]] = {"id": p["id"], "class": None, "role": p["role"], "kind": p["kind"], "place": p["place"], "size": p["size"]}
    win = world.get("window")
    if win:
        objs["window"] = {"id": 31, "class": 4, "width": win["width"], "sill": win["sill"], "top": win["top"], "low": win["low"],
                          "mullions": win["mullions"], "frame_width": win["frame_width"], "y": win["y"], "handle": win["handle"], "glass": True}
        ky = [round(win["y"] - win["width"] / 2 - win["frame_width"] - 0.1, 3), round(win["y"] + win["width"] / 2 + win["frame_width"] + 0.1, 3)]
        objs["window"]["keep_out"] = {"front": [{"x": [round(WALL_X - 0.35, 3), round(WALL_X - 0.05, 3)], "y": list(ky)}],
                                      "back": [{"x": [round(WALL_X + 0.05, 3), round(WALL_X + 0.35, 3)], "y": list(ky)}]}
        objs["window_glass"] = {"id": 33, "class": None, "part": "glass_pane", "part_of": "window"}  # free model: no DetachableJoint (never moves)
        if win["handle"]:
            objs["window_handle"] = {"id": 32, "class": None, "part": "window_handle", "part_of": "window", "face": win["handle"]}
    classes = {"1": "door", "2": "handle"}
    if any(o.get("class") == 3 for o in objs.values()):
        classes["3"] = "hinge"
    if win:
        classes["4"] = "window"
    return {"_note": f"Ground truth per object for {name} (door room v4: two random doors in one wall, see sim/doorsim/doors.py). "
                     "'id' = the Gazebo label on that model (unique per object, 1-255). Classes: 1 = door, 2 = handle, 3 = hinge (door room v5 only: three knuckles per hinged door, hinge_<slot>_<i>), 4 = window (door room v7: one fixed window in the door wall, parts window_glass / window_handle). World frame: "
                     "x toward the wall, y left, z up (metres). 'front' = camera at x < 3 looking +x, 'back' = camera at x > 3 looking -x. "
                     "Push / pull, hinge side and slide direction depend on the camera side: seen_from has both, the keys on the object "
                     "itself are the front view and the generator sets them per frame. Each door has a handle on each face "
                     "(handle_<slot> front, handle_<slot>_back back). 'heldout' = the door's property combo was never in training worlds. "
                     "approach = where viewpoints are placed per side, keep_out = floor areas clutter stays off. hinge_world / slide_world = "
                     "a point + direction from the joints in the world file; 'joint' = how to open the door (open = command). Objects with "
                     "class null are parts of another object (part_of), e.g. a glass pane, or the clutter pool (ids 100-254, no box).",
            "classes": classes,
            "camera": {"world_position": [0, 0, 1.2], "world_rpy": [0, 0, 0], "frame": "ROS optical: x right, y down, z forward"},
            "objects": objs}


# --- per-frame helpers ---
def keep_out(objects):
    """{"front": [rects], "back": [rects]} over all doors."""
    out = {"front": [], "back": []}
    for o in objects.values():
        for side, rects in o.get("keep_out", {}).items():
            out[side] += rects
    return out


def apply_side(objects, side):
    """In place: set the properties the camera on `side` sees (opening, hinge_side, slide_direction) on every object that has them."""
    for o in objects.values():
        if "seen_from" in o:
            o.update(o["seen_from"][side])


def side_of(position):
    return "front" if position[0] < WALL_X else "back"


# --- the world file ---
def _g(v):
    return f"{float(v) + 0.0:.8g}"


def _pose(*v):
    return " ".join(_g(a) for a in v)


def _mat(rgb, specular=None):
    c = " ".join(_g(a) for a in rgb)
    return f"<material><ambient>{c} 1</ambient><diffuse>{c} 1</diffuse>" + (f"<specular>{specular}</specular>" if specular else "") + "</material>"


def _box(sx, sy, sz):
    return f"<geometry><box><size>{_pose(sx, sy, sz)}</size></box></geometry>"


def _vis(name, pose, geom, rgb, specular=None):
    pose = tuple(pose) + (0,) * (6 - len(pose))
    return f'<visual name="{name}"><pose>{_pose(*pose)}</pose>{geom}{_mat(rgb, specular)}</visual>'


def _inertial(m, sx, sy, sz):
    return (f"<inertial><mass>{m}</mass><inertia><ixx>{_g(m / 12 * (sy**2 + sz**2))}</ixx><iyy>{_g(m / 12 * (sx**2 + sz**2))}</iyy>"
            f"<izz>{_g(m / 12 * (sx**2 + sy**2))}</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>")


def _label(i):
    return f'<plugin filename="gz-sim-label-system" name="gz::sim::systems::Label"><label>{i}</label></plugin>'


def _names(slot, names=None):
    """Model names of one door's parts: the v5 ones (door_a, handle_a, handle_a_back, glass_a, hinge_a_<i>) unless `names` is given
    (keys door, handle_front, handle_back, glass, hinge = a format string with {} for the knuckle index)."""
    return names or {"door": f"door_{slot}", "handle_front": f"handle_{slot}", "handle_back": f"handle_{slot}_back",
                     "glass": f"glass_{slot}", "hinge": f"hinge_{slot}_{{}}"}


def _detach(model, topics=None):
    t = f"<detach_topic>/{model}/detach</detach_topic><attach_topic>/{model}/attach</attach_topic>" if topics else ""
    return (f'<plugin filename="gz-sim-detachable-joint-system" name="gz::sim::systems::DetachableJoint">'
            f"<parent_link>panel</parent_link><child_model>{model}</child_model><child_link>link</child_link>{t}</plugin>")


def _panel_visuals(spec, g):
    """Visuals of the door panel in its link frame: one coloured box, solid boxes around the pane of a partly glazed door, or a
    frame (stiles + rails) around the glass pane."""
    pw, ph, t = g["pw"], g["ph"], 0.04
    if not spec["glass"] and spec.get("glazing", "none") != "none":
        dy, dz, w, h = glass_rect(spec, g)
        y0, y1, z0, z1 = dy - w / 2, dy + w / 2, dz - h / 2, dz + h / 2
        out = []
        for n, ya, yb, za, zb in (("left", -pw / 2, y0, -ph / 2, ph / 2), ("right", y1, pw / 2, -ph / 2, ph / 2),
                                  ("top", y0, y1, z1, ph / 2), ("bottom", y0, y1, -ph / 2, z0)):
            if yb - ya > 1e-6 and zb - za > 1e-6:
                out.append(_vis(f"part_{n}", (0, (ya + yb) / 2, (za + zb) / 2), _box(t, yb - ya, zb - za), spec["colour"]))
        return out
    if not spec["glass"]:
        return [_vis("v", (0, 0, 0), _box(t, pw, ph), spec["colour"])]
    f, spec_f = spec["frame_colour"], "0.5 0.5 0.5 1"
    return [_vis("stile_hinge", (0, -(pw / 2 - 0.05), 0), _box(t, 0.1, ph), f, spec_f), _vis("stile_lock", (0, pw / 2 - 0.05, 0), _box(t, 0.1, ph), f, spec_f),
            _vis("rail_top", (0, 0, ph / 2 - 0.05), _box(t, pw - 0.2, 0.1), f, spec_f), _vis("rail_bottom", (0, 0, -(ph / 2 - 0.07)), _box(t, pw - 0.2, 0.14), f, spec_f)]


def _has_hinges(spec):
    return spec["door_type"] == "hinged" and bool(spec.get("door_stop"))


def _hinge_model(slot, k, spec, g, i, names=None):
    """One hinge knuckle (door_stop worlds): a metal cylinder on the hinge axis at the hinge edge, own label 10 k + 5 + i, held on the panel."""
    z = (0.25, g["h"] / 2, g["h"] - 0.25)[i]
    x, y = g["xc"] + 0.02 * spec["swing_x"], g["y"] + spec["hinge_y"] * (g["pw"] / 2 + 0.005)
    return f"""    <model name="{_names(slot, names)["hinge"].format(i)}">
      <pose>{_pose(x, y, z)} 0 0 0</pose>
      <link name="link">
        <gravity>false</gravity>
        {_inertial(0.05, 0.02, 0.02, 0.09)}
        {_vis("knuckle", (0, 0, 0), "<geometry><cylinder><radius>0.008</radius><length>0.09</length></cylinder></geometry>", METAL_RGB)}
      </link>
      {_label(10 * k + 5 + i)}
    </model>
"""


def _door_model(slot, k, spec, g, names=None, initial=None):
    """Door model (panel + joint + controller + detachable joints). `names` = see _names; `initial` = joint position the controller
    starts at (the door is open by that much as built), None = no initial_position (v5)."""
    nm = _names(slot, names)
    hinged = spec["door_type"] == "hinged"
    pw, ph = g["pw"], g["ph"]
    vis = _panel_visuals(spec, g)
    if hinged:
        inertial = _inertial(25, 0.04, pw, ph)
        joint = (f'<joint name="hinge" type="revolute"><parent>frame_anchor</parent><child>panel</child>'
                 f'<pose>{_pose(0.02 * spec["swing_x"], spec["hinge_y"] * pw / 2, 0)} 0 0 0</pose><axis><xyz>0 0 1</xyz>'
                 f'<limit><lower>{g["range"][0]}</lower><upper>{g["range"][1]}</upper></limit><dynamics><damping>2.0</damping></dynamics></axis></joint>')
        ctrl, gains, jname = "hinge", "<p_gain>60</p_gain><i_gain>0.1</i_gain><d_gain>8</d_gain>", "hinge"
    else:
        inertial = _inertial(20, 0.04, pw, ph) + "<gravity>false</gravity>"
        for i, s in enumerate((-1, 1)):  # two hangers from the panel top up into the rail (rail z h+0.05 .. h+0.11)
            vis.append(_vis(f"hanger_{i}", (0, s * (pw / 2 - 0.14), g["h"] + 0.0675 - g["zc"]), _box(0.02, 0.06, 0.055), METAL_RGB))
        joint = (f'<joint name="slide" type="prismatic"><parent>frame_anchor</parent><child>panel</child><axis><xyz>0 {spec["slide_y"]} 0</xyz>'
                 f'<limit><lower>0</lower><upper>{g["travel"]}</upper></limit><dynamics><damping>20.0</damping></dynamics></axis></joint>')
        ctrl, gains, jname = "slide", "<p_gain>200</p_gain><i_gain>0.5</i_gain><d_gain>40</d_gain>", "slide"
    children = [_detach(nm["handle_front"]), _detach(nm["handle_back"])] + ([_detach(nm["glass"], topics=True)] if has_pane(spec) else [])
    children += [_detach(nm["hinge"].format(i)) for i in range(3) if _has_hinges(spec)]  # knuckles: own models so they get own labels
    body = "\n".join("        " + v for v in vis)
    start = "" if initial is None else f"<initial_position>{_g(initial)}</initial_position>"
    # the order of links / joints follows door_room_v3: anchor link fixed to the world, panel link on the joint
    return f"""    <model name="{nm["door"]}">
      <pose>{_pose(g["xc"], g["y"], 0)} 0 0 0</pose>
      <link name="frame_anchor"><inertial><mass>1</mass></inertial></link>
      <joint name="anchor" type="fixed"><parent>world</parent><child>frame_anchor</child></joint>
      <link name="panel">
        <pose>0 0 {_g(g["zc"])} 0 0 0</pose>
        {inertial}
        <collision name="c">{_box(0.04, pw, ph)}</collision>
{body}
      </link>
      {joint}
      <plugin filename="gz-sim-joint-position-controller-system" name="gz::sim::systems::JointPositionController">
        <joint_name>{ctrl}</joint_name><topic>/{nm["door"]}/cmd_pos</topic>{gains}{start}
      </plugin>
      {chr(10).join("      " + c for c in children).strip()}
      <plugin filename="gz-sim-joint-state-publisher-system" name="gz::sim::systems::JointStatePublisher"/>
      {_label(10 * k + 1)}
    </model>
"""


def _glass_model(slot, k, g, names=None, spec=None):
    dy, dz, w, h = glass_rect(spec, g) if spec is not None else (0, 0.02, g["pw"] - 0.2, g["ph"] - 0.24)
    return f"""    <model name="{_names(slot, names)["glass"]}">
      <pose>{_pose(g["xc"], g["y"] + dy, g["zc"] + dz)} 0 0 0</pose>
      <link name="link">
        <gravity>false</gravity>
        {_inertial(0.5, 0.01, w, h)}
        <visual name="glass">
          {_box(0.01, w, h)}
          <material><diffuse>0.75 0.88 0.92 1</diffuse><specular>0.9 0.9 0.9 1</specular></material>
          <transparency>0.85</transparency>
          <visibility_flags>2</visibility_flags>
        </visual>
      </link>
      {_label(10 * k + 3)}
    </model>
"""


def _handle_visuals(kind, bar):
    """Handle geometry in its own frame: origin on the door face, local -x pointing out of the door, bar = local y sign toward the hinge."""
    rose_pose = (-0.005, 0, 0, 0, 1.5708, 0)  # cylinders point along x
    metal, spec = (0.75, 0.75, 0.78), "0.6 0.6 0.6 1"
    if kind == "flush_pull":
        return [_vis("plate", (-0.001, 0, 0), _box(0.002, 0.04, 0.16), (0.2, 0.2, 0.22), "0.4 0.4 0.4 1"),
                _vis("grip", (-0.0025, 0, 0), _box(0.002, 0.025, 0.13), (0.05, 0.05, 0.05))]
    if kind == "pull_bar":  # vertical bar on two standoffs
        return [_vis("standoff_0", (-0.0125, 0, 0.12), _box(0.025, 0.02, 0.02), metal, spec),
                _vis("standoff_1", (-0.0125, 0, -0.12), _box(0.025, 0.02, 0.02), metal, spec),
                _vis("bar", (-0.035, 0, 0), _box(0.02, 0.02, 0.36), metal, spec)]
    if kind == "push_plate":  # thin flat plate
        return [_vis("plate", (-0.002, 0, 0), _box(0.004, 0.1, 0.3), metal, spec)]
    if kind == "knob":
        return [_vis("rose", rose_pose, "<geometry><cylinder><radius>0.03</radius><length>0.01</length></cylinder></geometry>", metal, spec),
                _vis("neck", (-0.025, 0, 0, 0, 1.5708, 0), "<geometry><cylinder><radius>0.01</radius><length>0.03</length></cylinder></geometry>", metal, spec),
                _vis("knob", (-0.05, 0, 0), "<geometry><sphere><radius>0.03</radius></sphere></geometry>", metal, spec)]
    return [_vis("rose", rose_pose, "<geometry><cylinder><radius>0.025</radius><length>0.01</length></cylinder></geometry>", metal, spec),
            _vis("neck", (-0.02, 0, 0), _box(0.02, 0.02, 0.02), metal, spec),
            _vis("lever", (-0.035, bar * 0.05, 0), _box(0.02, 0.13, 0.02), metal, spec)]


def _handle_model(slot, k, spec, g, face, names=None):
    front = face == "front"
    x = g["xc"] + (-0.02 if front else 0.02)  # the panel's face on that side
    bar = g["bar"] if front else -g["bar"]  # yaw pi turns local y around
    vis = "\n".join("        " + v for v in _handle_visuals(face_handle(spec, face), bar))
    name = _names(slot, names)["handle_front" if front else "handle_back"]
    return f"""    <model name="{name}">
      <pose>{_pose(x, g["lock_y"], 1.0)} 0 0 {0 if front else 3.14159}</pose>
      <link name="link">
        <gravity>false</gravity>
        {_inertial(0.2, 0.05, 0.05, 0.05)}
{vis}
      </link>
      {_label(10 * k + (2 if front else 4))}
    </model>
"""


def _boxes_model(name, comment, boxes, collide=True):
    """Static model of coloured boxes (name, rgb, x0, x1, y0, y1, z0, z1); visual + collision each."""
    out = [f"    <!-- {comment} -->", f'    <model name="{name}">', "      <static>true</static>", '      <link name="link">']
    for n, rgb, x0, x1, y0, y1, z0, z1 in boxes:
        pose, geom = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), _box(x1 - x0, y1 - y0, z1 - z0)
        out.append("        " + _vis(n, pose, geom, rgb))
        if collide:
            out.append(f'        <collision name="{n}"><pose>{_pose(*pose)} 0 0 0</pose>{geom}</collision>')
    return "\n".join(out + ["      </link>", "    </model>"]) + "\n"


def _room_shell():
    boxes = [("ceiling", CEILING_RGB, -2.0, 8.0, -3.1, 3.1, ROOM_HEIGHT, 2.8), ("wall_left", WALL_RGB, -2.0, 8.0, 3.0, 3.1, 0.0, ROOM_HEIGHT),
             ("wall_right", WALL_RGB, -2.0, 8.0, -3.1, -3.0, 0.0, ROOM_HEIGHT), ("wall_behind_camera", WALL_RGB, -2.1, -2.0, -3.1, 3.1, 0.0, 2.8),
             ("wall_far", WALL_RGB, 8.0, 8.1, -3.1, 3.1, 0.0, 2.8)]
    return _boxes_model("room_shell", "closed room, 5 m deep on both sides of the wall: ceiling, side walls, end walls (background, no label)", boxes)


def _door_wall_visuals(spec, g, slot):
    """Visual-only boxes of one door in the wall frame (origin on the wall centre plane, y as in _layout): trim on both faces, the door
    stop of a hinged door with door_stop, the top rail of a sliding door. `slot` = name tag of the boxes."""
    visuals = []
    for side, sgn in (("front", -1), ("back", 1)):
        if g["tw"]:
            x, tw, p, y, w, h = sgn * (0.05 + g["proud"] / 2), g["tw"], g["proud"], g["y"], g["w"], g["h"]
            c = spec["trim_colour"]
            visuals += [_vis(f"trim_{slot}_{side}_l", (x, y - (w / 2 + tw / 2), h / 2), _box(p, tw, h), c),
                        _vis(f"trim_{slot}_{side}_r", (x, y + (w / 2 + tw / 2), h / 2), _box(p, tw, h), c),
                        _vis(f"trim_{slot}_{side}_t", (x, y, h + tw / 2), _box(p, w + 2 * tw, tw), c)]
    if spec["door_type"] == "hinged" and spec.get("door_stop"):  # stop on the push side, against the panel's push face
        s, y, w, h = spec["swing_x"], g["y"], g["w"], g["h"]
        c = spec["trim_colour"] if g["tw"] else WALL_RGB
        xs, dx = 0.0025 * s, 0.015  # x from -0.005 s to +0.01 s
        visuals += [_vis(f"stop_{slot}_l", (xs, y - w / 2 + 0.015, h / 2), _box(dx, 0.03, h), c),
                    _vis(f"stop_{slot}_r", (xs, y + w / 2 - 0.015, h / 2), _box(dx, 0.03, h), c),
                    _vis(f"stop_{slot}_t", (xs, y, h - 0.015), _box(dx, w, 0.03), c)]
    if spec["door_type"] == "sliding":  # rail over the opening and the slide path, on the face the panel hangs on
        lo, hi = g["path"]
        visuals.append(_vis(f"slide_rail_{slot}", (g["sgn"] * (0.09 + g["proud"]), (lo + hi) / 2, g["h"] + 0.08), _box(0.04, hi - lo, 0.06), METAL_RGB))
    return visuals


def _back_wall(world):
    """The wall at WALL_X: boxes around the two openings (each as wide as its door), a lintel over each, trim on both faces, and
    the top rail of sliding doors, the door stop of hinged doors with door_stop. Trim / rail / stop boxes are visual only."""
    parts, edge = [], -3.0
    lay = sorted(((SLOT_Y[s], s) for s in SLOTS))
    full = lambda n, y0, y1: (n, WALL_RGB, -0.05, 0.05, y0, y1, 0.0, ROOM_HEIGHT)
    for i, (y, slot) in enumerate(lay):
        h, w = world["doors"][slot]["height"], world["doors"][slot]["width"]
        parts += [full(f"wall_{i}", edge, y - w / 2), ("lintel_" + slot, WALL_RGB, -0.05, 0.05, y - w / 2, y + w / 2, h, ROOM_HEIGHT)]
        edge = y + w / 2
    parts.append(full(f"wall_{len(lay)}", edge, 3.0))
    if world.get("window"):
        parts = _window_wall(parts, world["window"])
    visuals = []
    for slot in SLOTS:
        spec = world["doors"][slot]
        visuals += _door_wall_visuals(spec, _layout(spec, slot), slot)
    text = _boxes_model("back_wall", f"wall at x = {WALL_X:g}: 0.1 thick, openings as wide / high as their doors, lintels, trim", parts)
    text = text.replace("      <static>true</static>", f"      <static>true</static>\n      <pose>{WALL_X:g} 0 0 0 0 0</pose>")
    return text.replace("      </link>", "".join("        " + v + "\n" for v in visuals) + "      </link>")


def _swap(text, old, new):
    if old not in text:
        raise ValueError(f"door_room_v3.sdf has no {old!r}")
    return text.replace(old, new, 1)


def world_sdf(world, name):
    """Full SDF text of one world, built from door_room_v3.sdf (plugins, scene, lights, floor, camera rig, clutter pool) with the
    v3 wall, doors and room shell replaced: a second room lamp behind the wall, a room 5 m deep on both sides, this world's doors."""
    v3 = (ROOM / "door_room_v3.sdf").read_text()
    head = v3[v3.index("<sdf "):v3.index("    <!-- Back wall at x = 3")]
    head = _swap(head, '<world name="door_room_v3">', f'<world name="{name}">')
    i = head.index('<light type="point" name="ceiling_light">')
    j = head.index("</light>", i) + len("</light>")
    back_lamp = head[i:j].replace('name="ceiling_light"', 'name="ceiling_light_back"').replace("<pose>1.5 0 2.5", "<pose>4.5 0 2.5")
    head = head[:j] + "\n    " + back_lamp + head[j:]
    head = _swap(head, '<model name="floor">\n      <static>true</static>', '<model name="floor">\n      <static>true</static>\n      <pose>3 0 0 0 0 0</pose>')
    rig_and_pool = v3[v3.index("    <!-- Fixed sensor rig"):v3.index("    <!-- closed room")]
    parts = [f"<?xml version=\"1.0\" ?>\n<!-- Door room v4 world {name}: two random doors (sim/doorsim/doors.py), room on both sides of the wall. Generated by sim/door_room/make_v4.py. -->\n", head,
             _back_wall(world)]
    for k, slot in enumerate(SLOTS, 1):
        spec = world["doors"][slot]
        g = _layout(spec, slot)
        parts += ["\n", _door_model(slot, k, spec, g)]
        if has_pane(spec):
            parts.append(_glass_model(slot, k, g, spec=spec))
        parts += [_handle_model(slot, k, spec, g, "front"), _handle_model(slot, k, spec, g, "back")]
        if _has_hinges(spec):
            parts += [_hinge_model(slot, k, spec, g, i) for i in range(3)]
    if world.get("window"):
        parts.append(_window_models(world["window"]))
    return "".join(parts) + "\n" + rig_and_pool + _room_shell() + "  </world>\n</sdf>\n"
