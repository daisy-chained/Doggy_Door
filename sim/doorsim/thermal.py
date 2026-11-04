"""Thermal camera (Madelief 2026-11-04): temperatures per model for a room-temperature and a fire-heated scenario, the
thermal sensor + temperatures injected into a v4 world, a replay plan to render thermal over an existing dataset, and
16-bit PNG encoding in 0.01 K (the ThermalSensor resolution)."""
import copy
import xml.etree.ElementTree as ET
import numpy as np

try:
    from doorsim import clutter
except ImportError:  # run with sim/doorsim itself on the path
    import clutter


def to_png16(kelvin):
    return np.clip(np.round(np.asarray(kelvin) * 100), 0, 65535).astype(np.uint16)


def from_png16(png):
    return np.asarray(png, float) / 100


C = 273.15
MIN_K, MAX_K, RES = 253.15, 655.35, 0.01  # sensor range: 16 bit x 0.01 K tops out at 655.35 K, so no PNG value clips
# (low, high) ranges. room: air in degC; offsets are K from the air (a model's apparent temperature), signed or magnitude
ROOM_AIR = (18, 24)          # degC, a normal room
ROOM_STRUCT = (-1, 1)        # K, walls / floor / shell: close to the air
ROOM_DOOR = (0.5, 2)         # K magnitude, door leaves: slightly off the air (random sign)
ROOM_GLASS = (1, 3)          # K magnitude, glass: shows the other room / outside
ROOM_HANDLE = (1, 3)         # K magnitude, metal: low emissivity, reflects its surroundings
ROOM_CLUTTER = (0.5, 2)      # K magnitude, furniture and boxes
# fire: absolute degC
FIRE_AIR, FIRE_LAYER = (30, 45), (150, 400)  # room air; hot smoke layer under the ceiling
FIRE_SHELL, FIRE_WALL = (40, 80), (30, 60)   # ceiling + walls (room_shell); back wall and floor
FIRE_DOOR, FIRE_HANDLE, FIRE_GLASS, FIRE_CLUTTER = (30, 70), (30, 80), (35, 90), (25, 50)


def kind(name):
    """shell | wall | door | glass | handle | clutter | rig: how a model is heated."""
    if name == "rig":
        return "rig"
    if name == "room_shell":
        return "shell"
    if name in ("floor", "back_wall"):
        return "wall"
    for k in ("door", "glass", "handle"):
        if name.startswith(k + "_"):
            return k
    return "clutter"


def temperatures(rng, scenario, models):
    """{"air": K, "layer": K or None, "models": {name: K}} for scenario "room" (near room temperature) or "fire"
    (hot room, hot smoke layer); the camera rig gets no temperature."""
    if scenario not in ("room", "fire"):
        raise ValueError(f"unknown thermal scenario {scenario!r}")
    u = lambda r: float(rng.uniform(*r))
    sign = lambda: float(rng.choice([-1.0, 1.0]))
    out = {}
    if scenario == "room":
        air = u(ROOM_AIR) + C
        off = {"shell": lambda: u(ROOM_STRUCT), "wall": lambda: u(ROOM_STRUCT), "door": lambda: sign() * u(ROOM_DOOR),
               "glass": lambda: sign() * u(ROOM_GLASS), "handle": lambda: sign() * u(ROOM_HANDLE), "clutter": lambda: sign() * u(ROOM_CLUTTER)}
        for n in models:
            if kind(n) != "rig":
                out[n] = air + off[kind(n)]()
        return {"air": air, "layer": None, "models": out}
    air, layer = u(FIRE_AIR) + C, u(FIRE_LAYER) + C
    rng_of = {"shell": FIRE_SHELL, "wall": FIRE_WALL, "door": FIRE_DOOR, "glass": FIRE_GLASS, "handle": FIRE_HANDLE, "clutter": FIRE_CLUTTER}
    for n in models:
        if kind(n) != "rig":
            out[n] = u(rng_of[kind(n)]) + C
    return {"air": air, "layer": layer, "models": out}


def _thermal_plugin(k):
    p = ET.Element("plugin", filename="gz-sim-thermal-system", name="gz::sim::systems::Thermal")
    ET.SubElement(p, "temperature").text = f"{k:.2f}"
    return p


def inject(sdf_text, temps, t_range=(MIN_K, MAX_K)):
    """World SDF text with a Thermal plugin in every visual of each model in temps (per visual: gz-sim8 ignores it on the
    model) and a thermal camera next to the rig's rgbd camera (same FOV, clip, rate; 320 x 240 L16 on topic "thermal")."""
    root = ET.fromstring(sdf_text.encode())
    world = root.find("world")
    for m in world.findall("model"):
        if m.get("name") in temps:
            for v in m.iter("visual"):
                v.append(_thermal_plugin(temps[m.get("name")]))
    rig = next(m for m in world.findall("model") if m.get("name") == "rig")
    for link in rig.iter("link"):
        rgbd = next((s for s in link.findall("sensor") if s.get("type") == "rgbd_camera"), None)
        if rgbd is None:
            continue
        cam = rgbd.find("camera")
        s = ET.Element("sensor", name="thermal_camera", type="thermal")
        if rgbd.find("pose") is not None:
            s.append(copy.deepcopy(rgbd.find("pose")))
        ET.SubElement(s, "topic").text = "thermal"
        for tag in ("update_rate", "always_on"):
            ET.SubElement(s, tag).text = rgbd.find(tag).text
        c = ET.SubElement(s, "camera")
        ET.SubElement(c, "horizontal_fov").text = cam.find("horizontal_fov").text
        img = ET.SubElement(c, "image")
        for tag, v in (("width", "320"), ("height", "240"), ("format", "L16")):
            ET.SubElement(img, tag).text = v
        c.append(copy.deepcopy(cam.find("clip")))
        pl = ET.SubElement(s, "plugin", filename="gz-sim-thermal-sensor-system", name="gz::sim::systems::ThermalSensor")
        for tag, v in (("min_temp", t_range[0]), ("max_temp", t_range[1]), ("resolution", RES)):
            ET.SubElement(pl, tag).text = str(v)
        link.insert(list(link).index(rgbd) + 1, s)
        break
    else:
        raise ValueError("rig has no rgbd_camera sensor")
    ET.indent(root)
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def replay_plan(metas):
    """[{"world", "scene", "doors": {door: joint pos}, "views": [(frame, camera)], "clutter": {every pool item: placement}}]
    per (world, scene) in frame order, to render another sensor over an existing dataset."""
    plan = {}
    for frame in sorted(metas):
        m = metas[frame]
        key = (m["world"], m["scene"])
        if key not in plan:
            parked = {p["name"]: {"position": list(clutter.PARK), "yaw": 0.0, "colour": None, "in_scene": False} for p in clutter.POOL}
            doors = {n: o["joint_pos"] for n, o in m["objects"].items() if "joint" in o}
            plan[key] = {"world": m["world"], "scene": m["scene"], "doors": doors, "views": [], "clutter": {**parked, **m.get("clutter", {})}}
        plan[key]["views"].append((frame, m["camera"]))
    return list(plan.values())
