"""Random scenes (door open amounts, lights) and viewpoints (camera poses in an approach zone per door).
Everything takes a numpy Generator, so a seed gives the same dataset every time."""
import math
import numpy as np


def door_targets(rng, attrs, cfg, scene=None):
    """{door name: joint position}. Each door is closed with probability p_closed, otherwise opened
    a random fraction (min_open_fraction .. 1) of its range, toward the open end. With cfg balance_states, door i (sorted
    names) is closed iff (scene + i) is even, so each door is closed in half the scenes."""
    out = {}
    bal = cfg.get("balance_states")
    items = sorted(attrs["objects"].items()) if bal else attrs["objects"].items()
    i = 0
    for name, o in items:
        if "joint" not in o:
            continue
        k, i = i, i + 1
        lo, hi = o["joint"]["range"]
        end = lo if abs(lo) > abs(hi) else hi
        if (scene + k) % 2 == 0 if bal else rng.random() < cfg["p_closed"]:
            out[name] = 0.0
        else:
            out[name] = float(end * rng.uniform(cfg["min_open_fraction"], 1.0))
    return out


def lights(rng, cfg, level=1.0):
    """{light name: {"diffuse": [r, g, b]}}: base colour times a random intensity, plus a small random tint,
    then times `level` (lighting mode; same random draws whatever the level)."""
    out = {}
    for name, c in cfg.items():
        k = rng.uniform(*c["intensity"])
        tint = rng.uniform(-c["tint"], c["tint"], 3)
        out[name] = {"diffuse": [float(min(1.0, max(0.0, b * k * (1 + t))) * level) for b, t in zip(c["base"], tint)]}
    return out


def scene_rngs(seed, scene):
    """(scene stream, light stream) for one scene: scenes can be rendered on their own, and the lighting
    never changes what the scene stream draws (doors, clutter, views, glass)."""
    return np.random.default_rng([seed, scene]), np.random.default_rng([seed, scene, 1])


def lighting(rng, cfg):
    """{"mode", "level", "headlight"} for one scene: mode by share p, room light level log-uniform in the mode's range,
    headlight on with a chance per mode (brightness uniform), else 0. cfg["fixed"] = {level, headlight} skips the draws."""
    if cfg.get("fixed"):
        f = cfg["fixed"]
        return {"mode": "fixed", "level": float(f["level"]), "headlight": float(f.get("headlight", 0.0))}
    modes = list(cfg["modes"])
    p = np.array([cfg["modes"][m]["p"] for m in modes], float)
    mode = modes[rng.choice(len(modes), p=p / p.sum())]
    lo, hi = cfg["modes"][mode]["level"]
    level = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    on = rng.random() < cfg["headlight"]["p"][mode]
    head = float(rng.uniform(*cfg["headlight"]["level"])) if on else 0.0
    return {"mode": mode, "level": level, "headlight": head}


def flicker(rng, cfg, level):
    """Room light level for one view: with chance p it dips to level x (1 - depth). Fixed lighting never flickers."""
    if cfg.get("fixed"):
        return level
    f = cfg["flicker"]
    return float(level * (1 - rng.uniform(*f["depth"]))) if rng.random() < f["p"] else level


def look_at(position, target):
    """(yaw, pitch) that point the camera body x axis from position at target (roll 0).
    Gazebo: positive pitch looks down."""
    dx, dy, dz = (t - p for t, p in zip(target, position))
    return math.atan2(dy, dx), math.atan2(-dz, math.hypot(dx, dy))


def view(rng, attrs, cfg, index=None):
    """A camera pose in front of a random door: distance, angle off the door's facing direction and
    camera height from cfg ranges; aimed at a point on the door (target_height, target_jitter along the wall).
    With cfg balance and an index, view i takes door i % n and side (i // n) % 2 (front, back) instead of drawing them.
    Doors with `approach` (v4) are seen from the front or, with chance cfg p_back, from the back; old doors only from the front."""
    doors = [n for n, o in attrs["objects"].items() if "front_world" in o or "approach" in o]
    fixed = bool(cfg.get("balance")) and index is not None
    door = doors[index % len(doors)] if fixed else doors[rng.integers(len(doors))]
    o = attrs["objects"][door]
    side = "front"
    if "approach" in o:
        if fixed:
            side = ("front", "back")[(index // len(doors)) % 2]
        else:
            side = "back" if rng.random() < cfg.get("p_back", 0) else "front"
    f = o["approach"][side] if "approach" in o else o["front_world"]
    px, py, pz = f["point"]
    nx, ny = f["direction"][0], f["direction"][1]
    a = math.radians(rng.uniform(-cfg["angle_deg"], cfg["angle_deg"]))
    d = rng.uniform(*cfg["distance"])
    vx, vy = nx * math.cos(a) - ny * math.sin(a), nx * math.sin(a) + ny * math.cos(a)
    position = [px + d * vx, py + d * vy, rng.uniform(*cfg["height"])]
    along = rng.uniform(-cfg["target_jitter"], cfg["target_jitter"])  # along the wall: tangent = normal turned 90 deg
    target = [px - ny * along, py + nx * along, rng.uniform(*cfg["target_height"])]
    yaw, pitch = look_at(position, target)
    roll = math.radians(rng.uniform(-cfg["roll_deg"], cfg["roll_deg"]))
    return {"world_position": position, "world_rpy": [roll, pitch, yaw], "target_door": door, "target_world": target, "side": side}
