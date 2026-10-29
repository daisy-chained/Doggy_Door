"""Random scenes (door open amounts, lights) and viewpoints (camera poses in an approach zone per door).
Everything takes a numpy Generator, so a seed gives the same dataset every time."""
import math


def door_targets(rng, attrs, cfg):
    """{door name: joint position}. Each door is closed with probability p_closed, otherwise opened
    a random fraction (min_open_fraction .. 1) of its range, toward the open end."""
    out = {}
    for name, o in attrs["objects"].items():
        if "joint" not in o:
            continue
        lo, hi = o["joint"]["range"]
        end = lo if abs(lo) > abs(hi) else hi
        if rng.random() < cfg["p_closed"]:
            out[name] = 0.0
        else:
            out[name] = float(end * rng.uniform(cfg["min_open_fraction"], 1.0))
    return out


def lights(rng, cfg):
    """{light name: {"diffuse": [r, g, b]}}: base colour times a random intensity, plus a small random tint."""
    out = {}
    for name, c in cfg.items():
        k = rng.uniform(*c["intensity"])
        tint = rng.uniform(-c["tint"], c["tint"], 3)
        out[name] = {"diffuse": [float(min(1.0, max(0.0, b * k * (1 + t)))) for b, t in zip(c["base"], tint)]}
    return out


def look_at(position, target):
    """(yaw, pitch) that point the camera body x axis from position at target (roll 0).
    Gazebo: positive pitch looks down."""
    dx, dy, dz = (t - p for t, p in zip(target, position))
    return math.atan2(dy, dx), math.atan2(-dz, math.hypot(dx, dy))


def view(rng, attrs, cfg):
    """A camera pose in front of a random door: distance, angle off the door's facing direction and
    camera height from cfg ranges; aimed at a point on the door (target_height, target_jitter along the wall)."""
    doors = [n for n, o in attrs["objects"].items() if "front_world" in o]
    door = doors[rng.integers(len(doors))]
    f = attrs["objects"][door]["front_world"]
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
    return {"world_position": position, "world_rpy": [roll, pitch, yaw], "target_door": door, "target_world": target}
