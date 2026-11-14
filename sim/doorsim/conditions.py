"""Patrol conditions (Madelief 2026-11-07 / 2026-11-13): darkness in Gazebo (room lights x level + the robot's headlight), smoke and
camera noise on the saved frames (doorsim.effects, the same code as the training sets). No Gazebo / ROS here.

make(): building SDF text -> world SDF text with scaled lights + cond (one light level / smoke visibility per run, or "mixed" =
a draw per space like the training mix). apply(): one saved stop frame -> rgb_clean / depth_clean kept, rgb / depth with the effects.
Spec and design: sim/tests/test_conditions.py (header); CLI: sim/building/conditions.py."""
import json, math, pathlib, re, shutil, sys
import numpy as np
from PIL import Image

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import effects, glass

LIGHT_MODES = {"normal": {"p": 0.34, "level": [0.6, 1.2]}, "dim": {"p": 0.33, "level": [0.08, 0.4]},
               "dark": {"p": 0.33, "level": [0.003, 0.05]}}                     # = door_room_v9_6k.yaml lighting.modes
HEADLIGHT = {"range": 10, "attenuation": [0.2, 0.05, 0.02], "spot": [0.4, 0.8, 1.0]}   # = the training rig's headlight


# --- lights in the world file ---
def scale_lights(sdf_text, levels, sun_level):
    """SDF text with the diffuse rgb of light "ceiling_light_<space>" x levels[space] (float = all spaces) and of "sun" x sun_level;
    nothing else changes (alpha and every other line stay as they are)."""
    def factor(name):
        if name == "sun":
            return sun_level
        if name.startswith("ceiling_light_"):
            return levels if not isinstance(levels, dict) else levels[name[len("ceiling_light_"):]]
        return None

    def block(m):
        k = factor(m.group(1))
        if k is None:
            return m.group(0)
        def diffuse(d):
            r, g, b = (float(v) * k for v in d.group(1).split()[:3])
            return f"<diffuse>{r:.6g} {g:.6g} {b:.6g}{d.group(2)}</diffuse>"
        return re.sub(r"<diffuse>\s*(\S+\s+\S+\s+\S+)(.*?)</diffuse>", diffuse, m.group(0), count=1, flags=re.S)
    return re.sub(r'<light\b[^>]*\bname="([^"]+)"[^>]*>.*?</light>', block, sdf_text, flags=re.S)


# --- per-space draws ---
def sample_rooms(rng, spaces, light, smoke, smoke_cfg):
    """{space: {"mode", "light", "visibility_m"}}. light: float (all spaces, mode "fixed") or "mixed" (mode by LIGHT_MODES p,
    level log-uniform in the mode's range). smoke: None (clear), float (all spaces) or "mixed" (smoky with smoke_cfg["p"],
    V log-uniform smoke_cfg["visibility_m"], else None)."""
    modes = list(LIGHT_MODES)
    out = {}
    for s in spaces:
        if light == "mixed":
            m = modes[rng.choice(len(modes), p=np.array([LIGHT_MODES[k]["p"] for k in modes]) / sum(LIGHT_MODES[k]["p"] for k in modes))]
            lo, hi = LIGHT_MODES[m]["level"]
            lv = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        else:
            m, lv = "fixed", float(light)
        if smoke == "mixed":
            lo, hi = smoke_cfg["visibility_m"]
            smoky = rng.random() < smoke_cfg["p"]
            vis = float(math.exp(rng.uniform(math.log(lo), math.log(hi)))) if smoky else None
        else:
            vis = None if smoke is None else float(smoke)
        out[s] = {"mode": m, "light": lv, "visibility_m": vis}
    return out


def make(bjson, sdf_text, light, smoke, headlight, seed, effects_cfg):
    """(world SDF text, cond). bjson = building.json dict. cond = {light, smoke, headlight, seed, sun_level, spaces {space: {rect,
    mode, light, visibility_m}}, smoke_shape {layer, patch, patch_scale, seed}}; JSON-serialisable. The smoke shape is drawn
    first, so it does not depend on the light / smoke arguments."""
    rng = np.random.default_rng(seed)
    shape = effects.sample_smoke(rng, effects_cfg["smoke"], visibility=1.0)
    shape = {k: shape[k] for k in ("layer", "patch", "patch_scale", "seed")}
    lay = bjson["layout"]
    rects = {"corridor": lay["corridor"], **{r["id"]: r["rect"] for r in lay["rooms"]}}
    rooms = sample_rooms(rng, list(rects), light, smoke, effects_cfg["smoke"])
    sun = min(v["light"] for v in rooms.values())
    world = scale_lights(sdf_text, {s: v["light"] for s, v in rooms.items()}, sun)
    cond = {"light": light if light == "mixed" else float(light), "smoke": smoke if smoke in (None, "mixed") else float(smoke),
            "headlight": float(headlight), "seed": int(seed), "sun_level": sun,
            "spaces": {s: {"rect": list(rects[s]), **v} for s, v in rooms.items()}, "smoke_shape": shape}
    return world, cond


def smoke_params(cond):
    """smoke dict for effects.smoke_render (per-space visibilities + the run's smoke shape), None when no space is smoky."""
    sp = [{"rect": v["rect"], "visibility_m": v["visibility_m"]} for v in cond["spaces"].values()]
    if all(s["visibility_m"] is None for s in sp):
        return None
    return {"visibility_m": None, "spaces": sp, **cond["smoke_shape"]}


# --- frames ---
def camera_pose(x, y, yaw, height):
    """Camera dict for effects (world position + rpy) from the camera link's map pose and its height."""
    return {"world_position": [x, y, height], "world_rpy": [0.0, 0.0, yaw]}


def lamps(world_sdf_text, cond, cameras):
    """Lamps for effects.smoke_render: every light of the world with its diffuse (after scale_lights), plus when
    cond["headlight"] > 0 a spot per camera pose in `cameras`. Lamps with diffuse 0 are left out. Ceiling lights get
    "rect" (their space): they light the smoke in that space only."""
    out = list(effects.world_lights_text(world_sdf_text))
    for l in out:   # a ceiling light shines in its own space only (light_at has no shadows, so it would light the smoke through walls)
        sid = l["name"][len("ceiling_light_"):] if l["name"].startswith("ceiling_light_") else None
        if sid in cond["spaces"]:
            l["rect"] = cond["spaces"][sid]["rect"]
    h = cond["headlight"]
    for k, c in enumerate(cameras if h > 0 else []):
        yaw = c["world_rpy"][2]
        out.append({"name": f"headlight_{k}", "type": "spot", "diffuse": [h, h, 0.95 * h], "position": list(c["world_position"]),
                    "direction": [math.cos(yaw), math.sin(yaw), 0.0], "attenuation": [HEADLIGHT["range"], *HEADLIGHT["attenuation"]],
                    "spot": HEADLIGHT["spot"]})
    return [l for l in out if any(l["diffuse"])]


def apply(frame_dir, cond, effects_cfg, lamps, camera, K, rng):
    """Effects on one saved frame folder (rgb.png, depth.png uint16 mm, meta.json): clean copies kept, then smoke (if any space is
    smoky) + camera noise (always) -> rgb.png, depth.png (only with smoke), meta["effects"] (returned)."""
    d = pathlib.Path(frame_dir)
    shutil.copyfile(d / "rgb.png", d / "rgb_clean.png")
    shutil.copyfile(d / "depth.png", d / "depth_clean.png")
    rgb = np.asarray(Image.open(d / "rgb.png").convert("RGB"))
    sc, smoke = effects_cfg["smoke"], smoke_params(cond)
    if smoke:
        depth = effects.depth_m(d / "depth_clean.png")
        t, air = effects.smoke_render(depth, K, camera, smoke, lamps, sc["colour"], sc.get("samples", 24), sc.get("far", 12.0))
        rgb = effects.add_smoke(rgb, t, air * sc.get("scatter_gain", 1.0))
        depth = effects.smoke_depth(depth, t, rng, effects_cfg["depth"])
        Image.fromarray(glass.to_png16(depth)).save(d / "depth.png")
    rgb, gain = effects.sensor_noise(rgb, rng, effects_cfg["noise"])
    Image.fromarray(rgb).save(d / "rgb.png")
    x, y = camera["world_position"][:2]
    light = next((v["light"] for v in cond["spaces"].values()
                  if v["rect"][0] <= x <= v["rect"][2] and v["rect"][1] <= y <= v["rect"][3]), None)
    eff = {"smoke": smoke, "noise_gain": gain, "light": light, "headlight": cond["headlight"]}
    meta = json.loads((d / "meta.json").read_text())
    meta["effects"] = eff
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return eff
