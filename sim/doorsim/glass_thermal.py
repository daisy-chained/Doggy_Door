"""Glass in the thermal image (LWIR, 8-14 um): what the camera sees on a pane, as a step after the label render (no Gazebo).

Research-based, Claude decision 2026-11-08 (logged in the project note): real glass is opaque in LWIR. A pane shows a mix of
- its own emission:  eps(theta) T_glass^4          uncoated glass e0 ~ 0.84-0.92 at normal incidence (low-E coated: 0.1-0.3,
                                                   only as a stress test)
- a reflection:      R(theta) T_refl^4             R0 = 1 - e0 - leak, rising to 1 at grazing incidence (Schlick's
                                                   approximation of Fresnel: R = R0 + (1 - R0)(1 - cos)^5)
- a leak:            leak T_behind^4               transmission, ~0 for real glass, 0-10 % as a stress test
with eps = 1 - R - leak, all mixed as radiance (T^4, grey body) and blended per thermal pixel by the pane's area fraction.
T_refl is the surroundings (air + a per-pane offset + smooth noise) and, where the mirror image of the camera falls in the
pane, the robot's own warm mirror image: a disc at the projection of the mirror point of the camera (2 (c . n) n in the camera
frame, c = pane centre) with radius f radius_m / z.
Camera frame = optical (x right, y down, z forward), as meta box_3d. Pane "glass_x" belongs to door "door_x".
cfg (see sim/configs/glass_thermal*.yaml): {"emissivity": [lo, hi], "leak": [lo, hi], "env_K": [lo, hi], "env_noise_K": K,
"env_cell": px, "robot": {"p": probability, "radius_m": m}}."""
import numpy as np

from doorsim import glass


def schlick(cos_t, r0):
    """Reflectance at incidence cos_t = cos(theta) (1 = head on, 0 = grazing) for normal-incidence reflectance r0."""
    return r0 + (1 - r0) * (1 - cos_t) ** 5


def _rot(q):
    """Rotation matrix of a quaternion (x, y, z, w)."""
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pane_geometry(meta, pane):
    """{"cos": |n . c| / |c|, "mirror": 2 (c . n) n (m, camera frame)} of pane "glass_x" from its door's box_3d, or None when
    the door has no box_3d. n = the box axis with the smallest size, c = the box centre."""
    box = (meta["objects"].get("door_" + pane.split("_", 1)[1]) or {}).get("box_3d")
    if not box:
        return None
    n = _rot(box["orientation_xyzw"])[:, int(np.argmin(box["size"]))]
    c = np.asarray(box["center"], float)
    d = float(c @ n)
    return {"cos": abs(d) / float(np.linalg.norm(c)), "mirror": 2 * d * n}


def draw(rng, cfg):
    """Per-pane draw {"emissivity", "leak", "env_K" (floats), "robot" (bool)} uniform in the cfg ranges."""
    e = float(rng.uniform(*cfg["emissivity"]))
    l = float(rng.uniform(*cfg["leak"]))
    k = float(rng.uniform(*cfg["env_K"]))
    return {"emissivity": e, "leak": l, "env_K": k, "robot": bool(rng.random() < (cfg.get("robot") or {}).get("p", 0.0))}


def fractions(seg_full, meta, shape):
    """{pane: (H, W) fraction of each thermal pixel covered by the pane's seg id}; seg_full at an integer multiple of shape.
    Panes that do not appear are left out."""
    h, w = shape
    bh, bw = seg_full.shape[0] // h, seg_full.shape[1] // w
    out = {}
    for name, o in meta["objects"].items():
        if name.startswith("glass_") and o.get("id") is not None:
            f = (seg_full[:h * bh, :w * bw] == o["id"]).reshape(h, bh, w, bw).mean(axis=(1, 3))
            if f.any():
                out[name] = f
    return out


def apply(scene_K, behind_K, frac, geo, params, air_K, robot_K, K_thermal, rng, cfg):
    """(H, W) K: scene_K with each pane in frac replaced (by its area fraction) by the mix of emission, reflection and leak.
    behind_K None -> the leak sees the air. K_thermal = 9 numbers (row-major 3 x 3 pinhole)."""
    out4 = np.asarray(scene_K, np.float64) ** 4
    h, w = out4.shape
    fx, cx, fy, cy = K_thermal[0], K_thermal[2], K_thermal[4], K_thermal[5]
    behind4 = (air_K if behind_K is None else np.asarray(behind_K, np.float64)) ** 4
    yy, xx = np.mgrid[0:h, 0:w] + 0.5
    for name, fr in frac.items():
        p, g = params[name], geo[name]
        noise = glass.smooth_noise(rng, (h, w), int(cfg.get("env_cell", 16))).astype(np.float64) * 2 - 1
        t_refl = air_K + p["env_K"] + cfg.get("env_noise_K", 0.0) * noise
        if p["robot"]:
            x, y, z = g["mirror"]
            if z > 0:
                r = fx * cfg["robot"]["radius_m"] / z
                t_refl = np.where((xx - (fx * x / z + cx)) ** 2 + (yy - (fy * y / z + cy)) ** 2 <= r * r, robot_K, t_refl)
        R = schlick(g["cos"], 1 - p["emissivity"] - p["leak"])
        eps = 1 - R - p["leak"]
        pane4 = eps * np.asarray(scene_K, np.float64) ** 4 + R * t_refl ** 4 + p["leak"] * behind4
        out4 = (1 - fr) * out4 + fr * pane4
    return out4 ** 0.25
