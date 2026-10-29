"""Make depth on glass look like a real depth sensor, as a step after rendering.

Gazebo's depth camera sees glass as solid (it ignores transparency and visibility masks). Real depth
cameras mostly see through glass or give no value there. Per frame one mode is picked at random:
- holes: no depth on glass pixels (what ClearGrasp / LIDF train on)
- see_through: the depth of what is behind the glass (needs a second render with the pane moved away)
- partial: smooth random patches of holes / see-through / glass surface
Invalid depth is NaN here and 0 in the saved 16-bit PNG (millimetres, like a RealSense z16 image)."""
import numpy as np
from PIL import Image

MODES = ("holes", "see_through", "partial")


def pick_mode(rng, mix):
    """mix = {"holes": p, "see_through": p, "partial": p} (normalised here)."""
    p = np.array([mix.get(m, 0.0) for m in MODES], float)
    return MODES[rng.choice(len(MODES), p=p / p.sum())]


def smooth_noise(rng, shape, cell):
    """Values in 0..1 that change smoothly over about `cell` pixels."""
    h, w = shape
    coarse = rng.random((max(2, h // cell + 2), max(2, w // cell + 2))).astype(np.float32)
    big = Image.fromarray(coarse, mode="F").resize((w + 2 * cell, h + 2 * cell), Image.BILINEAR)
    return np.asarray(big)[cell:cell + h, cell:cell + w]


def apply(depth_gt, glass_mask, mode, rng, depth_behind=None, partial=None):
    """Sensor-like depth (float32 metres, NaN = invalid) from the rendered depth.
    depth_behind is needed for see_through and partial. partial = {"holes": p, "see_through": p, "cell": px},
    the rest of the glass keeps its surface depth."""
    out = np.where(np.isfinite(depth_gt) & (depth_gt > 0), depth_gt, np.nan).astype(np.float32)
    if not glass_mask.any():
        return out
    if mode in ("see_through", "partial") and depth_behind is None:
        raise ValueError(f"mode {mode} needs depth_behind")
    behind = None if depth_behind is None else np.where(np.isfinite(depth_behind) & (depth_behind > 0), depth_behind, np.nan)
    if mode == "holes":
        out[glass_mask] = np.nan
    elif mode == "see_through":
        out[glass_mask] = behind[glass_mask]
    elif mode == "partial":
        n = smooth_noise(rng, out.shape, partial.get("cell", 32))
        hole = glass_mask & (n < partial["holes"])
        through = glass_mask & ~hole & (n < partial["holes"] + partial["see_through"])
        out[hole] = np.nan
        out[through] = behind[through]
    else:
        raise ValueError(f"unknown glass depth mode {mode}")
    return out


def to_png16(depth):
    """float metres (NaN/inf/<= 0 = invalid) -> uint16 millimetres, 0 = invalid."""
    mm = np.where(np.isfinite(depth) & (depth > 0), np.round(depth * 1000.0), 0)
    return np.clip(mm, 0, 65535).astype(np.uint16)
