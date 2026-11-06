"""Adaptive late fusion, Madelief 2026-11-06 option A "quality rule": a per-image "can this sensor see" score made from the model's own
input image, no training. Hand-made version of illumination-aware fusion (IAF R-CNN, Li et al. 2019). Score -> weight in the fusion:
raw score / the run's reference (90th percentile over its validation images), clipped to [floor, 1]."""
import json, pathlib
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


def _detail(grey):
    """Band-pass contrast (blur 2 px minus blur 8 px): ignores grain and slow gradients, drops in smoke."""
    return float(np.std(gaussian_filter(grey, 2) - gaussian_filter(grey, 8)) / 255)


def scores(img, kind):
    """img: PIL image or uint8 array (H,W) / (H,W,3) -> raw quality scores of that input kind."""
    a = np.asarray(img)
    if kind in ("rgb", "thermal"):
        grey = a.astype(float) if a.ndim == 2 else a.astype(float).mean(axis=2)
        return {"bright": float(grey.mean() / 255), "detail": _detail(grey)} if kind == "rgb" else {"detail": _detail(grey)}
    if kind in ("depth", "dha"):
        d = (a if a.ndim == 2 else a[..., 0]).astype(float)  # channel 0 = encoded depth, 0 = invalid
        med = median_filter(d, size=3)
        return {"clean": float(((d > 0) & (np.abs(d - med) <= np.maximum(0.05 * med, 1))).mean())}
    raise ValueError(f"unknown input kind {kind!r}")


def reference(rows):
    """rows: list of score dicts -> 90th percentile per key."""
    return {k: float(np.percentile([r[k] for r in rows], 90)) for k in rows[0]}


def weight(raw, ref, kind, floor=0.05, power=1.0):
    """Ratio of raw scores to the reference, clipped to [0,1] (rgb: the worse of brightness and detail), ** power, at least floor."""
    keys = {"rgb": ("bright", "detail"), "thermal": ("detail",), "depth": ("clean",), "dha": ("clean",)}[kind]
    r = min(float(np.clip(raw[k] / ref[k], 0, 1)) if ref[k] > 0 else 1.0 for k in keys)
    return max(floor, r ** power)


def run_scores(run_dir, export_dir, split):
    """{image_id: raw scores} of a run's own input images."""
    from doordet.data import CocoDetection
    kind = json.loads((pathlib.Path(run_dir) / "config.json").read_text()).get("input", "rgb")
    ds = CocoDetection(export_dir, split, input=kind)
    out = {}
    for i in range(len(ds)):
        img, t = ds[i]
        out[t["image_id"]] = scores(img, kind)
    return out


def run_reference(run_dir, split="val"):
    """Reference of a run: computed on its training export's val images."""
    cfg = json.loads((pathlib.Path(run_dir) / "config.json").read_text())
    return reference(list(run_scores(run_dir, cfg["export"], split).values()))


def adaptive_weights(runs, export_dir, split, floor=0.05, power=1.0):
    """Per-image fusion weights of several runs -> ({image_id: [weight per run]} over images all runs have, mean weight per run)."""
    per_run = []
    for r in runs:
        kind = json.loads((pathlib.Path(r) / "config.json").read_text()).get("input", "rgb")
        ref, raw = run_reference(r), run_scores(r, export_dir, split)
        per_run.append({i: weight(s, ref, kind, floor, power) for i, s in raw.items()})
    ids = set.intersection(*(set(w) for w in per_run))
    iw = {i: [w[i] for w in per_run] for i in ids}
    return iw, [float(np.mean([iw[i][m] for i in iw])) for m in range(len(runs))]
