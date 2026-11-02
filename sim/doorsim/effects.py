"""Image effects on a saved generator dataset (no Gazebo): room smoke from depth, matching depth effects, camera noise.

Smoke: Beer-Lambert along each pixel's line of sight, t = exp(-integral of beta), beta = 3 / visibility (contrast drops to
5 % at the visibility distance), optionally denser above a height (hot smoke layer) and patchy (smooth 3D noise).
Image: I = J t + A (1 - t), in linear light; the airlight A follows the scene's own (blurred) brightness, so smoke glows
where the scene is lit (headlight cone) and stays dark in darkness. Depth: pixels in thick smoke get lost or return a
false close hit. Camera: shot + read noise with auto-gain up to a cap.

usage:
  python3 sim/doorsim/effects.py datasets/<name> --config sim/configs/effects_train.yaml [--out DIR]
         [--visibility V|inf] [--split test --export-seed N]
Random mode (no --visibility): per frame smoke or not, random level (training sets). --visibility V: every frame at that
visibility with the same layer / patch as the other levels (inf = camera noise only). --split keeps only that split's
scenes, written into dataset.json so export.py keeps them as that split.

Output: a dataset folder like the source (rgb.png, depth.png new; seg.png, depth_gt.png symlinked), meta.json with
"effects": {"smoke": params or None, "noise_gain"} and objects[...]["smoke_transmission"], dataset.json with "effects".
"""
import argparse, json, math, pathlib, sys
import numpy as np
import yaml
from PIL import Image

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import export, frames, glass

GAMMA = 2.2
NOISE_STD = 0.646  # std of the interpolated lattice noise below (so n has ~unit std)
LATTICE = 32       # noise repeats every 32 cells


def to_linear(rgb):
    return (rgb.astype(np.float32) / 255.0) ** GAMMA


def to_srgb(lin):
    return np.round(np.clip(lin, 0.0, 1.0) ** (1 / GAMMA) * 255.0).astype(np.uint8)


# --- geometry ---
def world_points(depth, K, camera):
    """(H, W, 3) world points of the depth pixels; depth = planar z-depth in m (0 / non-finite = nan).
    Optical frame (x right, y down, z forward) -> camera body (z, -x, -y) -> world."""
    H, W = depth.shape
    fx, cx, fy, cy = K[0], K[2], K[4], K[5]
    z = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan).astype(np.float64)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    body = np.stack([z, -(u - cx) / fx * z, -(v - cy) / fy * z], axis=-1)
    R = np.array(frames.rot(*camera["world_rpy"]))
    return body @ R.T + np.array(camera["world_position"])


def ray_directions(shape, K, camera):
    """(H, W, 3) unit world-frame direction of each pixel's ray."""
    H, W = shape
    d = world_points(np.ones(shape), K, camera) - np.array(camera["world_position"])
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


# --- smoke ---
def smoke_beta(z, visibility_m, layer):
    """Extinction (1/m) at height z: 3 / V, times a soft step up to `factor` above layer height."""
    beta = np.full(np.shape(z), 3.0 / visibility_m)
    if layer:
        beta = beta * (1 + (layer["factor"] - 1) / (1 + np.exp(-(np.asarray(z) - layer["height"]) / layer["width"])))
    return beta


def value_noise(p, cell, seed):
    """Smooth zero-mean ~unit-std noise at points p (..., 3) in m: random lattice values (cell size `cell` m, seeded),
    trilinear with smoothstep."""
    lattice = np.random.default_rng(seed).standard_normal((LATTICE,) * 3)
    q = p / cell
    i0 = np.floor(q).astype(np.int64)
    f = q - i0
    f = f * f * (3 - 2 * f)
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[..., 0] if dx else 1 - f[..., 0]) * (f[..., 1] if dy else 1 - f[..., 1]) * (f[..., 2] if dz else 1 - f[..., 2])
                out = out + w * lattice[(i0[..., 0] + dx) % LATTICE, (i0[..., 1] + dy) % LATTICE, (i0[..., 2] + dz) % LATTICE]
    return out / NOISE_STD


def transmission(depth, K, camera, smoke, samples=24, far=12.0):
    """(H, W) transmission exp(-tau) from the camera to each pixel's 3D point (tau = integral of beta, midpoint rule).
    Pixels without depth look `far` m along their ray. smoke = {visibility_m, layer, patch, patch_scale, seed}."""
    cam = np.array(camera["world_position"], np.float64)
    pts = world_points(depth, K, camera)
    nodata = np.isnan(pts[..., 0])
    ends = np.where(nodata[..., None], cam + far * ray_directions(depth.shape, K, camera), pts)
    seg = ends - cam
    dist = np.linalg.norm(seg, axis=-1)
    tau = np.zeros(depth.shape)
    for k in range(samples):
        p = cam + seg * ((k + 0.5) / samples)
        beta = smoke_beta(p[..., 2], smoke["visibility_m"], smoke.get("layer"))
        if smoke.get("patch", 0) > 0:
            beta = beta * np.maximum(0.0, 1 + smoke["patch"] * value_noise(p, smoke["patch_scale"], smoke["seed"]))
        tau += beta
    return np.exp(-tau * dist / samples)


def blur(a, sigma):
    """Gaussian blur of a 2D float array (sigma px): box-downsample, blur, upsample (edge values extend)."""
    f = max(1, int(sigma // 4))
    H, W = a.shape
    ph, pw = -H % f, -W % f
    small = np.pad(a, ((0, ph), (0, pw)), mode="edge").reshape((H + ph) // f, f, (W + pw) // f, f).mean(axis=(1, 3))
    s = sigma / f
    r = max(1, int(math.ceil(3 * s)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2)
    k /= k.sum()
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        padded = np.pad(small, pad, mode="edge")
        small = sum(w * np.take(padded, range(i, i + small.shape[axis]), axis=axis) for i, w in enumerate(k))
    big = Image.fromarray(small.astype(np.float32), mode="F").resize((W + pw, H + ph), Image.BILINEAR)
    return np.asarray(big)[:H, :W]


def add_smoke(rgb, t, colour, illum_sigma=40, illum_ref=0.35):
    """I = J t + A (1 - t) in linear light; A = colour x blurred scene brightness / illum_ref (0..1.5).
    t == 1 leaves a pixel unchanged."""
    t = np.asarray(t, np.float32)
    if (t >= 1).all():
        return rgb.copy()
    J = to_linear(rgb)
    lum = J @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    A = np.array(colour, np.float32) * np.clip(blur(lum, illum_sigma) / illum_ref, 0, 1.5)[..., None]
    out = to_srgb(J * t[..., None] + A * (1 - t[..., None]))
    return np.where((t >= 1)[..., None], rgb, out)


def smoke_depth(depth_m, t, rng, cfg):
    """Depth sensor in smoke: a pixel is lost with probability 1 - clip(t / t_lost, 0, 1); a lost pixel gives a false
    close return (depth x f, f = first-scatter point's share of the path) with probability p_false, else 0.
    Pixels without depth stay as they are."""
    depth = np.asarray(depth_m, np.float32)
    t = np.clip(np.asarray(t, np.float64), 1e-6, 1.0)
    lost = rng.random(depth.shape) < 1 - np.clip(t / cfg["t_lost"], 0, 1)
    false = lost & (rng.random(depth.shape) < cfg["p_false"])
    r = np.minimum(rng.random(depth.shape), 1 - 1e-9)
    tau = np.maximum(-np.log(t), 1e-9)
    f = -np.log(1 - r * (1 - t)) / tau
    valid = np.isfinite(depth) & (depth > 0)
    out = depth.copy()
    out[lost & valid] = 0
    out[false & valid] = (depth * f)[false & valid]
    return out


# --- camera ---
def sensor_noise(rgb, rng, cfg):
    """(uint8 image, gain): auto-gain to a target mean brightness (1..max_gain), then Poisson shot noise + Gaussian read
    noise on the electrons. A scene darker than max_gain can fix stays dark."""
    lin = to_linear(rgb)
    mean = float(lin.mean())
    gain = cfg["max_gain"] if mean <= 0 else float(np.clip(cfg["target_mean"] / mean, 1, cfg["max_gain"]))
    e = rng.poisson(lin * cfg["full_well"]) + rng.normal(0, cfg["read_noise"], lin.shape)
    return to_srgb(e / cfg["full_well"] * gain), float(gain)


def object_transmission(t, seg, attrs):
    """{object: mean transmission over its pixels (its id + ids of its parts)} for objects with a class; None when
    not in view (same grouping as clutter.visible_fraction)."""
    objs = attrs["objects"]
    out = {}
    for name, o in objs.items():
        if o.get("class") is None:
            continue
        ids = [o["id"]] + [q["id"] for q in objs.values() if q.get("part_of") == name]
        m = np.isin(seg, ids)
        out[name] = float(np.asarray(t)[m].mean()) if m.any() else None
    return out


# --- sampling ---
def sample_smoke(rng, cfg, visibility=None):
    """Smoke parameters for one frame, or None (no smoke). Random mode: smoke with chance p, visibility log-uniform.
    Fixed mode (visibility given): the same random draws in the same order, so layer / patch / seed are identical across
    levels for the same rng; inf = no smoke (camera noise only)."""
    on = rng.random() < cfg["p"]
    lo, hi = cfg["visibility_m"]
    vis = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    L = cfg["layer"]
    has_layer = rng.random() < L["p"]
    layer = {"height": float(rng.uniform(*L["height"])), "factor": float(rng.uniform(*L["factor"])), "width": L["width"]}
    patch = float(rng.uniform(*cfg["patch"]))
    seed = int(rng.integers(2**31))
    if visibility is None:
        if not on:
            return None
    elif math.isinf(visibility):
        return None
    else:
        vis = float(visibility)
    return {"visibility_m": vis, "layer": layer if has_layer else None, "patch": patch,
            "patch_scale": cfg["patch_scale"], "seed": seed}


# --- dataset ---
def depth_m(path):
    return np.asarray(Image.open(path)).astype(np.float32) / 1000.0


def apply_dataset(src, out, cfg, visibility=None, split=None, export_seed=0, ratios=(0.7, 0.15, 0.15)):
    """Effects on every frame (or only the scenes of `split`) of a generator dataset -> new dataset folder `out`.
    Returns the new dataset.json dict."""
    src, out = pathlib.Path(src).resolve(), pathlib.Path(out)
    if (out / "dataset.json").exists():
        raise FileExistsError(f"{out} already has a dataset; pick another --out")
    ds = json.loads((src / "dataset.json").read_text())
    K, attrs = ds["camera_intrinsics"]["K"], ds["attributes"]
    metas = {d.name: json.loads((d / "meta.json").read_text()) for d in sorted((src / "frames").iterdir())}
    keep = metas
    given = None
    if split:
        given = export.split_scenes([m["scene"] for m in metas.values()], ratios, export_seed)
        keep = {n: m for n, m in metas.items() if m["scene"] in set(given[split])}
    sc = cfg["smoke"]
    out.mkdir(parents=True, exist_ok=True)
    for k, (name, meta) in enumerate(keep.items()):
        rng = np.random.default_rng([cfg["seed"], int(name)])
        s, d = src / "frames" / name, out / "frames" / name
        d.mkdir(parents=True, exist_ok=True)
        rgb = np.asarray(Image.open(s / "rgb.png").convert("RGB"))
        depth, depth_gt = depth_m(s / "depth.png"), depth_m(s / "depth_gt.png")
        seg = np.asarray(Image.open(s / "seg.png"))
        smoke = sample_smoke(rng, sc, visibility)
        if smoke:
            t = transmission(depth_gt, K, meta["camera"], smoke, sc.get("samples", 24), sc.get("far", 12.0))
            rgb = add_smoke(rgb, t, sc["colour"], sc.get("illum_sigma", 40), sc.get("illum_ref", 0.35))
            depth = smoke_depth(depth, t, rng, cfg["depth"])
            for n, v in object_transmission(t, seg, attrs).items():
                if n in meta["objects"]:
                    meta["objects"][n]["smoke_transmission"] = v
        rgb, gain = sensor_noise(rgb, rng, cfg["noise"])
        Image.fromarray(rgb).save(d / "rgb.png")
        Image.fromarray(glass.to_png16(depth)).save(d / "depth.png")
        for f in ("seg.png", "depth_gt.png"):
            (d / f).symlink_to((s / f).resolve())
        meta["effects"] = {"smoke": smoke, "noise_gain": gain}
        (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        if (k + 1) % 50 == 0 or k + 1 == len(keep):
            print(f"{k + 1}/{len(keep)} frames", flush=True)
    info = dict(ds)
    info["frames"] = len(keep)
    if split:
        info["split_scenes"] = given
    info["effects"] = {"source": str(src), "config": cfg, "visibility_m": visibility, "split": split}
    (out / "dataset.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--config", required=True, help="effects YAML (sim/configs/effects_train.yaml)")
    ap.add_argument("--out", help="output folder (default: datasets/<name>_fx, or _vis<V> with --visibility)")
    ap.add_argument("--visibility", type=float, help="fixed visibility in m for every frame (inf = camera noise only)")
    ap.add_argument("--split", choices=export.SPLITS, help="only this split's scenes")
    ap.add_argument("--export-seed", type=int, default=0, help="seed of that split (export.py --seed)")
    args = ap.parse_args()
    src = pathlib.Path(args.dataset).resolve()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    tag = "_fx" if args.visibility is None else f"_vis{args.visibility:g}"
    out = pathlib.Path(args.out) if args.out else SIM.parent / "datasets" / (src.name + tag)
    try:
        info = apply_dataset(src, out, cfg, args.visibility, args.split, args.export_seed)
    except FileExistsError as e:
        sys.exit(str(e))
    print(f"done: {info['frames']} frames in {out}")


if __name__ == "__main__":
    main()
