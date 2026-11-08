"""Smoke in the thermal image (LWIR), on a dataset with rendered thermal frames (no Gazebo).

LWIR extinction = k x the visible one (k = 0.1 baseline, 0.3 check, Madelief 2026-11-04): along a pixel's ray
t_lwir = exp(-k * integral of beta) = t_vis ** k, with beta from effects.smoke_beta (+ patchiness). The smoke is also a
gas that radiates at its own temperature: radiance ~ T^4 (grey body, Stefan-Boltzmann proxy), integrated as discrete
radiative transfer along the ray: R = t R_scene + sum_i (1 - exp(-tau_i)) R_gas,i exp(-sum_{j<i} tau_j).
Gas temperature: the air everywhere, the ceiling layer's share of beta at its own temperature (fire scenario).
Then thermal noise N(0, NETD). --glass: panes (seg id of glass_*) get the emission / reflection / leak mix of glass_thermal
before the smoke (the leak reads frames/<f>/thermal_<scenario>_behind.png from thermal_render --behind), draws in meta
effects.thermal.glass.

usage:
  python3 sim/doorsim/thermal_fx.py SRC --out DST --scenario room|fire|mix --k 0.1 [--seed 0] [--workers N]
         [--visibility V] [--config sim/configs/effects_train.yaml] [--split test --export-seed N]
         [--glass sim/configs/glass_thermal.yaml]
SRC frames hold thermal_room.png / thermal_fire.png (thermal_render, 320 x 240, 0.01 K) and SRC/thermal/<world>_<scenario>.json
({"air", "layer", "models"}). If SRC is itself an effects.py output (rgb.png symlinks, no thermal files) the thermal files
are taken from its source dataset (dataset.json effects.source). Smoke comes from meta effects.smoke of each frame, so a
fixed-level RGB test set (v4_test_vis<V>) gives the thermal set with the same smoke, frame for frame. --visibility V
instead samples the smoke like `effects.py --visibility V` does (rng [config seed, frame number], same config file), which
reproduces those smoke parameters from a clear base dataset (the RGB / depth of such a SRC stay clear; meta effects.smoke is
set to the sampled smoke). Scenario mix: room or fire per scene (deterministic from seed + scene).
Output: frames/<f>/ symlinks to the source files (rgb, depth, depth_gt, seg, ...), thermal.png (16 bit, 0.01 K), meta.json
with effects.thermal and per object smoke_transmission (LWIR) next to smoke_transmission_vis. Thermal camera:
SRC/thermal/camera.json ({"K": [...]} or the 9 numbers) or else the RGB intrinsics scaled to the thermal image size.
"""
import argparse, json, multiprocessing, os, pathlib, sys
import numpy as np
from PIL import Image

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import effects, export, glass_thermal, thermal

NETD = 0.05  # K, thermal noise (std)


def render(scene_K, depth_m, K_thermal, camera, smoke, k, gas, rng, netd=NETD, samples=32, far=12.0):
    """(thermal image (H, W) K, LWIR transmission (H, W)). depth_m = planar z-depth at the thermal resolution."""
    scene = np.asarray(scene_K, np.float64)
    if not smoke:
        return scene + rng.normal(0, netd, scene.shape), np.ones(scene.shape)
    air = float(gas["air"]) if gas and gas.get("air") is not None else float(scene.mean())
    hot = gas.get("layer") if gas else None
    cam = np.array(camera["world_position"], np.float64)
    pts = effects.world_points(depth_m, K_thermal, camera)
    nodata = np.isnan(pts[..., 0])
    ends = np.where(nodata[..., None], cam + far * effects.ray_directions(scene.shape, K_thermal, camera), pts)
    seg = ends - cam
    ds = np.linalg.norm(seg, axis=-1) / samples
    layer = smoke.get("layer")
    tau, emit = np.zeros(scene.shape), np.zeros(scene.shape)
    for i in range(samples):
        p = cam + seg * ((i + 0.5) / samples)
        beta = effects.smoke_beta(p[..., 2], smoke["visibility_m"], layer)
        if layer and hot is not None:
            w = 1 - effects.smoke_beta(p[..., 2], smoke["visibility_m"], None) / beta  # layer's share of beta here
            r_gas = (1 - w) * air ** 4 + w * float(hot) ** 4
        else:
            r_gas = air ** 4
        if smoke.get("patch", 0) > 0:
            beta = beta * np.maximum(0.0, 1 + smoke["patch"] * effects.value_noise(p, smoke["patch_scale"], smoke["seed"]))
        step = k * beta * ds
        emit += (1 - np.exp(-step)) * r_gas * np.exp(-tau)
        tau += step
    t = np.exp(-tau)
    out = (t * scene ** 4 + emit) ** 0.25
    return out + rng.normal(0, netd, scene.shape), t


def apply(scene_K, depth_m, K_thermal, camera, smoke, k, gas, rng, netd=NETD):
    """Thermal image (H, W) in K: the rendered scene seen through smoke (see module doc) + N(0, netd) noise."""
    return render(scene_K, depth_m, K_thermal, camera, smoke, k, gas, rng, netd)[0]


def object_transmission(t_lwir, seg, attrs):
    """{object: mean LWIR transmission over its pixels}, None when not in view (effects.object_transmission on a given
    t map; t and seg at the same resolution)."""
    return effects.object_transmission(t_lwir, seg, attrs)


# --- dataset ---
def _nearest(a, shape):
    """a resampled to shape by nearest neighbour (pixel centres)."""
    ys = np.minimum((np.arange(shape[0]) + 0.5) * a.shape[0] / shape[0], a.shape[0] - 1).astype(int)
    xs = np.minimum((np.arange(shape[1]) + 0.5) * a.shape[1] / shape[1], a.shape[1] - 1).astype(int)
    return a[np.ix_(ys, xs)]


def thermal_k(root, ds, shape):
    """Thermal intrinsics: root/thermal/camera.json, else the RGB ones scaled to the thermal image shape (H, W)."""
    p = root / "thermal" / "camera.json"
    if p.exists():
        c = json.loads(p.read_text())
        return list(c["K"] if isinstance(c, dict) else c)
    ci = ds["camera_intrinsics"]
    K = list(ci["K"])
    sx, sy = shape[1] / ci["width"], shape[0] / ci["height"]
    K[0] *= sx; K[2] *= sx; K[4] *= sy; K[5] *= sy
    return K


def scene_scenario(scenario, seed, scene):
    if scenario != "mix":
        return scenario
    return "room" if np.random.default_rng([seed, int(scene)]).random() < 0.5 else "fire"


def process_frame(ctx, name, meta):
    src, out, tsrc, ds, K_rgb_ds, scenario, k, seed, netd, cfg, visibility, glass = ctx
    s, d = src / "frames" / name, out / "frames" / name
    d.mkdir(parents=True, exist_ok=True)
    sc = scene_scenario(scenario, seed, meta["scene"])
    tdir = s if (s / f"thermal_{sc}.png").exists() else tsrc / "frames" / name  # effects output: thermal in its source
    scene_K = thermal.from_png16(np.asarray(Image.open(tdir / f"thermal_{sc}.png")))
    gas = json.loads((tsrc / "thermal" / f"{meta['world']}_{sc}.json").read_text())
    Kt = thermal_k(tsrc, ds, scene_K.shape)
    depth = _nearest(effects.depth_m(s / "depth_gt.png"), scene_K.shape)
    seg = _nearest(np.asarray(Image.open(s / "seg.png")), scene_K.shape)
    if visibility is None:
        smoke = (meta.get("effects") or {}).get("smoke")
    else:
        smoke = effects.sample_smoke(np.random.default_rng([cfg["seed"], int(name)]), cfg["smoke"], visibility)
    rng = np.random.default_rng([seed, int(name)])
    drawn = {}
    if glass:  # before the smoke: the pane mix is part of the scene radiance
        frac = glass_thermal.fractions(np.asarray(Image.open(s / "seg.png")), meta, scene_K.shape)
        geo = {n: g for n in frac if (g := glass_thermal.pane_geometry(meta, n)) is not None}
        grng = np.random.default_rng([seed, int(name), 7])  # own stream: frames without glass keep their noise
        drawn = {n: glass_thermal.draw(grng, glass) for n in geo}
        bf = tdir / f"thermal_{sc}_behind.png"
        behind = thermal.from_png16(np.asarray(Image.open(bf))) if bf.exists() else None
        air = float(gas["air"])
        rob = (glass.get("robot") or {}).get("dK")  # the rig has no temperature of its own: robot = air + dK
        robot_K = air + float(grng.uniform(*rob)) if rob else float((gas.get("models") or {}).get("rig", air + 10))
        scene_K = glass_thermal.apply(scene_K, behind, {n: frac[n] for n in geo}, geo, drawn, air, robot_K, Kt, grng, glass)
    img, t = render(scene_K, depth, Kt, meta["camera"], smoke, k, gas, rng, netd)
    Image.fromarray(thermal.to_png16(img)).save(d / "thermal.png")
    for f in s.iterdir():
        if f.name == "meta.json" or f.name.startswith("thermal_"):
            continue
        link = d / f.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.realpath(f))
    if visibility is not None:
        meta.setdefault("effects", {})["smoke"] = smoke
    if smoke:
        for n, v in object_transmission(t, seg, meta).items():
            o = meta["objects"].get(n)
            if o is None:
                continue
            old = o.get("smoke_transmission")
            if old is not None:
                o["smoke_transmission_vis"] = old
            o["smoke_transmission"] = v if v is not None else (None if old is None else old ** k)
    meta.setdefault("effects", {})["thermal"] = {"scenario": sc, "k": k, "netd": netd}
    if glass:
        meta["effects"]["thermal"]["glass"] = drawn
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


_ctx = None


def _init_worker(ctx):
    global _ctx
    _ctx = ctx


def _run_chunk(chunk):
    for name, meta in chunk:
        process_frame(_ctx, name, meta)
    return len(chunk)


def apply_dataset(src, out, scenario, k, seed=0, workers=None, netd=NETD, visibility=None, config=None, split=None, export_seed=0, glass=None):
    """Thermal frames for every frame of dataset `src` -> new dataset folder `out` (FileExistsError if it exists).
    glass = cfg dict of glass_thermal (see sim/configs/glass_thermal.yaml) or None. Returns the new dataset.json dict."""
    src, out = pathlib.Path(src).resolve(), pathlib.Path(out)
    if out.exists():
        raise FileExistsError(f"{out} already exists; pick another --out")
    if scenario not in ("room", "fire", "mix"):
        raise ValueError("scenario must be room, fire or mix")
    ds = json.loads((src / "dataset.json").read_text())
    tsrc = src
    if not (src / "thermal").exists() and (ds.get("effects") or {}).get("source"):
        tsrc = pathlib.Path(ds["effects"]["source"])
    cfg = None
    if visibility is not None:
        import yaml
        cfg = yaml.safe_load(pathlib.Path(config or SIM / "configs" / "effects_train.yaml").read_text())
    metas = {d.name: json.loads((d / "meta.json").read_text()) for d in sorted((src / "frames").iterdir())}
    keep, given = effects.split_frames(ds, metas, split, (0.7, 0.15, 0.15), export_seed)
    out.mkdir(parents=True)
    workers = effects.default_workers() if workers is None else workers
    ctx = (src, out, tsrc, ds, None, scenario, k, seed, netd, cfg, visibility, glass)
    items = list(keep.items())
    if workers <= 1 or len(items) <= 1:
        workers = 1
        for i, (name, meta) in enumerate(items):
            process_frame(ctx, name, meta)
            effects.progress(i, i + 1, len(items))
    else:
        size = max(1, min(10, len(items) // (workers * 4)))
        chunks = [items[i:i + size] for i in range(0, len(items), size)]
        saved = {v: os.environ.get(v) for v in effects.THREAD_VARS}
        os.environ.update({v: "1" for v in effects.THREAD_VARS})
        try:
            with multiprocessing.get_context("spawn").Pool(workers, _init_worker, (ctx,)) as pool:
                n = 0
                for n_done in pool.imap_unordered(_run_chunk, chunks):
                    effects.progress(n, n + n_done, len(items))
                    n += n_done
        finally:
            for v, x in saved.items():
                os.environ.pop(v, None) if x is None else os.environ.__setitem__(v, x)
    info = dict(ds)
    info["frames"] = len(keep)
    if split:
        info["split_scenes"] = given
    info["thermal"] = {"scenario": scenario, "k": k, "seed": seed, "source": str(src)}
    if visibility is not None:
        info["thermal"]["visibility_m"] = visibility
    if glass:
        info["thermal"]["glass"] = glass
    (out / "dataset.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenario", choices=("room", "fire", "mix"), required=True)
    ap.add_argument("--k", type=float, default=0.1, help="LWIR extinction / visible extinction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=effects.default_workers())
    ap.add_argument("--visibility", type=float, help="sample the smoke at this visibility like effects.py --visibility")
    ap.add_argument("--config", help="effects YAML for --visibility (default sim/configs/effects_train.yaml)")
    ap.add_argument("--glass", help="glass_thermal YAML (e.g. sim/configs/glass_thermal.yaml): pane emission / reflection / leak")
    ap.add_argument("--split", choices=export.SPLITS)
    ap.add_argument("--export-seed", type=int, default=0)
    args = ap.parse_args()
    glass = None
    if args.glass:
        import yaml
        glass = yaml.safe_load(pathlib.Path(args.glass).read_text())
    try:
        info = apply_dataset(args.dataset, args.out, args.scenario, args.k, args.seed, args.workers, visibility=args.visibility,
                             config=args.config, split=args.split, export_seed=args.export_seed, glass=glass)
    except FileExistsError as e:
        sys.exit(str(e))
    print(f"done: {info['frames']} frames in {args.out}")


if __name__ == "__main__":
    main()
