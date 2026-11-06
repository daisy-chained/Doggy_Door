"""Radar BEV training export (Madelief 2026-11-04, "just do both"): the radar's raw range x azimuth power map as a top-down
(BEV) image for the D-FINE doorway detector (class "doorway"). One generator dataset with radar renders (radar_render.py) ->
the same layout export.py writes, but the images are BEV pictures.

usage:
  python3 sim/doorsim/radar_export.py datasets/door_room_v4_2k --model single|imaging --views 1|3 --out exports/radar_<model>_v<views>
         [--size-m 8.0] [--res 0.05] [--workers N] [--limit N]

Per split (dataset.json split_scenes): one image per target frame. --views 1: every frame, its own map. --views 3: the windows of
radar_doorways.view_windows (last 3 views of the same world, scene and side; targets = frames with 2 earlier views), the maps of
the 3 frames placed in the target's frame (each with its own camera pose) and combined (radar_heatmap.bev). 8-bit grey PNG in 3
channels (the RGB loader reads it): dB mapped from [noise floor, noise floor + 60 dB] to 0..255 (limits in export.json).
  images/<split>/<frame>.png, coco/<split>.json (images, annotations, categories [{"id": 1, "name": "doorway"}]), export.json.
Annotations: the doorway segment in BEV pixels (radar_heatmap.bev_box), attributes door, state, distance_m, in_view (the door's
centre is in the target frame's radar view and not hidden, radar_doorways.in_view). Only OPEN doorways are labelled; a closed door
is wall to a radar (Claude decision, Madelief can overrule): closed and out-of-view / hidden ones stay as ignore boxes (iscrowd 1 + attributes.ignored, like
export.py --ignore-below): scored as ignore, left out of the training targets.
"""
import argparse, json, multiprocessing, os, pathlib, sys, time, zlib
import numpy as np
from PIL import Image

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import effects, export, radar, radar_heatmap as rh

SPLITS = ("train", "val", "test")
DB_RANGE = 60.0             # dB above the noise floor mapped to 255
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
_ctx = None
_mats = {}


def to_png(img_db):
    """dB image -> uint8 (H, W, 3) grey."""
    g = np.clip((img_db - rh.NOISE_DB) / DB_RANGE, 0, 1) * 255
    return np.repeat(g.round().astype(np.uint8)[..., None], 3, axis=2)


def _init(ctx):
    global _ctx
    _ctx = ctx


def _map(frame):
    cfg = _ctx
    root, ds, K, model = cfg["root"], cfg["ds"], cfg["K"], cfg["model"]
    meta = json.loads((root / "frames" / frame / "meta.json").read_text())
    fd = root / "frames" / frame
    wf = effects.world_file(ds, meta)
    if wf not in _mats:
        _mats[wf] = radar.materials(json.loads((wf.parent / f"{wf.stem}.attributes.json").read_text()))
    rng = np.random.default_rng(zlib.crc32(f"{frame}".encode()))
    m = rh.ra_map(effects.depth_m(fd / "radar_depth.png"), np.asarray(Image.open(fd / "radar_seg.png")),
                  effects.depth_m(fd / "radar_depth_noglass.png"), K, meta["camera"], radar.MODELS[model], _mats[wf], rng)
    return m, meta


def process(task):
    """(split, target, window frames) -> (task, annotation dicts without ids); writes the PNG."""
    from doorsim import radar_doorways as rd
    split, target, frames = task
    cfg = _ctx
    maps, cams, tmeta = [], [], None
    for f in frames:
        m, meta = _map(f)
        maps.append(m)
        cams.append(meta["camera"])
        if f == target:
            tmeta = meta
    img = rh.bev(maps, cams, tmeta["camera"], cfg["size_m"], cfg["res"])
    out = cfg["out"] / "images" / split / f"{target}.png"
    Image.fromarray(to_png(img)).save(out)
    tdepth = effects.depth_m(cfg["root"] / "frames" / target / "radar_depth.png")
    cam = tmeta["camera"]
    anns = []
    for name, g in rd.ground_truth(tmeta).items():
        b = rh.bev_box(g, cam, cfg["size_m"], cfg["res"])
        if b is None:
            continue
        x0, y0, x1, y1 = b
        seen = bool(rd.in_view(g, cam, cfg["K"], tdepth))
        ignored = g["state"] != "open" or not seen  # closed = wall to a radar; out of view / hidden = nothing to learn from
        dist = float(np.hypot(*(np.array(g["center"][:2]) - np.array(cam["world_position"][:2]))))
        anns.append({"category_id": 1, "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0),
                     "iscrowd": int(ignored), "segmentation": [], "object": name, "size_px": [x1 - x0, y1 - y0],
                     "attributes": {"door": name, "state": g["state"], "distance_m": dist,
                                    "in_view": seen, **({"ignored": True} if ignored else {})}})
    return task, anns


def closed_as_negatives(coco):
    """Copy of a radar COCO dict where closed doors in view have no box (background: punished if detected) instead of an ignore
    box. Out-of-view / hidden doors stay ignore. (2026-11-06 sanity check, Claude: with closed = ignore the learned detector
    also "found" 32-67 % of closed doors.)"""
    out = dict(coco)
    out["annotations"] = [dict(a) for a in coco["annotations"]
                          if not (a["attributes"].get("state") != "open" and a["attributes"].get("in_view"))]
    return out


def relabel_closed(src, out):
    """Radar export -> new export with closed_as_negatives labels; images are a symlink to the source's (no re-render)."""
    src, out = pathlib.Path(src).resolve(), pathlib.Path(out).resolve()
    if (out / "export.json").exists():
        raise FileExistsError(f"{out} already has an export; pick another --out")
    (out / "coco").mkdir(parents=True, exist_ok=True)
    if not (out / "images").exists():
        (out / "images").symlink_to(src / "images")
    for f in sorted((src / "coco").glob("*.json")):
        (out / "coco" / f.name).write_text(json.dumps(closed_as_negatives(json.loads(f.read_text()))))
    info = json.loads((src / "export.json").read_text())
    info.setdefault("labels", {})["closed"] = "negative"
    info["labels"]["closed_note"] = "closed doors in view have no box (background); out-of-view / hidden stay ignore (sanity check 2026-11-06)"
    info["relabelled_from"] = str(src)
    (out / "export.json").write_text(json.dumps(info, indent=2) + "\n")
    return out


def export_radar(src, out, model, views, size_m=8.0, res=0.05, workers=None, limit=None):
    from doorsim import radar_doorways as rd
    src, out = pathlib.Path(src).resolve(), pathlib.Path(out).resolve()
    if (out / "export.json").exists():
        raise FileExistsError(f"{out} already has an export; pick another --out")
    ds = json.loads((src / "dataset.json").read_text())
    K = json.loads((src / "radar" / "camera.json").read_text())["K"]
    metas = {d.name: json.loads((d / "meta.json").read_text()) for d in sorted((src / "frames").iterdir()) if (d / "meta.json").exists()}
    rendered = {n for n in metas if (src / "frames" / n / "radar_depth_noglass.png").exists()}
    given = ds.get("split_scenes") or export.split_scenes([m["scene"] for m in metas.values()], (0.7, 0.15, 0.15), 0)
    split_of = {s: sp for sp, sc in given.items() for s in sc}
    tasks = []  # (split, target, window frames)
    if views == 1:
        tasks = [(split_of[m["scene"]], n, [n]) for n, m in metas.items() if n in rendered]
    else:
        for t, fr in rd.view_windows({n: metas[n] for n in sorted(rendered)}, [views])[views]:
            tasks.append((split_of[metas[t]["scene"]], t, fr))
    skipped = len(metas) - len(rendered)
    tasks.sort(key=lambda t: (SPLITS.index(t[0]), t[1]))
    tasks = tasks[:limit]
    for sp in SPLITS:
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
    (out / "coco").mkdir(exist_ok=True)
    cfg = {"root": src, "ds": ds, "K": K, "model": model, "size_m": size_m, "res": res, "out": out}
    workers = effects.default_workers() if workers is None else workers
    t0 = time.time()
    results = []
    if workers <= 1:
        _init(cfg)
        for k, t in enumerate(tasks):
            results.append(process(t))
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(tasks)}", flush=True)
    else:
        saved = {k: os.environ.get(k) for k in THREAD_VARS}
        os.environ.update({k: "1" for k in THREAD_VARS})
        try:
            with multiprocessing.get_context("spawn").Pool(workers, _init, (cfg,)) as pool:
                for k, r in enumerate(pool.imap_unordered(process, tasks, chunksize=8)):
                    results.append(r)
                    if (k + 1) % 200 == 0:
                        print(f"{k + 1}/{len(tasks)}", flush=True)
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    by_task = {(t[0], t[1]): a for t, a in results}
    W = int(round(size_m / res))
    coco = {sp: {"images": [], "categories": [{"id": 1, "name": "doorway"}], "annotations": []} for sp in SPLITS}
    counts = {sp: {"images": 0, "doorway": 0, "ignored": 0, "in_view": 0} for sp in SPLITS}
    for t in tasks:
        sp, target, frames = t
        c = coco[sp]
        image = {"id": len(c["images"]), "file_name": f"images/{sp}/{target}.png", "width": W, "height": W,
                 "scene": metas[target]["scene"], "view": metas[target]["view"], "frame": target, "frames": frames}
        c["images"].append(image)
        counts[sp]["images"] += 1
        for a in by_task[sp, target]:
            c["annotations"].append({"id": len(c["annotations"]) + 1, "image_id": image["id"], **a})
            counts[sp]["ignored" if a["iscrowd"] else "doorway"] += 1
            counts[sp]["in_view"] += int(a["attributes"]["in_view"] and not a["iscrowd"])
    for sp in SPLITS:
        (out / "coco" / f"{sp}.json").write_text(json.dumps(coco[sp]) + "\n")
    info = {"source": {"path": str(src), "git": ds.get("git"), "seed": ds.get("seed"), "world": ds.get("world"), "frames": ds.get("frames")},
            "kind": "radar_bev", "model": model, "views": views, "size_m": size_m, "res": res, "image_px": W,
            "layout": "robot (target frame's camera) at bottom centre, forward = up, left = left",
            "png": {"channels": 3, "db_low": rh.NOISE_DB, "db_high": rh.NOISE_DB + DB_RANGE,
                    "note": "8-bit = (dB - db_low) / (db_high - db_low) x 255, clipped; dB = 10 log10 of radar power (radar.NOISE_FLOOR units absolute)"},
            "combine": "mean of linear power over the maps covering a pixel; outside every map = noise floor",
            "labels": {"classes": {"1": "doorway"}, "open_only": True,
                       "note": "Claude decision (Madelief can overrule): a closed door is wall to a radar, so only OPEN doorways are labelled; "
                               "closed ones are ignore boxes (iscrowd 1, attributes.ignored), left out of the training targets",
                       "box": "doorway segment: centre +- width / 2 along the wall, +- 0.15 m across, axis-aligned in BEV pixels",
                       "in_view": "attribute: the doorway centre is in the target frame's radar view and not hidden"},
            "seed": "per frame: crc32(frame name)", "scenes": {sp: sorted(set(s)) for sp, s in given.items()},
            "counts": counts, "frames_not_rendered": skipped, "seconds": round(time.time() - t0, 1), "workers": workers}
    (out / "export.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--model", required=True, choices=sorted(radar.MODELS))
    ap.add_argument("--views", type=int, required=True, choices=(1, 3))
    ap.add_argument("--out", required=True)
    ap.add_argument("--size-m", type=float, default=8.0)
    ap.add_argument("--res", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    try:
        info = export_radar(a.dataset, a.out, a.model, a.views, a.size_m, a.res, a.workers, a.limit)
    except FileExistsError as e:
        sys.exit(str(e))
    print(f"done: {sum(c['images'] for c in info['counts'].values())} images in {a.out} ({info['seconds']} s)")


if __name__ == "__main__":
    main()
