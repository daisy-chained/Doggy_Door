"""Training export: one generator dataset folder -> YOLO + COCO, split by scene (never by frame).

usage:
  python3 sim/doorsim/export.py datasets/<name> [--out DIR] [--seed N] [--ratios 0.7 0.15 0.15] [--doorway]

Output (default exports/<name> in the repo; refuses to overwrite an existing export):
  images/<split>/000000.png    symlink to the source rgb.png (RGB only, no copies)
  labels/<split>/000000.txt    YOLO rows: class-1 cx cy w h, normalised; objects with a class and a box only
  data.yaml                    YOLO dataset file (train/val/test folders, names 0: door, 1: handle)
  coco/<split>.json            COCO: images, categories (attribute class ids), annotations with extra fields
                               (object, object_id, size_px, truncated, attributes = object meta without box/joint/*_world)
  export.json                  source dataset, seed, ratios, scenes per split, counts per split (+ ignore setting)

--ignore-below T: door / handle objects whose smoke_transmission is below T get iscrowd 1 + attributes.ignored in the
train / val COCO files (--ignore-splits) and no YOLO row. Madelief 2026-11-03: doors that have nearly vanished in smoke
should neither teach the model to hallucinate doors nor punish it; test sets keep counting them (the visibility curve).
Objects without a smoke_transmission are never ignored.

--doorway: adds class 3 "doorway" (YOLO 2) = the frame opening of every door in view (it has a box) and an "approach" entry (doorsim/doorway.py),
cut to the part the frame's depth shows (depth_gt.png if present, else depth.png). Its smoke_transmission is the door's, so
--ignore-below treats it like the door.
"""
import argparse, json, pathlib, sys
import numpy as np
from PIL import Image

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import doorway as dw

SPLITS = ("train", "val", "test")
DROP = {"id", "class", "box", "joint", "seen_from", "approach", "keep_out"}  # seen_from etc: v4 helpers, the frame has the side already


def split_scenes(scenes, ratios, seed):
    """Shuffle the sorted unique scenes with the seed; val/test get at least 1 scene each."""
    scenes = sorted(set(scenes))
    order = [scenes[i] for i in np.random.default_rng(seed).permutation(len(scenes))]
    n_val, n_test = max(1, round(ratios[1] * len(scenes))), max(1, round(ratios[2] * len(scenes)))
    return {"train": order[n_val + n_test:], "val": order[:n_val], "test": order[n_val:n_val + n_test]}


def export(src, out, seed=0, ratios=(0.7, 0.15, 0.15), ignore_below=None, ignore_splits=("train", "val"), doorway=False):
    src, out = pathlib.Path(src).resolve(), pathlib.Path(out).resolve()
    if (out / "export.json").exists():
        raise FileExistsError(f"{out} already has an export; pick another --out")
    ds = json.loads((src / "dataset.json").read_text())
    names = {int(k): v for k, v in ds["attributes"]["classes"].items()}
    W, H = ds["camera_intrinsics"]["width"], ds["camera_intrinsics"]["height"]
    metas = {d.name: json.loads((d / "meta.json").read_text()) for d in sorted((src / "frames").iterdir())}
    given = ds.get("split_scenes")  # a dataset made for one split (generator --split / effects --split) keeps it
    if given is None:
        given = split_scenes([m["scene"] for m in metas.values()], ratios, seed)
    split_of = {s: sp for sp, scenes in given.items() for s in scenes}
    missing = sorted({m["scene"] for m in metas.values()} - set(split_of))
    if missing:
        raise ValueError(f"scenes {missing} are not in the dataset's split_scenes")
    scenes = {sp: sorted(s for s, v in split_of.items() if v == sp) for sp in SPLITS}
    if doorway:
        names[3] = "doorway"
        K = ds["camera_intrinsics"]["K"]
    coco = {sp: {"images": [], "categories": [{"id": i, "name": n} for i, n in names.items()], "annotations": []} for sp in SPLITS}
    counts = {sp: {"images": 0, **{n: 0 for n in names.values()}} for sp in SPLITS}
    for sp in SPLITS:
        for sub in ("images", "labels", "coco"):
            (out / sub / sp if sub != "coco" else out / sub).mkdir(parents=True, exist_ok=True)
    for frame, meta in metas.items():
        sp = split_of[meta["scene"]]
        (out / "images" / sp / f"{frame}.png").symlink_to(src / "frames" / frame / "rgb.png")
        image = {"id": len(coco[sp]["images"]), "file_name": f"images/{sp}/{frame}.png", "width": W, "height": H,
                 "scene": meta["scene"], "view": meta["view"], "frame": frame}
        coco[sp]["images"].append(image)
        counts[sp]["images"] += 1
        rows = []
        items = [(name, o, o.get("box"), None) for name, o in meta["objects"].items()]
        if doorway:
            f = src / "frames" / frame
            depth = np.asarray(Image.open(f / ("depth_gt.png" if (f / "depth_gt.png").exists() else "depth.png")), np.float32) / 1000.0
            for name, o in meta["objects"].items():
                r = dw.box(meta, name, K, depth) if o.get("class") == 1 and "approach" in o else None
                if r:
                    items.append((f"{name}_doorway", o, r["box"], {"door": name, "visible_fraction": r["visible_fraction"]}))
        for name, o, b, extra in items:
            if o.get("class") is None or b is None:
                continue
            cls, oid = (3, o["id"] + 1000) if extra else (o["class"], o["id"])
            w, h = b["x_max"] - b["x_min"] + 1, b["y_max"] - b["y_min"] + 1  # Gazebo boxes = inclusive pixel indices
            t = o.get("smoke_transmission")
            ignored = ignore_below is not None and sp in ignore_splits and t is not None and t < ignore_below
            if not ignored:
                rows.append(f"{cls - 1} {(b['x_min'] + w / 2) / W:.6f} {(b['y_min'] + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
            coco[sp]["annotations"].append({
                "id": len(coco[sp]["annotations"]) + 1, "image_id": image["id"], "category_id": cls,
                "bbox": [b["x_min"], b["y_min"], w, h], "area": w * h, "iscrowd": int(ignored), "segmentation": [],
                "object": name, "object_id": oid, "size_px": [w, h],
                "truncated": b["x_min"] <= 0 or b["y_min"] <= 0 or b["x_max"] >= W - 1 or b["y_max"] >= H - 1,
                "attributes": {**(extra if extra else {k: v for k, v in o.items() if k not in DROP and not k.endswith(("_world", "_topic"))}),
                               **({"smoke_transmission": t} if extra and t is not None else {}),
                               **({"ignored": True} if ignored else {})}})
            counts[sp][names[cls]] += 1
        (out / "labels" / sp / f"{frame}.txt").write_text("".join(r + "\n" for r in rows))
    for sp in SPLITS:
        (out / "coco" / f"{sp}.json").write_text(json.dumps(coco[sp]) + "\n")
    (out / "data.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
                                   + "".join(f"  {i - 1}: {n}\n" for i, n in sorted(names.items())))
    info = {"source": {"path": str(src), "git": ds.get("git"), "seed": ds.get("seed"), "world": ds.get("world"), "frames": ds.get("frames"),
                       "effects": ds.get("effects"), "lighting": ds.get("lighting")},
            "seed": seed, "ratios": list(ratios), "scenes": scenes, "counts": counts, "doorway": doorway}
    if ignore_below is not None:
        info["ignore"] = {"below": ignore_below, "splits": list(ignore_splits)}
    (out / "export.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--out", help="output folder (default: exports/<dataset name> in the repo)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--ignore-below", type=float, help="ignore objects with smoke_transmission below this (see above)")
    ap.add_argument("--ignore-splits", nargs="+", default=["train", "val"], choices=SPLITS)
    ap.add_argument("--doorway", action="store_true", help="add the doorway class (see above)")
    args = ap.parse_args()
    src = pathlib.Path(args.dataset).resolve()
    out = pathlib.Path(args.out) if args.out else pathlib.Path(__file__).resolve().parents[2] / "exports" / src.name
    try:
        info = export(src, out, args.seed, tuple(args.ratios), args.ignore_below, tuple(args.ignore_splits), args.doorway)
    except FileExistsError as e:
        sys.exit(str(e))
    print(f"done: {sum(c['images'] for c in info['counts'].values())} images in {out}")


if __name__ == "__main__":
    main()
