"""Training export: one generator dataset folder -> YOLO + COCO, split by scene (never by frame).

usage:
  python3 sim/doorsim/export.py datasets/<name> [--out DIR] [--seed N] [--ratios 0.7 0.15 0.15]

Output (default exports/<name> in the repo; refuses to overwrite an existing export):
  images/<split>/000000.png    symlink to the source rgb.png (RGB only, no copies)
  labels/<split>/000000.txt    YOLO rows: class-1 cx cy w h, normalised; objects with a class and a box only
  data.yaml                    YOLO dataset file (train/val/test folders, names 0: door, 1: handle)
  coco/<split>.json            COCO: images, categories (attribute class ids), annotations with extra fields
                               (object, object_id, size_px, truncated, attributes = object meta without box/joint/*_world)
  export.json                  source dataset, seed, ratios, scenes per split, counts per split
"""
import argparse, json, pathlib, sys
import numpy as np

SPLITS = ("train", "val", "test")
DROP = {"id", "class", "box", "joint"}


def split_scenes(scenes, ratios, seed):
    """Shuffle the sorted unique scenes with the seed; val/test get at least 1 scene each."""
    scenes = sorted(set(scenes))
    order = [scenes[i] for i in np.random.default_rng(seed).permutation(len(scenes))]
    n_val, n_test = max(1, round(ratios[1] * len(scenes))), max(1, round(ratios[2] * len(scenes)))
    return {"train": order[n_val + n_test:], "val": order[:n_val], "test": order[n_val:n_val + n_test]}


def export(src, out, seed=0, ratios=(0.7, 0.15, 0.15)):
    src, out = pathlib.Path(src).resolve(), pathlib.Path(out).resolve()
    if (out / "export.json").exists():
        raise FileExistsError(f"{out} already has an export; pick another --out")
    ds = json.loads((src / "dataset.json").read_text())
    names = {int(k): v for k, v in ds["attributes"]["classes"].items()}
    W, H = ds["camera_intrinsics"]["width"], ds["camera_intrinsics"]["height"]
    metas = {d.name: json.loads((d / "meta.json").read_text()) for d in sorted((src / "frames").iterdir())}
    split_of = {s: sp for sp, scenes in split_scenes([m["scene"] for m in metas.values()], ratios, seed).items() for s in scenes}
    scenes = {sp: sorted(s for s, v in split_of.items() if v == sp) for sp in SPLITS}
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
        for name, o in meta["objects"].items():
            b = o.get("box")
            if o.get("class") is None or b is None:
                continue
            w, h = b["x_max"] - b["x_min"] + 1, b["y_max"] - b["y_min"] + 1  # Gazebo boxes = inclusive pixel indices
            rows.append(f"{o['class'] - 1} {(b['x_min'] + w / 2) / W:.6f} {(b['y_min'] + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
            coco[sp]["annotations"].append({
                "id": len(coco[sp]["annotations"]) + 1, "image_id": image["id"], "category_id": o["class"],
                "bbox": [b["x_min"], b["y_min"], w, h], "area": w * h, "iscrowd": 0, "segmentation": [],
                "object": name, "object_id": o["id"], "size_px": [w, h],
                "truncated": b["x_min"] <= 0 or b["y_min"] <= 0 or b["x_max"] >= W - 1 or b["y_max"] >= H - 1,
                "attributes": {k: v for k, v in o.items() if k not in DROP and not k.endswith(("_world", "_topic"))}})
            counts[sp][names[o["class"]]] += 1
        (out / "labels" / sp / f"{frame}.txt").write_text("".join(r + "\n" for r in rows))
    for sp in SPLITS:
        (out / "coco" / f"{sp}.json").write_text(json.dumps(coco[sp]) + "\n")
    (out / "data.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
                                   + "".join(f"  {i - 1}: {n}\n" for i, n in sorted(names.items())))
    info = {"source": {"path": str(src), "git": ds.get("git"), "seed": ds.get("seed"), "world": ds.get("world"), "frames": ds.get("frames")},
            "seed": seed, "ratios": list(ratios), "scenes": scenes, "counts": counts}
    (out / "export.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--out", help="output folder (default: exports/<dataset name> in the repo)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    args = ap.parse_args()
    src = pathlib.Path(args.dataset).resolve()
    out = pathlib.Path(args.out) if args.out else pathlib.Path(__file__).resolve().parents[2] / "exports" / src.name
    try:
        info = export(src, out, args.seed, tuple(args.ratios))
    except FileExistsError as e:
        sys.exit(str(e))
    print(f"done: {sum(c['images'] for c in info['counts'].values())} images in {out}")


if __name__ == "__main__":
    main()
