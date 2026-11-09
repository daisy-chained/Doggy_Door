"""usage: train/.venv/bin/python train/widen_boxes.py SRC OUT [--class hinge] [--min-width 8]
Training copy of an export with the boxes of one class widened to a minimum width (same centre, kept inside the image).
OUT symlinks everything in SRC except coco/ and export.json; scoring stays on the original export."""
import argparse, copy, json, pathlib, sys


def widen_bbox(bbox, min_w, img_w):
    x, y, w, h = bbox
    if w >= min_w:
        return list(bbox)
    nw = min(min_w, img_w)
    nx = x + w / 2 - nw / 2
    nx = max(0, min(nx, img_w - nw))
    return [nx, y, nw, h]


def widen_coco(coco, name, min_w):
    cats = [c["id"] for c in coco["categories"] if c["name"] == name]
    if not cats:
        raise ValueError(f"no category named {name!r}")
    out = copy.deepcopy(coco)
    widths = {im["id"]: im["width"] for im in out["images"]}
    for a in out["annotations"]:
        if a["category_id"] in cats:
            a["bbox"] = widen_bbox(a["bbox"], min_w, widths[a["image_id"]])
            a["area"] = a["bbox"][2] * a["bbox"][3]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--class", dest="name", default="hinge")
    ap.add_argument("--min-width", type=float, default=8)
    args = ap.parse_args(argv)
    src, out = pathlib.Path(args.src), pathlib.Path(args.out)
    min_w = int(args.min_width) if args.min_width == int(args.min_width) else args.min_width
    if out.exists() or out.is_symlink():
        sys.exit(f"error: {out} exists")
    out.mkdir(parents=True)
    for p in sorted(src.iterdir()):
        if p.name not in ("coco", "export.json"):
            (out / p.name).symlink_to(p.resolve())
    (out / "coco").mkdir()
    for sp in ("train", "val", "test"):
        f = src / "coco" / f"{sp}.json"
        if f.exists():
            wide = widen_coco(json.loads(f.read_text()), args.name, min_w)
            (out / "coco" / f"{sp}.json").write_text(json.dumps(wide))
    exp = json.loads((src / "export.json").read_text())
    exp["widen"] = {"class": args.name, "min_width": min_w, "from": str(src.resolve())}
    (out / "export.json").write_text(json.dumps(exp, indent=1))


if __name__ == "__main__":
    main()
