"""usage: train/.venv/bin/python train/strip_hinges.py SRC OUT [--min-width 8]
Training + scoring copy of an export where the hinge knuckle boxes of each door (per image) are replaced by ONE box: the union
of the door's visible knuckle boxes, widened to a minimum width (same centre, kept inside the image). Hinges without a door
attribute stay single (only widened). OUT symlinks everything in SRC except coco/ and export.json; the YOLO labels/ keep the knuckles."""
import argparse, copy, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import widen_boxes


def _make(ann, bbox, obj, attrs, crowd):
    return {**ann, "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": crowd, "object": obj, "attributes": attrs,
            **({"size_px": [bbox[2], bbox[3]]} if "size_px" in ann else {})}


def strip_coco(coco, min_w=8):
    cats = [c["id"] for c in coco["categories"] if c["name"] == "hinge"]
    if not cats:
        raise ValueError("no category named 'hinge'")
    out = copy.deepcopy(coco)
    widths = {im["id"]: im["width"] for im in out["images"]}
    others, groups, order = [], {}, []
    for a in out["annotations"]:
        if a["category_id"] not in cats:
            others.append(a)
            continue
        door = (a.get("attributes") or {}).get("door")
        if door is None:
            order.append(("single", a))
            continue
        key = (a["image_id"], door, a.get("iscrowd", 0))
        if key not in groups:
            groups[key] = []
            order.append(("group", key))
        groups[key].append(a)
    hinges = []
    for kind, item in order:
        if kind == "single":
            a = item
            bbox = widen_boxes.widen_bbox(a["bbox"], min_w, widths[a["image_id"]])
            hinges.append({**a, "bbox": bbox, "area": bbox[2] * bbox[3], **({"size_px": [bbox[2], bbox[3]]} if "size_px" in a else {})})
            continue
        members = groups[item]
        x0 = min(m["bbox"][0] for m in members)
        y0 = min(m["bbox"][1] for m in members)
        x1 = max(m["bbox"][0] + m["bbox"][2] for m in members)
        y1 = max(m["bbox"][1] + m["bbox"][3] for m in members)
        bbox = widen_boxes.widen_bbox([x0, y0, x1 - x0, y1 - y0], min_w, widths[item[0]])
        hinges.append(_make(members[0], bbox, f"hinge_strip_{item[1]}",
                            {"door": item[1], "knuckles": len(members)}, item[2]))
    out["annotations"] = others + hinges
    for i, a in enumerate(out["annotations"], 1):
        a["id"] = i
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
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
            (out / "coco" / f"{sp}.json").write_text(json.dumps(strip_coco(json.loads(f.read_text()), min_w)))
    exp = json.loads((src / "export.json").read_text())
    exp["strip"] = {"class": "hinge", "min_width": min_w, "from": str(src.resolve())}
    (out / "export.json").write_text(json.dumps(exp, indent=1))


if __name__ == "__main__":
    main()
