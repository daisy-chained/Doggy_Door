"""Box check for a trained run: how tight are the predicted boxes, and what do they look like?

usage:
  train/.venv/bin/python train/check_boxes.py runs/<name> [--split test] [--score 0.5] [--sheet 8]

Uses the run's config.json + best/ weights. Prints per class: matched boxes, misses, extra boxes and the
mean / 10th-percentile IoU of matched boxes (greedy match, IoU >= 0.1). Writes runs/<name>/boxes_<split>.png:
ground truth in white, predictions in red (door) / green (handle) with their score.
"""
import argparse, json, pathlib, sys
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from doordet import adapters
from doordet.data import CocoDetection
from torch.utils.data import DataLoader


def iou(a, b):
    """a: (N, 4), b: (M, 4) xyxy → (N, M)."""
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], axis=1)
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(gt, pred):
    """Greedy by IoU. Returns matched IoUs, n missed gt, n extra predictions."""
    if not len(gt) or not len(pred):
        return [], len(gt), len(pred)
    m, ious, used_g, used_p = iou(gt, pred), [], set(), set()
    for g, p in sorted(np.argwhere(m >= 0.1), key=lambda gp: -m[gp[0], gp[1]]):
        if g not in used_g and p not in used_p:
            used_g.add(g); used_p.add(p); ious.append(float(m[g, p]))
    return ious, len(gt) - len(used_g), len(pred) - len(used_p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--split", default="test")
    ap.add_argument("--score", type=float, default=0.5, help="min score for a prediction to count")
    ap.add_argument("--sheet", type=int, default=8, help="images on the contact sheet")
    args = ap.parse_args()
    run = pathlib.Path(args.run)
    cfg = json.loads((run / "config.json").read_text())
    cfg["model"]["checkpoint"] = str(run / "best")
    ds = CocoDetection(cfg["export"], args.split)
    adapter = adapters.get(cfg["model"]["family"])(cfg, ds.id2label)
    device = torch.device(cfg["device"])
    adapter.model.to(device).eval()
    preds = {}
    for inputs in DataLoader(ds, batch_size=cfg["batch"], collate_fn=adapter.collate):
        preds.update(adapter.predict(inputs, device))

    stats = {c: {"ious": [], "missed": 0, "extra": 0, "gt": 0} for c in ds.id2label}
    sheet = []
    for i in range(len(ds)):
        img, t = ds[i]
        p = dict(preds[t["image_id"]])
        b = p["boxes"].reshape(-1, 4)
        p["boxes"] = np.concatenate([np.minimum(b[:, :2], b[:, 2:]), np.maximum(b[:, :2], b[:, 2:])], axis=1)  # untrained models can flip corners
        keep = p["scores"] >= args.score
        for c in ds.id2label:
            g = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                          for a in t["annotations"] if a["category_id"] == c], float).reshape(-1, 4)
            ious, missed, extra = match(g, p["boxes"][keep & (p["labels"] == c)])
            s = stats[c]; s["ious"] += ious; s["missed"] += missed; s["extra"] += extra; s["gt"] += len(g)
        if len(sheet) < args.sheet and i % max(len(ds) // args.sheet, 1) == 0:
            im = img.convert("RGB"); d = ImageDraw.Draw(im)
            for a in t["annotations"]:
                x, y, w, h = a["bbox"]; d.rectangle([x, y, x + w, y + h], outline=(255, 255, 255), width=1)
            for b, s_, l in zip(p["boxes"][keep], p["scores"][keep], p["labels"][keep]):
                col = (255, 0, 0) if l == 0 else (0, 255, 0)
                d.rectangle(list(b), outline=col, width=2); d.text((b[0] + 2, b[1] + 2), f"{s_:.2f}", fill=col)
            sheet.append(im)
    for c, s in stats.items():
        ious = np.array(s["ious"])
        print(f"{ds.id2label[c]:7s} gt {s['gt']:4d}  matched {len(ious):4d}  missed {s['missed']:3d}  extra {s['extra']:3d}  "
              f"IoU mean {ious.mean() if len(ious) else 0:.3f}  p10 {np.percentile(ious, 10) if len(ious) else 0:.3f}")
    if sheet:
        w, h = sheet[0].size
        out = Image.new("RGB", (w * 2, h * ((len(sheet) + 1) // 2)))
        for k, im in enumerate(sheet):
            out.paste(im, ((k % 2) * w, (k // 2) * h))
        out.save(run / f"boxes_{args.split}.png")
        print(f"sheet: {run / f'boxes_{args.split}.png'}")


if __name__ == "__main__":
    main()
