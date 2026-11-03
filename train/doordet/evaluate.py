"""Predictions -> COCO results -> mAP (overall, by size, per class) with pycocotools."""
import contextlib, io, json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

STATS = ("map", "map_50", "map_75", "map_small", "map_medium", "map_large")


def to_coco_results(preds):
    out = []
    for image_id, p in preds.items():
        for (x1, y1, x2, y2), s, l in zip(np.asarray(p["boxes"], float).reshape(-1, 4), p["scores"], p["labels"]):
            out.append({"image_id": int(image_id), "category_id": int(l) + 1,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})
    return out


def coco_metrics(gt_json, preds):
    gt = json.loads(open(gt_json).read())
    for a in gt["annotations"]:
        a["id"] += 1  # pycocotools treats id 0 as 'unmatched'; exports made before 2026-10-29 number annotations from 0
    names = {c["id"]: c["name"] for c in gt["categories"]}
    metrics = {k: 0.0 for k in (*STATS, *(f"ap_{n}" for n in names.values()))}
    results = to_coco_results(preds)
    if not results:
        return metrics
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        ev = COCOeval(coco_gt, coco_gt.loadRes(results), "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    for k, v in zip(STATS, ev.stats[:6]):
        metrics[k] = max(float(v), 0.0)
    for k, cid in enumerate(ev.params.catIds):
        p = ev.eval["precision"][:, :, k, 0, 2]
        p = p[p > -1]
        metrics[f"ap_{names[cid]}"] = float(p.mean()) if p.size else 0.0
    return metrics


def iou_xywh(a, b):
    """IoU of one xywh box with an (n, 4) array of xywh boxes."""
    a, b = np.asarray(a, float), np.asarray(b, float).reshape(-1, 4)
    iw = np.clip(np.minimum(a[0] + a[2], b[:, 0] + b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[1] + a[3], b[:, 1] + b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = iw * ih
    return inter / np.maximum(a[2] * a[3] + b[:, 2] * b[:, 3] - inter, 1e-9)


def match(gt_boxes, pred_boxes, scores, iou=0.5):
    """Greedy by descending score: each prediction takes the free GT with the highest IoU >= iou (xywh boxes). -> {gt index: pred index}"""
    taken, gt_boxes = {}, np.asarray(gt_boxes, float).reshape(-1, 4)
    for p in np.argsort(-np.asarray(scores, float), kind="stable"):
        if len(gt_boxes) == 0:
            break
        ious = iou_xywh(pred_boxes[p], gt_boxes)
        ious[list(taken)] = -1
        g = int(ious.argmax())
        if ious[g] >= iou:
            taken[g] = int(p)
    return taken


def recall_by_distance(gt_json, preds, category, bins, score_min=0.3, iou=0.5):
    """Share of `category` GT boxes (not ignore, with box_3d) found by a prediction >= score_min, per camera distance bin
    (norm of the box_3d centre, m): {"lo-hi": {"n", "found", "recall"}}; recall None for an empty bin."""
    gt = json.loads(open(gt_json).read())
    cid = next(c["id"] for c in gt["categories"] if c["name"] == category)
    fmt = lambda v: str(int(v)) if float(v) == int(v) else str(v)
    out = {f"{fmt(lo)}-{fmt(hi)}": {"n": 0, "found": 0, "recall": None} for lo, hi in zip(bins[:-1], bins[1:])}
    by_image = {}
    for a in gt["annotations"]:
        if a["category_id"] == cid and not a.get("iscrowd", 0):
            by_image.setdefault(a["image_id"], []).append(a)
    for image_id, anns in by_image.items():
        p = preds.get(image_id) or preds.get(str(image_id)) or {"boxes": [], "scores": [], "labels": []}
        keep = [i for i, (s, l) in enumerate(zip(p["scores"], p["labels"])) if s >= score_min and l == cid - 1]
        boxes = np.asarray(p["boxes"], float).reshape(-1, 4)[keep]
        xywh = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
        found = match([a["bbox"] for a in anns], xywh, np.asarray(p["scores"], float)[keep], iou)
        for i, a in enumerate(anns):
            box = (a.get("attributes") or {}).get("box_3d")
            if not box:
                continue
            d = float(np.linalg.norm(box["center"]))
            for (lo, hi), row in zip(zip(bins[:-1], bins[1:]), out.values()):
                if lo <= d < hi:
                    row["n"] += 1
                    row["found"] += i in found
                    break
    for row in out.values():
        row["recall"] = row["found"] / row["n"] if row["n"] else None
    return out
