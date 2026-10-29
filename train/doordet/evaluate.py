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
