"""Predictions -> COCO results -> mAP (overall, by size, per class) with pycocotools."""
import contextlib, io, json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

STATS = ("map", "map_50", "map_75", "map_small", "map_medium", "map_large")
POINT_CLASSES = ("hinge", "handle")
MIN_RADIUS = 4  # px


def to_coco_results(preds):
    out = []
    for image_id, p in preds.items():
        for (x1, y1, x2, y2), s, l in zip(np.asarray(p["boxes"], float).reshape(-1, 4), p["scores"], p["labels"]):
            out.append({"image_id": int(image_id), "category_id": int(l) + 1,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})
    return out


def point_metrics(gt_json, preds, classes=POINT_CLASSES, score_min=0.5):
    """Centre-distance score per class: {"point_ap_<name>", "point_recall_<name>"}; a prediction hits a GT whose centre is within max(MIN_RADIUS, half its longer side) px."""
    gt = json.loads(open(gt_json).read())
    ids = {c["name"]: c["id"] for c in gt["categories"]}
    out = {}
    for name in (n for n in classes if n in ids):
        cid, n_gt, kept, hits = ids[name], 0, [], 0
        by_img = {}
        for a in gt["annotations"]:
            if a["category_id"] == cid:
                by_img.setdefault(a["image_id"], []).append(a)
        for img in gt["images"]:
            anns = by_img.get(img["id"], [])
            g = np.array([a["bbox"] for a in anns if not a.get("iscrowd", 0)], float).reshape(-1, 4)
            crowd = np.array([a["bbox"] for a in anns if a.get("iscrowd", 0)], float).reshape(-1, 4)
            n_gt += len(g)
            gc, rad = g[:, :2] + g[:, 2:] / 2, np.maximum(MIN_RADIUS, 0.5 * g[:, 2:].max(1, initial=0))
            p = preds.get(img["id"]) or preds.get(str(img["id"])) or {"boxes": [], "scores": [], "labels": []}
            boxes, scores = np.asarray(p["boxes"], float).reshape(-1, 4), np.asarray(p["scores"], float)
            idx = [i for i, l in enumerate(p["labels"]) if l == cid - 1]
            free = np.ones(len(g), bool)
            for i in sorted(idx, key=lambda i: -scores[i]):
                c = (boxes[i, :2] + boxes[i, 2:]) / 2
                d = np.where(free, np.linalg.norm(gc - c, axis=1), np.inf)
                d[d > rad] = np.inf
                if len(d) and np.isfinite(d.min()):
                    free[d.argmin()] = False
                    kept.append((scores[i], 1))
                    hits += scores[i] >= score_min
                elif not ((c >= crowd[:, :2]) & (c <= crowd[:, :2] + crowd[:, 2:])).all(1).any():
                    kept.append((scores[i], 0))
        ap = 0.0
        if n_gt and kept:
            tp = np.array([t for _, t in sorted(kept, key=lambda k: -k[0])])
            tpc, fpc = np.cumsum(tp), np.cumsum(1 - tp)
            prec, rec = np.maximum.accumulate((tpc / (tpc + fpc))[::-1])[::-1], tpc / n_gt
            at = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
            ap = float(np.mean([prec[i] if i < len(prec) else 0.0 for i in at]))
        out[f"point_ap_{name}"], out[f"point_recall_{name}"] = ap, float(hits / n_gt) if n_gt else 0.0
    return out


def door_hinge_metrics(gt_json, preds, score_min=0.5, margin=0.1):
    """Per-door "is there a hinge" score: door_hinge_recall / _false_yes / _n_pos / _n_neg / _recall_<state>; spec in tests/test_door_hinge.py."""
    gt = json.loads(open(gt_json).read())
    ids = {c["name"]: c["id"] for c in gt["categories"]}
    n_pos, n_neg, found, false_yes, by_state = 0, 0, 0, 0, {}
    for img in gt["images"]:
        anns = [a for a in gt["annotations"] if a["image_id"] == img["id"]]
        hinges = [a for a in anns if a["category_id"] == ids.get("hinge")]
        live = [a for a in hinges if not a.get("iscrowd", 0)]
        g = np.array([a["bbox"] for a in live], float).reshape(-1, 4)
        gc, rad = g[:, :2] + g[:, 2:] / 2, np.maximum(MIN_RADIUS, 0.5 * g[:, 2:].max(1, initial=0))
        p = preds.get(img["id"]) or preds.get(str(img["id"])) or {"boxes": [], "scores": [], "labels": []}
        boxes, scores = np.asarray(p["boxes"], float).reshape(-1, 4), np.asarray(p["scores"], float)
        keep = [i for i, l in enumerate(p["labels"]) if l == ids.get("hinge", 0) - 1 and scores[i] >= score_min]
        pc = (boxes[keep, :2] + boxes[keep, 2:]) / 2
        hit = (np.linalg.norm(pc[:, None] - gc[None], axis=2) <= rad).reshape(len(pc), len(gc))  # pred x GT hinge
        for d in (a for a in anns if a["category_id"] == ids.get("door")):
            mine = [k for k, a in enumerate(live) if a["attributes"].get("door") == d["object"]]
            if mine:
                n_pos += 1
                ok = bool(hit[:, mine].any())
                found += ok
                st = by_state.setdefault(d["attributes"].get("state"), [0, 0])
                st[0] += ok; st[1] += 1
            elif not any(a["attributes"].get("door") == d["object"] for a in hinges):
                n_neg += 1
                x, y, w, h = d["bbox"]
                inside = (pc[:, 0] >= x - margin * w) & (pc[:, 0] <= x + w + margin * w) & (pc[:, 1] >= y) & (pc[:, 1] <= y + h)
                false_yes += bool((inside & ~hit.any(1)).any())
    out = {"door_hinge_recall": found / n_pos if n_pos else 0.0, "door_hinge_false_yes": false_yes / n_neg if n_neg else 0.0,
           "door_hinge_n_pos": n_pos, "door_hinge_n_neg": n_neg}
    out.update({f"door_hinge_recall_{s}": v[0] / v[1] for s, v in by_state.items()})
    return out


def coco_metrics(gt_json, preds):
    gt = json.loads(open(gt_json).read())
    for a in gt["annotations"]:
        a["id"] += 1  # pycocotools treats id 0 as 'unmatched'; exports made before 2026-10-29 number annotations from 0
    names = {c["id"]: c["name"] for c in gt["categories"]}
    metrics = {k: 0.0 for k in (*STATS, *(f"ap_{n}" for n in names.values()))}
    metrics.update(point_metrics(gt_json, preds))
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
