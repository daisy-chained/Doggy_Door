"""Detector confidence calibration, Madelief 2026-11-06 (2d): make a score of 0.8 mean ~80 % of those boxes are right.
Per-class temperature scaling (Guo et al. 2017: p = sigmoid(logit(s) / T), rank-preserving) vs box-aware logistic calibration (Kuppers et al. 2020: score + box position + size),
compared with the detection calibration error (D-ECE). Rows = one per prediction with a right / wrong label."""
import json
import numpy as np
from scipy.optimize import minimize

from . import evaluate

MIN_ROWS = 10
FEATS = ("cx", "cy", "w", "h")


def rows(gt_json, preds, score_min=0.1, iou=0.5):
    """-> [{image_id, label, score, correct, cx, cy, w, h}] for every prediction >= score_min; box features / image size.
    Correct = greedy match (by score) to a free non-ignore GT of the same class; wrong detections on an ignore GT are dropped."""
    gt = json.loads(open(gt_json).read())
    size = {im["id"]: (im["width"], im["height"]) for im in gt["images"]}
    anns = {}
    for a in gt["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    out = []
    for image_id, (iw, ih) in size.items():
        p = preds.get(image_id) or preds.get(str(image_id))
        if p is None:
            continue
        boxes = np.asarray(p["boxes"], float).reshape(-1, 4)
        scores, labels = np.asarray(p["scores"], float), np.asarray(p["labels"], int)
        for label in sorted({int(l) for l in labels}):
            keep = np.where((labels == label) & (scores >= score_min))[0]
            if not len(keep):
                continue
            mine = [a for a in anns.get(image_id, []) if a["category_id"] - 1 == label]
            real = [a["bbox"] for a in mine if not a.get("iscrowd", 0)]
            ign = [a["bbox"] for a in mine if a.get("iscrowd", 0)]
            xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes[keep]]
            hit = set(evaluate.match(real, xywh, scores[keep], iou).values())
            for j, k in enumerate(keep):
                if j not in hit and ign and evaluate.iou_xywh(xywh[j], ign).max() >= iou:
                    continue
                x, y, w, h = xywh[j]
                out.append({"image_id": image_id, "label": label, "score": float(scores[k]), "correct": int(j in hit),
                            "cx": (x + w / 2) / iw, "cy": (y + h / 2) / ih, "w": w / iw, "h": h / ih})
    return out


def _ece(rs, bins):
    if not rs:
        return None
    s, c = np.array([r["score"] for r in rs], float), np.array([r["correct"] for r in rs], float)
    b = np.minimum((s * bins).astype(int), bins - 1)
    return float(sum(abs(s[b == k].mean() - c[b == k].mean()) * (b == k).sum() for k in range(bins) if (b == k).any()) / len(rs))


def dece(rows, bins=10):
    """{label: D-ECE of that class, "all": D-ECE of all rows pooled}; equal-width score bins, |mean score - hit rate| weighted by count."""
    out = {l: _ece([r for r in rows if r["label"] == l], bins) for l in sorted({r["label"] for r in rows})}
    out["all"] = _ece(rows, bins)
    return out


def dece_by_size(rows, edges, bins=10):
    """{"lo-hi": {"n", "dece"}} by box size sqrt(w*h) (image-normalised); dece None for an empty bin."""
    out, last = {}, len(edges) - 2
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = [r for r in rows if lo <= np.sqrt(r["w"] * r["h"]) < hi or (k == last and np.sqrt(r["w"] * r["h"]) == hi)]
        out[f"{lo}-{hi}"] = {"n": len(sel), "dece": _ece(sel, bins)}
    return out


def _logit(s):
    s = np.clip(np.asarray(s, float), 1e-6, 1 - 1e-6)
    return np.log(s / (1 - s))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _nll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _x(rs):
    return np.column_stack([_logit([r["score"] for r in rs])] + [[r[f] for r in rs] for f in FEATS])


def _fit_class(rs, kind):
    y = np.array([r["correct"] for r in rs], float)
    if kind == "temperature":
        z = _x(rs)[:, 0]
        res = minimize(lambda v: _nll(_sigmoid(z / np.exp(v[0])), y), np.zeros(1), bounds=[(-4, 4)], method="L-BFGS-B")
        return {"T": float(np.exp(res.x[0]))}
    x = _x(rs)
    f = lambda v: _nll(_sigmoid(x @ v[:5] + v[5]), y) + 1e-3 * float(v[:5] @ v[:5])
    res = minimize(f, np.array([1.0, 0, 0, 0, 0, 0]), method="L-BFGS-B")
    return {"a": [float(v) for v in res.x[:5]], "b": float(res.x[5])}


def fit(rows, kind):
    """kind "temperature" or "box" -> {"kind", "classes": {label: params}}; classes with < 10 rows or one outcome stay uncalibrated."""
    if kind not in ("temperature", "box"):
        raise ValueError(f"unknown calibration kind {kind!r}")
    classes = {}
    for l in sorted({r["label"] for r in rows}):
        rs = [r for r in rows if r["label"] == l]
        if len(rs) >= MIN_ROWS and 0 < sum(r["correct"] for r in rs) < len(rs):
            classes[l] = _fit_class(rs, kind)
    return {"kind": kind, "classes": classes}


def _calibrate(par, kind, x):
    if kind == "temperature":
        return _sigmoid(x[:, 0] / par["T"])
    return _sigmoid(x @ np.array(par["a"]) + par["b"])


def apply_rows(cal, rows):
    """New rows with calibrated scores; classes not in cal unchanged."""
    out = [dict(r) for r in rows]
    for l, par in cal["classes"].items():
        idx = [i for i, r in enumerate(out) if r["label"] == l]
        if idx:
            for i, s in zip(idx, _calibrate(par, cal["kind"], _x([out[i] for i in idx]))):
                out[i]["score"] = float(s)
    return out


def apply_preds(cal, preds, gt_json):
    """New predictions {image_id: ...} with calibrated scores; boxes / labels copied."""
    gt = json.loads(open(gt_json).read())
    size = {im["id"]: (im["width"], im["height"]) for im in gt["images"]}
    out = {}
    for image_id, p in preds.items():
        boxes = np.array(p["boxes"], float).reshape(-1, 4)
        scores, labels = np.array(p["scores"], float), np.array(p["labels"], int)
        iw, ih = size.get(image_id) or size[int(image_id)]
        new = scores.copy()
        for l, par in cal["classes"].items():
            m = labels == l
            if m.any():
                b = boxes[m]
                x = np.column_stack([_logit(scores[m]), (b[:, 0] + b[:, 2]) / 2 / iw, (b[:, 1] + b[:, 3]) / 2 / ih,
                                     (b[:, 2] - b[:, 0]) / iw, (b[:, 3] - b[:, 1]) / ih])
                new[m] = _calibrate(par, cal["kind"], x)
        out[image_id] = {"boxes": boxes, "scores": new, "labels": labels}
    return out


def to_json(cal):
    return {"kind": cal["kind"], "classes": {str(l): par for l, par in cal["classes"].items()}}


def from_json(d):
    return {"kind": d["kind"], "classes": {int(l): par for l, par in d["classes"].items()}}
