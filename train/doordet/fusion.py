"""Late fusion of detectors (Madelief 2026-11-04: RGB detector + depth detector): weighted box fusion, Solovyev et al. 2021
(arXiv 1910.13302). Own ~40 lines instead of a dependency. Boxes of all models are clustered per label by IoU; a cluster becomes one
box (coordinates averaged with the weighted scores) and its score = weighted mean over all models of each model's best score in it (absent = 0),
so a box that only one model sees is scored down. fuse_runs can also take per-image model weights (adaptive fusion, quality.py). Predictions: {"boxes" (N,4) xyxy, "scores" (N,), "labels" (N,)}."""
import numpy as np


def _ious(f, b):
    """IoU of box b with every row of f (K, 4)."""
    w, h = np.minimum(f[:, 2], b[2]) - np.maximum(f[:, 0], b[0]), np.minimum(f[:, 3], b[3]) - np.maximum(f[:, 1], b[1])
    inter = np.clip(w, 0, None) * np.clip(h, 0, None)
    return inter / ((f[:, 2] - f[:, 0]) * (f[:, 3] - f[:, 1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def empty():
    return {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, int)}


def wbf(preds, weights=None, iou=0.55, skip=0.0):
    """preds: one prediction dict per model -> fused prediction dict. Score = weighted mean over ALL models of each model's best
    score in the cluster (a model that missed it counts 0), i.e. the paper's absent-model-aware average with max per model."""
    w = np.ones(len(preds)) if weights is None else np.asarray(weights, float)
    items = [(int(l), m, float(s), np.asarray(b, float)) for m, p in enumerate(preds)
             for b, s, l in zip(p["boxes"], p["scores"], p["labels"]) if s >= skip]
    out = []
    for label in sorted({i[0] for i in items}):
        boxes = sorted((i for i in items if i[0] == label), key=lambda i: -i[2] * w[i[1]])
        clusters, fused = [], np.zeros((0, 4))  # each cluster: {"m": [(model, score)], "b": [boxes]}; fused[k] = its box
        for _, m, s, b in boxes:
            hit = np.nonzero(_ious(fused, b) > iou)[0]  # all clusters at once: detectors give ~300 boxes per image
            if len(hit):
                k = hit[0]
            else:
                k = len(clusters)
                clusters.append({"m": [], "b": []})
                fused = np.vstack([fused, b])
            c = clusters[k]
            c["m"].append((m, s)); c["b"].append(b)
            ws = np.array([s_ * w[m_] for m_, s_ in c["m"]])
            fused[k] = (np.array(c["b"]) * (ws / max(ws.sum(), 1e-9))[:, None]).sum(0)  # normalise first: a lone box stays exact
        for c, f in zip(clusters, fused):
            c["fused"] = f
        for c in clusters:
            best = {}  # each model's best score in the cluster: DETR-style detectors put many low-score duplicates around
            for m, s in c["m"]:  # an object, and averaging them in dragged the true box's score down (fusion 0.32 vs 0.41 / 0.43)
                best[m] = max(best.get(m, 0.0), s)
            out.append((c["fused"], sum(w[m] * s for m, s in best.items()) / w.sum(), label))
    if not out:
        return empty()
    out.sort(key=lambda o: -o[1])
    return {"boxes": np.array([o[0] for o in out]), "scores": np.array([o[1] for o in out]), "labels": np.array([o[2] for o in out], int)}


def fuse_runs(runs, weights=None, image_weights=None, **kw):
    """runs: one {image_id: prediction dict} per model -> {image_id: fused prediction dict} over the union of image ids.
    image_weights: {image_id: [weight per model]}, multiplied with the model weights; images missing from it use the model weights only."""
    ids = sorted({i for r in runs for i in r})
    w = np.ones(len(runs)) if weights is None else np.asarray(weights, float)
    return {i: wbf([r.get(i, empty()) for r in runs], w * np.asarray(image_weights[i], float) if image_weights and i in image_weights else w, **kw)
            for i in ids}
