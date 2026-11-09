# Point score for thin / tiny parts (Madelief 2026-11-09, "yes, add it"): hinge boxes are ~2-3 px wide, so mAP (IoU 0.5-0.95)
# punishes a 1 px offset as a miss (1 px off on a 3 px box = IoU ~0.5). For a robot it matters whether the detector points at
# the hinge, not whether the box edges are pixel-exact. So, next to mAP, hinges and handles also get a centre-distance score:
#   evaluate.point_metrics(gt_json, preds, classes=POINT_CLASSES, score_min=0.5) -> {"point_ap_<name>", "point_recall_<name>"}
#   for every name in `classes` that is a category of the export (others skipped). POINT_CLASSES = ("hinge", "handle").
#   - match: a prediction of that class hits a GT box (not iscrowd) of that class when the distance between the two box centres
#     is <= radius = max(MIN_RADIUS, 0.5 * max(gt w, gt h)) px; MIN_RADIUS = 4. Greedy per image by descending score: each
#     prediction takes the nearest free GT within its radius (one GT per prediction, one prediction per GT).
#   - ignore: a prediction that hits no GT but whose centre lies inside an iscrowd box of the same class is left out (neither
#     hit nor false alarm), like COCO's ignore regions.
#   - point_ap: area under precision / recall over all predictions of that class, sorted by score over the whole split, COCO
#     style (precision made non-increasing, sampled at 101 recall points 0, 0.01 .. 1; recall points beyond reach count 0).
#   - point_recall: share of GT boxes hit by a prediction with score >= score_min.
#   - no GT of a class -> both values 0.0. Predictions use 0-based labels (category_id - 1) and xyxy boxes, like coco_metrics.
#   coco_metrics also returns these keys (so training logs val_point_ap_hinge etc. and score.py reports them); all 0.0 when there
#   are no predictions.
# train/.venv/bin/python -m pytest train/tests/test_point_metrics.py
import json, pathlib, sys
import numpy as np
import pytest

TRAIN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))
from doordet import evaluate

CATS = [{"id": 1, "name": "door"}, {"id": 2, "name": "handle"}, {"id": 3, "name": "hinge"}]
HINGE = 2  # 0-based label of category 3


def gt_file(tmp_path, anns, n_images=2, cats=CATS):
    images = [{"id": i, "width": 640, "height": 480, "file_name": f"{i}.png"} for i in range(n_images)]
    out = []
    for k, (img, cat, box, *crowd) in enumerate(anns):
        out.append({"id": k + 1, "image_id": img, "category_id": cat, "bbox": box, "area": box[2] * box[3],
                    "iscrowd": int(bool(crowd and crowd[0]))})
    p = tmp_path / "gt.json"
    p.write_text(json.dumps({"images": images, "annotations": out, "categories": cats}))
    return p


def xyxy(cx, cy, w=3, h=14):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def pred(*items):
    """items: (label, xyxy box, score)"""
    return {"boxes": [b for _, b, _ in items], "scores": [s for *_, s in items], "labels": [l for l, *_ in items]}


HINGE_A = [99, 43, 2, 14]   # centre (100, 50), radius 7
HINGE_B = [299, 193, 2, 14]  # centre (300, 200)


def test_offset_hinge_is_a_point_hit_but_not_an_iou_hit(tmp_path):
    gt = gt_file(tmp_path, [(0, 3, HINGE_A)], n_images=1)
    preds = {0: pred((HINGE, xyxy(101.5, 50, w=2), 0.9))}  # 1.5 px off: IoU 0.2, but clearly the hinge
    m = evaluate.point_metrics(gt, preds)
    assert m["point_ap_hinge"] == pytest.approx(1.0) and m["point_recall_hinge"] == 1.0
    assert evaluate.coco_metrics(gt, preds)["ap_hinge"] < 0.1


def test_radius_scales_with_the_box_and_has_a_floor(tmp_path):
    gt = gt_file(tmp_path, [(0, 3, HINGE_A), (1, 3, [50, 50, 2, 2])])  # 2x2 box: radius MIN_RADIUS = 4
    assert evaluate.MIN_RADIUS == 4
    near = {0: pred((HINGE, xyxy(100, 56.9), 0.9)), 1: pred((HINGE, xyxy(51, 54.9, 2, 2), 0.9))}  # 6.9 and 3.9 px off
    far = {0: pred((HINGE, xyxy(100, 57.1), 0.9)), 1: pred((HINGE, xyxy(51, 55.1, 2, 2), 0.9))}   # 7.1 and 4.1 px off
    assert evaluate.point_metrics(gt, near)["point_recall_hinge"] == 1.0
    assert evaluate.point_metrics(gt, far)["point_recall_hinge"] == 0.0


def test_one_prediction_per_hinge_duplicates_are_false_alarms(tmp_path):
    gt = gt_file(tmp_path, [(0, 3, HINGE_A)], n_images=1)
    preds = {0: pred((HINGE, xyxy(100, 50), 0.9), (HINGE, xyxy(100, 51), 0.8))}
    m = evaluate.point_metrics(gt, preds)
    assert m["point_recall_hinge"] == 1.0 and m["point_ap_hinge"] == pytest.approx(1.0)  # dup ranks after the hit
    preds = {0: pred((HINGE, xyxy(100, 51), 0.95), (HINGE, xyxy(100, 50), 0.9), (HINGE, xyxy(20, 20), 0.99))}
    m = evaluate.point_metrics(gt, preds)  # false alarm first, then the hit: precision at full recall = 1/2
    assert m["point_ap_hinge"] == pytest.approx(0.5, abs=0.01)


def test_greedy_by_score_takes_the_nearest_free_hinge(tmp_path):
    two = [(0, 3, [99, 43, 2, 14]), (0, 3, [105, 43, 2, 14])]  # centres x 100 and 106
    gt = gt_file(tmp_path, two, n_images=1)
    # 0.9 at x 104 must take the nearer x 106 (2 px), not x 100 (4 px): else the 0.8 at x 98 can't reach x 106 (8 px > 7)
    preds = {0: pred((HINGE, xyxy(104, 50), 0.9), (HINGE, xyxy(98, 50), 0.8))}
    m = evaluate.point_metrics(gt, preds)
    assert m["point_recall_hinge"] == 1.0 and m["point_ap_hinge"] == pytest.approx(1.0)


def test_ap_ranks_over_the_whole_split_and_recall_uses_score_min(tmp_path):
    gt = gt_file(tmp_path, [(0, 3, HINGE_A), (1, 3, HINGE_B)])
    preds = {0: pred((HINGE, xyxy(100, 50), 0.9)), 1: pred((HINGE, xyxy(300, 200), 0.3), (HINGE, xyxy(500, 400), 0.6))}
    m = evaluate.point_metrics(gt, preds, score_min=0.5)
    assert m["point_recall_hinge"] == 0.5  # the 0.3 hit is below score_min
    # ranked: hit 0.9 (P 1, R .5), false 0.6 (P .5, R .5), hit 0.3 (P 2/3, R 1) -> interpolated P: 1 up to R .5, 2/3 after
    assert m["point_ap_hinge"] == pytest.approx((51 * 1 + 50 * 2 / 3) / 101, abs=1e-6)


def test_wrong_class_and_crowd(tmp_path):
    gt = gt_file(tmp_path, [(0, 3, HINGE_A), (0, 3, [400, 100, 10, 40], True)], n_images=1)
    preds = {0: pred((1, xyxy(100, 50), 0.99),              # a handle on the hinge: not a hinge hit
                     (HINGE, xyxy(405, 120), 0.95),          # inside the crowd box: ignored, not a false alarm
                     (HINGE, xyxy(100, 50), 0.9))}
    m = evaluate.point_metrics(gt, preds)
    assert m["point_recall_hinge"] == 1.0 and m["point_ap_hinge"] == pytest.approx(1.0)


def test_classes_missing_or_without_gt(tmp_path):
    cats = [{"id": 1, "name": "door"}, {"id": 2, "name": "handle"}]
    gt = gt_file(tmp_path, [(0, 1, [10, 10, 50, 100])], n_images=1, cats=cats)
    m = evaluate.point_metrics(gt, {0: pred((0, [10, 10, 60, 110], 0.9))})
    assert m == {"point_ap_handle": 0.0, "point_recall_handle": 0.0}  # no hinge category, handle without GT
    assert evaluate.POINT_CLASSES == ("hinge", "handle")


def test_string_image_ids_and_missing_images(tmp_path):
    gt = gt_file(tmp_path, [(0, 3, HINGE_A), (1, 3, HINGE_B)])
    m = evaluate.point_metrics(gt, {"0": pred((HINGE, xyxy(100, 50), 0.9))})  # image 1 has no predictions
    assert m["point_recall_hinge"] == 0.5


def test_coco_metrics_includes_point_keys(tmp_path):
    gt = gt_file(tmp_path, [(0, 3, HINGE_A), (0, 2, [200, 200, 15, 11])], n_images=1)
    m = evaluate.coco_metrics(gt, {0: pred((HINGE, xyxy(101.5, 50, w=2), 0.9))})
    assert m["point_ap_hinge"] == pytest.approx(1.0) and m["point_recall_handle"] == 0.0 and "map" in m
    empty = evaluate.coco_metrics(gt, {})
    assert empty["point_ap_hinge"] == 0.0 and empty["point_recall_handle"] == 0.0
