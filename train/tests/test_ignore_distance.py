# Unit tests for ignore boxes in training / scoring and detector recall per distance (no GPU, no downloads):
# train/.venv/bin/python -m pytest train/tests
import json, pathlib, sys
import numpy as np
import pytest
from PIL import Image

TRAIN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))
from doordet import evaluate
from doordet.data import CocoDetection

W, H = 200, 100


def gt(tmp_path, anns):
    """COCO json with one image; anns = [(category, bbox, iscrowd, distance_m or None)]."""
    root = tmp_path / "exp"
    (root / "images" / "train").mkdir(parents=True)
    Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(root / "images" / "train" / "000000.png")
    coco = {"images": [{"id": 0, "file_name": "images/train/000000.png", "width": W, "height": H}],
            "categories": [{"id": 1, "name": "door"}, {"id": 2, "name": "handle"}], "annotations": []}
    for i, (cat, bbox, crowd, dist) in enumerate(anns):
        attrs = {} if dist is None else {"box_3d": {"center": [0.0, 0.0, float(dist)], "orientation_xyzw": [0, 0, 0, 1], "size": [0.1, 0.1, 0.1]}}
        coco["annotations"].append({"id": i + 1, "image_id": 0, "category_id": cat, "bbox": bbox, "area": bbox[2] * bbox[3],
                                    "iscrowd": crowd, "attributes": attrs})
    (root / "coco").mkdir()
    for sp in ("train", "val", "test"):
        (root / "coco" / f"{sp}.json").write_text(json.dumps(coco))
    return root


def pred(*boxes):
    """boxes: (x, y, w, h, score, label0)"""
    return {0: {"boxes": np.array([[x, y, x + w, y + h] for x, y, w, h, *_ in boxes], float).reshape(-1, 4),
                "scores": np.array([b[4] for b in boxes], float), "labels": np.array([b[5] for b in boxes], int)}}


# --- ignore boxes ---
def test_training_targets_leave_out_ignore_boxes(tmp_path):
    root = gt(tmp_path, [(1, [10, 10, 40, 80], 0, None), (1, [100, 10, 40, 80], 1, None)])
    ds = CocoDetection(root, "train")
    _, t = ds[0]
    assert len(t["annotations"]) == 1 and t["annotations"][0]["bbox"] == [10, 10, 40, 80]


def test_scoring_does_not_punish_a_hit_on_an_ignore_box(tmp_path):
    root = gt(tmp_path, [(1, [10, 10, 40, 80], 0, None), (1, [100, 10, 40, 80], 1, None)])
    both = pred((10, 10, 40, 80, 0.9, 0), (100, 10, 40, 80, 0.95, 0))
    assert evaluate.coco_metrics(root / "coco" / "test.json", both)["map"] == pytest.approx(1.0)


# --- recall per distance ---
def test_recall_per_distance_bins(tmp_path):
    root = gt(tmp_path, [(2, [10, 10, 10, 10], 0, 0.8), (2, [50, 10, 10, 10], 0, 2.5), (2, [90, 10, 10, 10], 0, 2.7),
                         (1, [120, 10, 40, 80], 0, 1.5)])
    p = pred((10, 10, 10, 10, 0.9, 1), (50, 10, 10, 10, 0.2, 1), (120, 10, 40, 80, 0.8, 0))
    r = evaluate.recall_by_distance(root / "coco" / "test.json", p, "handle", bins=(0, 1, 2, 3, 4), score_min=0.3, iou=0.5)
    assert r["0-1"] == {"n": 1, "found": 1, "recall": 1.0}
    assert r["2-3"] == {"n": 2, "found": 0, "recall": 0.0}  # the 2.5 m hit is below score_min
    assert r["1-2"] == {"n": 0, "found": 0, "recall": None} and r["3-4"]["n"] == 0
    rd = evaluate.recall_by_distance(root / "coco" / "test.json", p, "door", bins=(0, 1, 2, 3, 4), score_min=0.3, iou=0.5)
    assert rd["1-2"] == {"n": 1, "found": 1, "recall": 1.0}


def test_recall_per_distance_skips_ignore_and_wrong_class(tmp_path):
    root = gt(tmp_path, [(2, [10, 10, 10, 10], 1, 0.8), (2, [50, 10, 10, 10], 0, 0.9)])
    p = pred((50, 10, 10, 10, 0.9, 0))  # right place, wrong class (door)
    r = evaluate.recall_by_distance(root / "coco" / "test.json", p, "handle", bins=(0, 1), score_min=0.3, iou=0.5)
    assert r["0-1"] == {"n": 1, "found": 0, "recall": 0.0}


def test_match_greedy_by_score():
    gt_boxes = [[0, 0, 10, 10], [20, 0, 10, 10]]
    pr = [[1, 0, 10, 10], [0, 0, 10, 10], [50, 50, 5, 5]]
    m = evaluate.match(gt_boxes, pr, [0.5, 0.9, 0.99], iou=0.5)
    assert m == {0: 1}  # best-scoring overlapping prediction wins; the far one matches nothing
