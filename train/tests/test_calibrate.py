# Unit tests for detector confidence calibration (2d, Madelief 2026-11-06): per-class temperature scaling (Guo et al. 2017) and
# box-aware logistic calibration (Kuppers et al. 2020: score + box position + size), compared with the detection calibration
# error (D-ECE). Fit on mixed val, tested on every level set; rgb / dha / thermal / fused.
# train/.venv/bin/python -m pytest train/tests/test_calibrate.py
import json, pathlib, sys
import numpy as np
import pytest

TRAIN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))
from doordet import calibrate

W, H = 200, 100


def gt_file(tmp_path, anns, n_images=1):
    coco = {"images": [{"id": i, "file_name": f"{i}.png", "width": W, "height": H} for i in range(n_images)],
            "categories": [{"id": 1, "name": "door"}, {"id": 2, "name": "handle"}],
            "annotations": [{"id": k + 1, "area": a["bbox"][2] * a["bbox"][3], "iscrowd": 0, **a} for k, a in enumerate(anns)]}
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(coco))
    return p


def pred(boxes, scores, labels):
    return {"boxes": np.array(boxes, float).reshape(-1, 4), "scores": np.array(scores, float), "labels": np.array(labels, int)}


# --- detections labelled right / wrong ---

def test_rows_mark_matched_detections_correct(tmp_path):
    gt = gt_file(tmp_path, [{"image_id": 0, "category_id": 1, "bbox": [10, 10, 40, 80]}])
    p = {0: pred([[10, 10, 50, 90], [11, 10, 51, 90], [150, 10, 190, 90], [10, 10, 50, 90]], [0.9, 0.8, 0.7, 0.6], [0, 0, 0, 1])}
    rows = calibrate.rows(gt, p, score_min=0.1)
    by_score = {r["score"]: r for r in rows}
    assert by_score[0.9]["correct"] and not by_score[0.8]["correct"]  # duplicate of a found door: wrong
    assert not by_score[0.7]["correct"]  # nothing there
    assert not by_score[0.6]["correct"] and by_score[0.6]["label"] == 1  # right place, wrong class
    r = by_score[0.9]  # box features normalised by the image size: centre x / y, width, height
    assert (r["cx"], r["cy"], r["w"], r["h"]) == pytest.approx((30 / W, 50 / H, 40 / W, 80 / H))


def test_rows_drop_low_scores_and_detections_on_ignore_boxes(tmp_path):
    gt = gt_file(tmp_path, [{"image_id": 0, "category_id": 1, "bbox": [10, 10, 40, 80], "iscrowd": 1}])
    p = {0: pred([[10, 10, 50, 90], [150, 10, 190, 90], [150, 10, 190, 90]], [0.9, 0.5, 0.05], [0, 0, 0])}
    rows = calibrate.rows(gt, p, score_min=0.1)
    assert [r["score"] for r in rows] == [0.5]  # ignore box: neither right nor wrong; 0.05 < score_min


def test_rows_include_images_without_predictions(tmp_path):
    gt = gt_file(tmp_path, [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 40, 80]}], n_images=2)
    assert calibrate.rows(gt, {}, score_min=0.1) == []


# --- D-ECE ---

def test_dece_zero_when_scores_match_hit_rates():
    rows = [{"label": 0, "score": 0.75, "correct": c} for c in [1, 1, 1, 0]] + [{"label": 0, "score": 0.25, "correct": c} for c in [1, 0, 0, 0]]
    assert calibrate.dece(rows)[0] == pytest.approx(0.0)


def test_dece_per_class_and_overall_weighted():
    rows = [{"label": 0, "score": 0.9, "correct": 0}] * 3 + [{"label": 1, "score": 0.5, "correct": 1}]
    d = calibrate.dece(rows)
    assert d[0] == pytest.approx(0.9) and d[1] == pytest.approx(0.5)
    assert d["all"] == pytest.approx((3 * 0.9 + 0.5) / 4)


def test_dece_by_box_size():
    rows = [{"label": 0, "score": 0.9, "correct": 0, "w": 0.05, "h": 0.05}, {"label": 0, "score": 0.9, "correct": 1, "w": 0.5, "h": 0.5}]
    d = calibrate.dece_by_size(rows, edges=[0, 0.1, 1.0])
    assert d["0-0.1"]["dece"] == pytest.approx(0.9) and d["0.1-1.0"]["dece"] == pytest.approx(0.1)
    assert d["0-0.1"]["n"] == 1


# --- calibrators ---

def overconfident(n=400, seed=0):
    """Rows where the true hit rate is lower than the score (like a detector in smoke), class 0; small boxes are worse."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.3, 0.99, n)
    size = rng.uniform(0.02, 0.6, n)
    p_true = 1 / (1 + np.exp(-(1.5 * np.log(s / (1 - s)) - 1.0 + 4 * (size - 0.3))))
    return [{"label": 0, "score": float(a), "correct": int(rng.random() < b), "cx": 0.5, "cy": 0.5, "w": float(z), "h": float(z)}
            for a, b, z in zip(s, p_true, size)]


@pytest.mark.parametrize("kind", ["temperature", "box"])
def test_calibration_lowers_dece(kind):
    fit, test = overconfident(seed=0), overconfident(seed=1)
    cal = calibrate.fit(fit, kind)
    out = calibrate.apply_rows(cal, test)
    assert calibrate.dece(out)["all"] < 0.6 * calibrate.dece(test)["all"]


def test_temperature_keeps_the_ranking_per_class():
    rows = overconfident()
    cal = calibrate.fit(rows, "temperature")
    out = calibrate.apply_rows(cal, rows)
    order = lambda rs: np.argsort([r["score"] for r in rs], kind="stable").tolist()
    assert order(out) == order(rows)


def test_box_aware_beats_temperature_when_size_matters():
    fit, test = overconfident(2000, seed=0), overconfident(2000, seed=1)
    t = calibrate.dece_by_size(calibrate.apply_rows(calibrate.fit(fit, "temperature"), test), edges=[0, 0.15, 1.0])
    b = calibrate.dece_by_size(calibrate.apply_rows(calibrate.fit(fit, "box"), test), edges=[0, 0.15, 1.0])
    assert b["0-0.15"]["dece"] < t["0-0.15"]["dece"]


def test_class_without_fit_rows_is_left_alone():
    cal = calibrate.fit(overconfident(), "temperature")
    r = {"label": 1, "score": 0.7, "correct": 1, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}
    assert calibrate.apply_rows(cal, [r])[0]["score"] == pytest.approx(0.7)


def test_calibrator_round_trips_through_json():
    for kind in ("temperature", "box"):
        cal = calibrate.fit(overconfident(), kind)
        back = calibrate.from_json(json.loads(json.dumps(calibrate.to_json(cal))))
        rows = overconfident(seed=3)
        assert [r["score"] for r in calibrate.apply_rows(back, rows)] == pytest.approx([r["score"] for r in calibrate.apply_rows(cal, rows)])


def test_unknown_kind():
    with pytest.raises(ValueError):
        calibrate.fit(overconfident(), "isotonic")


# --- whole predictions (for AP before / after) ---

def test_apply_to_predictions_changes_scores_only(tmp_path):
    gt = gt_file(tmp_path, [{"image_id": 0, "category_id": 1, "bbox": [10, 10, 40, 80]}])
    p = {0: pred([[10, 10, 50, 90], [150, 10, 190, 90]], [0.9, 0.5], [0, 0])}
    cal = calibrate.fit(overconfident(), "box")
    out = calibrate.apply_preds(cal, p, gt)
    assert np.array_equal(out[0]["boxes"], p[0]["boxes"]) and np.array_equal(out[0]["labels"], p[0]["labels"])
    assert not np.allclose(out[0]["scores"], p[0]["scores"])
    assert np.array_equal(p[0]["scores"], [0.9, 0.5])  # input untouched


# --- CLI ---

def test_cli_args():
    import calibrate as cli
    a = cli.parse(["runs/rgb", "--fit", "exports/mix", "--eval", "exports/vis1", "exports/vis2", "--out", "x"])
    assert a.runs == ["runs/rgb"] and a.fit == "exports/mix" and a.eval == ["exports/vis1", "exports/vis2"]
    assert a.fit_split == "val" and a.split == "test" and a.score_min == 0.1 and not a.adaptive
    a = cli.parse(["runs/rgb", "runs/dha", "--fit", "e", "--eval", "f", "--adaptive", "--out", "x"])
    assert len(a.runs) == 2 and a.adaptive  # several runs: the fused output is calibrated
    with pytest.raises(SystemExit):
        cli.parse(["runs/rgb", "--fit", "e", "--eval", "f", "--adaptive", "--out", "x"])  # adaptive needs at least two runs
