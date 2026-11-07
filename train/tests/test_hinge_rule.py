# Unit tests for the hinge-side rule (Madelief 2026-11-07, "use the handle position for hinge side"): a door's handle sits on
# the lock edge, so the hinge is on the other side. Rule = the handle box whose centre lies inside the door box (the largest one
# if several); handle centre left of the door centre (image x) -> hinge "right", right of it -> "left"; no handle inside -> None.
# Combined = the classifier's own answer when it says "none" (sliding) or the rule has no handle, else the rule.
# Reported next to the classifier on true boxes (GT handles) and on detector boxes (the detector's own handles, score >= min):
#   acc_hinge_side_rule (over the hinged doors the rule answers), rule_coverage_hinge_side (share it answers), acc_hinge_side_combined.
# train/.venv/bin/python -m pytest train/tests/test_hinge_rule.py
import json, pathlib, sys
import numpy as np
import pytest

TRAIN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))
from doordet import props
from test_props import make_export


def test_rule_handle_left_means_hinge_right():
    door = [100, 50, 80, 200]                              # centre x 140
    assert props.hinge_from_handle(door, [[110, 150, 10, 6]]) == "right"
    assert props.hinge_from_handle(door, [[160, 150, 10, 6]]) == "left"


def test_rule_ignores_handles_outside_and_takes_the_largest_inside():
    door = [100, 50, 80, 200]
    assert props.hinge_from_handle(door, []) is None
    assert props.hinge_from_handle(door, [[10, 150, 10, 6]]) is None          # another door's handle
    assert props.hinge_from_handle(door, [[110, 150, 4, 3], [165, 150, 12, 8]]) == "left"


def test_combine():
    assert props.combine_hinge("none", "left") == "none"     # sliding stays sliding
    assert props.combine_hinge("left", None) == "left"       # no handle: classifier
    assert props.combine_hinge("left", "right") == "right"   # handle seen: rule


def test_rule_scores_on_true_and_detector_boxes(tmp_path):
    root = make_export(tmp_path / "export")
    # fixture: door_a [10, 10, 50, 100] (centre x 35) with handle_a at x 50..56 -> rule "left"; its label is hinge "right"
    # door_b is sliding (label "none"), its handle at x 92..95 inside it
    cfg = props.resolve({"export": str(root), "crop": "door", "backbone": "resnet18", "pretrained": False,
                         "imgsz": 64, "epochs": 1, "batch": 4, "device": "cpu", "workers": 0, "out": str(tmp_path / "runs"),
                         "mlflow": {"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}", "experiment": "test"}})
    r = props.train(cfg, run_name="unit")
    t = props.hinge_rule_true_boxes(root, "test")
    assert t["rule_coverage_hinge_side"] == pytest.approx(1.0)   # every hinged door has its handle inside
    assert t["acc_hinge_side_rule"] == pytest.approx(0.0)        # fixture handle is on the hinge label's side
    preds = {}
    for i in range(4):  # door_a + its handle found; in image 0 the handle is missed
        boxes, scores, labels = [[10, 10, 60, 110]], [0.9], [0]
        if i > 0:
            boxes.append([50, 60, 56, 63]); scores.append(0.8); labels.append(1)
        preds[i] = {"boxes": np.array(boxes, float), "scores": np.array(scores), "labels": np.array(labels)}
    s = props.score_on_boxes(r["out"], root, "test", preds, score_min=0.3, iou=0.5)
    assert s["rule_coverage_hinge_side"] == pytest.approx(3 / 4)
    assert s["acc_hinge_side_rule"] == pytest.approx(0.0)
    assert 0.0 <= s["acc_hinge_side_combined"] <= 1.0
    assert "acc_hinge_side" in s                                  # the classifier's own number stays
