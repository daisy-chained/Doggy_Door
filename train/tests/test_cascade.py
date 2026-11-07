# Unit tests for the door property cascade, step A (Madelief 2026-11-07, "A then B"): first decide the door type, then ask the
# hinged-only questions. The flat classifier answered opening / hinge side in one go and called 61 % of sliding doors push or pull.
#   1. door type: from the handle when one is seen inside the door box (flush pull -> sliding, lever / knob -> hinged; the handle
#      type comes from the handle classifier), else from a type classifier (a door_type head, or the opening head's P(slide)).
#   2. sliding -> opening "slide", hinge side "none" (no hinge question).
#   3. hinged -> push / pull from the opening head restricted to push vs pull (B will replace this with "hinges visible?");
#      left / right from the handle rule (hinge opposite the handle), else the hinge_side head restricted to left vs right.
# Scored on true boxes (GT handles) and on detector boxes (the detector's handles), next to the flat classifier's numbers.
# train/.venv/bin/python -m pytest train/tests/test_cascade.py
import json, pathlib, sys
import numpy as np
import pytest

TRAIN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))
from doordet import cascade, props
from test_props import make_export


# --- the door_type head ---
def test_door_type_head_is_opt_in():
    assert props.HEADS["door"]["door_type"] == ["hinged", "sliding"]
    assert props.labels({"opening": "slide", "door_type": "sliding"}, "door")["door_type"] == "sliding"
    assert props.labels({"opening": "pull", "hinge_side": "left", "door_type": "hinged"}, "door")["door_type"] == "hinged"
    base = {"export": "x", "crop": "door"}
    assert props.resolve(base)["heads"] == ["opening", "hinge_side", "glass"]       # default heads unchanged (old configs, old runs)
    assert props.resolve({**base, "heads": ["door_type"]})["heads"] == ["door_type"]


# --- pure decisions ---
def test_type_from_handle():
    assert cascade.type_from_handle("flush_pull") == "sliding"
    assert cascade.type_from_handle("lever") == "hinged" and cascade.type_from_handle("knob") == "hinged"
    assert cascade.type_from_handle(None) is None


def test_largest_handle_inside_matches_the_hinge_rule():
    door = [100, 50, 80, 200]
    hs = [[10, 150, 10, 6], [110, 150, 4, 3], [165, 150, 12, 8]]
    assert cascade.largest_handle_inside(door, hs) == 2
    assert cascade.largest_handle_inside(door, hs[:1]) is None
    assert cascade.largest_handle_inside(door, []) is None
    assert props.hinge_from_handle(door, hs) == "left"   # same handle choice as the rule


def test_decide_sliding_skips_the_hinge_question():
    d = cascade.decide(p_sliding=0.9, handle_type=None, p_push_pull=(0.7, 0.3), rule_hinge="left", p_left_right=(0.8, 0.2))
    assert d == {"door_type": "sliding", "type_source": "classifier", "opening": "slide", "hinge_side": "none", "hinge_source": None}


def test_decide_handle_overrules_the_type_classifier():
    d = cascade.decide(p_sliding=0.9, handle_type="lever", p_push_pull=(0.3, 0.7), rule_hinge="right", p_left_right=(0.8, 0.2))
    assert d["door_type"] == "hinged" and d["type_source"] == "handle"
    assert d["opening"] == "pull"                                          # push vs pull only
    assert d["hinge_side"] == "right" and d["hinge_source"] == "rule"      # rule before the classifier
    d = cascade.decide(p_sliding=0.1, handle_type="flush_pull", p_push_pull=(0.9, 0.1), rule_hinge="left", p_left_right=(0.9, 0.1))
    assert (d["door_type"], d["type_source"], d["opening"], d["hinge_side"]) == ("sliding", "handle", "slide", "none")


def test_decide_hinged_without_handle_uses_the_classifiers():
    d = cascade.decide(p_sliding=0.2, handle_type=None, p_push_pull=(0.6, 0.4), rule_hinge=None, p_left_right=(0.3, 0.7))
    assert d == {"door_type": "hinged", "type_source": "classifier", "opening": "push", "hinge_side": "right", "hinge_source": "classifier"}
    assert cascade.decide(0.5, None, (0.5, 0.5), None, (0.5, 0.5))["door_type"] == "sliding"   # ties: p_sliding >= 0.5 -> sliding


def test_scores_from_decisions():
    truth = [{"opening": "slide", "hinge_side": "none"}, {"opening": "push", "hinge_side": "left"},
             {"opening": "pull", "hinge_side": "right"}, {"opening": "slide", "hinge_side": "none"}]
    dec = [{"door_type": "sliding", "type_source": "handle", "opening": "slide", "hinge_side": "none", "hinge_source": None},
           {"door_type": "hinged", "type_source": "handle", "opening": "push", "hinge_side": "right", "hinge_source": "rule"},
           {"door_type": "hinged", "type_source": "classifier", "opening": "pull", "hinge_side": "right", "hinge_source": "classifier"},
           {"door_type": "hinged", "type_source": "classifier", "opening": "push", "hinge_side": "left", "hinge_source": "classifier"}]
    s = cascade.scores(dec, truth)
    assert s["acc_door_type"] == pytest.approx(3 / 4)
    assert s["acc_opening"] == pytest.approx(3 / 4)
    assert s["acc_hinge_side"] == pytest.approx(2 / 4)
    assert s["type_from_handle"] == pytest.approx(2 / 4)            # share of doors typed by their handle
    assert s["acc_door_type_handle"] == pytest.approx(1.0) and s["acc_door_type_classifier"] == pytest.approx(1 / 2)
    assert s["acc_push_pull_hinged"] == pytest.approx(1.0)          # over true hinged doors the cascade called hinged
    assert s["acc_left_right_hinged"] == pytest.approx(1 / 2)
    assert s["doors"] == 4
    assert cascade.scores([], [])["doors"] == 0


# --- end to end on the tiny export ---
def _runs(tmp_path, root):
    def run(crop, name, **kw):
        extra = {"min_px": 2} if crop == "handle" else {}  # fixture handles are 3-6 px
        cfg = props.resolve({"export": str(root), "crop": crop, "backbone": "resnet18", "pretrained": False, "imgsz": 64, "epochs": 1, **extra,
                             "batch": 4, "device": "cpu", "workers": 0, "out": str(tmp_path / "runs"),
                             "mlflow": {"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}", "experiment": "test"}, **kw})
        return props.train(cfg, run_name=name)["out"]
    return run("door", "door"), run("handle", "handle"), run("door", "type", heads=["door_type"])


def test_score_on_true_and_detector_boxes(tmp_path):
    root = make_export(tmp_path / "export")
    door_run, handle_run, type_run = _runs(tmp_path, root)
    # fixture: door_a hinged (push, right, lever handle inside), door_b sliding (flush pull handle inside), 4 test images
    t = cascade.score(door_run, handle_run, root, "test")
    assert t["doors"] == 8 and t["type_from_handle"] == pytest.approx(1.0)      # every door has its handle inside
    for k in ("acc_door_type", "acc_opening", "acc_hinge_side", "acc_door_type_handle", "flat_acc_opening", "flat_acc_hinge_side"):
        assert 0.0 <= t[k] <= 1.0
    t2 = cascade.score(door_run, handle_run, root, "test", type_run=type_run)
    assert t2["doors"] == 8
    # detector boxes: both doors found in every image, door_b's handle missed in image 0 -> 7 of 8 typed by handle
    preds = {}
    for i in range(4):
        boxes, scores, labels = [[10, 10, 60, 110], [90, 10, 140, 110], [50, 60, 56, 63]], [0.9, 0.9, 0.8], [0, 0, 1]
        if i > 0:
            boxes.append([92, 60, 95, 68]); scores.append(0.8); labels.append(1)
        preds[i] = {"boxes": np.array(boxes, float), "scores": np.array(scores), "labels": np.array(labels)}
    d = cascade.score(door_run, handle_run, root, "test", preds=preds, score_min=0.3)
    assert d["doors"] == 8 and d["matched"] == 8
    assert d["type_from_handle"] == pytest.approx(7 / 8)


def test_cli_writes_json(tmp_path, monkeypatch):
    root = make_export(tmp_path / "export")
    door_run, handle_run, _ = _runs(tmp_path, root)
    import score_cascade
    monkeypatch.setattr(score_cascade.score, "predict_run", lambda *a, **k: (None, {}, None))
    out = tmp_path / "c.json"
    monkeypatch.setattr(sys, "argv", ["score_cascade.py", str(door_run), str(handle_run), "det", str(root), "--out", str(out)])
    score_cascade.main()
    d = json.loads(out.read_text())
    assert {"true_boxes", "detector_boxes", "door_run", "handle_run", "type_run"} <= set(d)
    assert d["true_boxes"]["doors"] == 8


# --- step B: hinges as their own detector class (Madelief 2026-11-07, "A then B") ---
# Hinge boxes (class 3) are visible-only, so a hinge box at the door means the camera is on the pull side. Per hinged door:
#   hinges seen (a hinge box whose centre lies in the door box widened by 10 % of its width per side) -> opening "pull"
#   (opening_source "hinges"), and the hinge side from where they are (left of the door centre -> "left"), before the handle rule;
#   no hinges seen -> the push / pull classifier (absence is ambiguous: out of view, occluded, smoke) and the handle rule as in A.
# Scores add: hinges_seen (share of true-hinged doors called hinged with hinges seen), acc_pull_when_hinges_seen,
# acc_push_pull_hinged_strict (no hinges seen -> push, the plain "hinges on my side?" rule).
def test_hinges_near_the_door():
    door = [100, 50, 80, 200]                               # x 100..180, widened by 8 px per side
    assert cascade.hinges_near(door, [[95, 60, 3, 12], [178, 60, 3, 12], [300, 60, 3, 12]]) == [0, 1]
    assert cascade.hinges_near(door, [[90, 60, 3, 12]]) == []    # centre x 91.5 < 92
    assert cascade.hinges_near(door, []) == []
    assert props.CATEGORY["hinge"] == 3


def test_decide_with_hinges():
    d = cascade.decide(0.2, "lever", (0.9, 0.1), "right", (0.5, 0.5), hinge_xs=[105.0], door_cx=140.0)
    assert d["opening"] == "pull" and d["opening_source"] == "hinges"
    assert d["hinge_side"] == "left" and d["hinge_source"] == "hinges"     # hinges before the handle rule
    assert d["hinges_seen"] is True
    d = cascade.decide(0.2, "lever", (0.9, 0.1), "right", (0.5, 0.5), hinge_xs=[], door_cx=140.0)
    assert d["opening"] == "push" and d["opening_source"] == "classifier"
    assert d["hinge_side"] == "right" and d["hinge_source"] == "rule" and d["hinges_seen"] is False
    d = cascade.decide(0.9, None, (0.9, 0.1), None, (0.5, 0.5), hinge_xs=[105.0], door_cx=140.0)
    assert d["door_type"] == "sliding" and d["opening"] == "slide"           # type first: hinges don't overrule it
    # without hinge_xs (step A) the answer has no step-B keys
    assert "hinges_seen" not in cascade.decide(0.2, None, (0.6, 0.4), None, (0.3, 0.7))


def test_scores_with_hinges():
    truth = [{"opening": "pull", "hinge_side": "left"}, {"opening": "push", "hinge_side": "left"},
             {"opening": "pull", "hinge_side": "right"}, {"opening": "slide", "hinge_side": "none"}]
    mk = lambda o, seen: {"door_type": "hinged", "type_source": "handle", "opening": o, "hinge_side": "left",
                          "hinge_source": "hinges" if seen else "rule", "opening_source": "hinges" if seen else "classifier", "hinges_seen": seen}
    dec = [mk("pull", True), mk("pull", False), mk("pull", False),
           {"door_type": "sliding", "type_source": "handle", "opening": "slide", "hinge_side": "none", "hinge_source": None,
            "opening_source": None, "hinges_seen": None}]
    s = cascade.scores(dec, truth)
    assert s["hinges_seen"] == pytest.approx(1 / 3)
    assert s["acc_pull_when_hinges_seen"] == pytest.approx(1.0)
    assert s["acc_push_pull_hinged"] == pytest.approx(2 / 3)
    assert s["acc_push_pull_hinged_strict"] == pytest.approx(2 / 3)   # pull (seen) ok, push ok, pull-not-seen -> push wrong
    assert "hinges_seen" not in cascade.scores(dec[:0], [])          # step A keys only without hinge decisions


def test_score_with_hinges_on_true_and_detector_boxes(tmp_path):
    root = make_export(tmp_path / "export")
    for split in ("train", "val", "test"):                       # door_a's hinge (pull side seen) in the test images
        p = root / "coco" / f"{split}.json"
        coco = json.loads(p.read_text())
        coco["categories"].append({"id": 3, "name": "hinge"})
        for im in coco["images"]:
            coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": im["id"], "category_id": 3,
                                        "bbox": [11, 20, 2, 9], "area": 18, "iscrowd": 0, "object": "hinge_a_0",
                                        "attributes": {"door": "door_a"}})
        p.write_text(json.dumps(coco))
    door_run, handle_run, _ = _runs(tmp_path, root)
    t = cascade.score(door_run, handle_run, root, "test", hinges=True)
    assert "hinges_seen" in t and 0.0 <= t["acc_push_pull_hinged_strict"] <= 1.0
    preds = {i: {"boxes": np.array([[10, 10, 60, 110], [90, 10, 140, 110], [50, 60, 56, 63], [92, 60, 95, 68], [11, 20, 13, 29]], float),
                 "scores": np.array([0.9, 0.9, 0.8, 0.8, 0.7]), "labels": np.array([0, 0, 1, 1, 2])} for i in range(4)}
    d = cascade.score(door_run, handle_run, root, "test", preds=preds, hinges=True)
    assert d["matched"] == 8 and "hinges_seen" in d
    a = cascade.score(door_run, handle_run, root, "test")             # step A unchanged
    assert "hinges_seen" not in a
