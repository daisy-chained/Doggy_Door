# Unit tests for the learned decision tree over the cascade's cues (Madelief 2026-11-07, "lets get started on that"): the hand-built
# cascade fixes the order and thresholds itself; here a model learns them from the same per-door cues.
#   cascade.cue_table(door_run, handle_run, export_dir, split, preds=None, score_min=0.3, iou=0.5, type_run=None, state_run=None,
#                     hinges=True, angle=True) -> (rows, truth): one dict per scored door (true boxes, or detector boxes matched like
#                     score()), keys = cascade.FEATURES, missing cues as NaN; truth = {"opening", "hinge_side"[, "state"]}.
#     FEATURES: p_slide, p_push, p_left (door run, renormalised like decide), p_closed (state run, NaN without), h_lever, h_knob,
#     h_flush, h_none (handle type of the largest handle inside, one-hot), rule_side (+1 handle rule says hinge right, -1 left, 0 no
#     handle), hinges_n (hinge boxes near the door), hinges_side (+1 right of the door centre, -1 left, 0 none), panel_off,
#     near_wall (door_angle.geometry, NaN without), box_w, box_h (door box, px).
#   learned_tree.fit(rows, labels, kind="tree" | "boost", seed=0) -> model: "tree" = DecisionTreeClassifier, depth picked by
#     5-fold CV from 2..6 (min_samples_leaf 10), NaN handled by sklearn; "boost" = HistGradientBoostingClassifier (max_depth 3).
#   learned_tree.predict(model, rows) -> (labels, probabilities N x classes, classes); learned_tree.evaluate(model, rows, labels)
#     -> {"acc", "top2", "n"} (top2 = right answer among the two most likely: the robot's next try, Madelief's failsafe idea).
#   learned_tree.rules(model) -> readable text of the tree (sklearn export_text with the feature names), "" for boost.
#   train/fit_tree.py: fits on the val split, scores on the test split, true boxes and detector boxes apart (a box set with < 20 val
#     doors is skipped: {"skipped": reason}), next to the hand
#     cascade on the same doors; writes <out>/learned_tree.json + rules_<target>_<boxes>.txt.
# train/.venv/bin/python -m pytest train/tests/test_learned_tree.py
import json, math, pathlib, sys
import numpy as np
import pytest

TRAIN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))
from doordet import cascade, learned_tree
from test_cascade import _runs
from test_props import make_export


def synthetic(n=400, seed=0):
    """Rows where the opening follows a known tree: flush pull -> slide; else panel in front -> pull, behind -> push;
    no geometry (NaN) -> the classifier's p_push decides."""
    rng = np.random.default_rng(seed)
    rows, ys = [], []
    for _ in range(n):
        flush = rng.random() < 0.3
        geo = rng.random() < 0.7
        r = {f: 0.0 for f in cascade.FEATURES}
        lever = not flush and rng.random() < 0.5                      # lever / knob split: h_flush is the only clean sliding cue
        r.update(h_flush=float(flush), h_lever=float(lever), h_knob=float(not flush and not lever), p_slide=rng.random(), p_push=rng.random(), p_left=rng.random(),
                 p_closed=rng.random(), panel_off=rng.normal(0, 0.3) if geo else math.nan, near_wall=math.nan, box_w=rng.uniform(20, 200),
                 box_h=rng.uniform(60, 400))
        if flush:
            y = "slide"
        elif geo:
            y = "pull" if r["panel_off"] > 0 else "push"
        else:
            y = "push" if r["p_push"] > 0.5 else "pull"
        rows.append(r); ys.append(y)
    return rows, ys


def test_features_fixed_order():
    assert cascade.FEATURES[:3] == ["p_slide", "p_push", "p_left"]
    assert {"p_closed", "h_flush", "h_none", "rule_side", "hinges_n", "hinges_side", "panel_off", "near_wall", "box_w", "box_h"} <= set(cascade.FEATURES)


@pytest.mark.parametrize("kind", ["tree", "boost"])
def test_learns_a_known_tree_with_missing_values(kind):
    rows, ys = synthetic(600, 0)
    m = learned_tree.fit(rows, ys, kind=kind, seed=0)
    test_rows, test_ys = synthetic(300, 1)
    e = learned_tree.evaluate(m, test_rows, test_ys)
    assert e["n"] == 300 and e["acc"] > 0.9 and e["top2"] >= e["acc"]
    labels, proba, classes = learned_tree.predict(m, test_rows)
    assert sorted(classes) == ["pull", "push", "slide"] and proba.shape == (300, 3)
    assert np.allclose(proba.sum(1), 1.0)


def test_tree_rules_readable_and_seeded():
    rows, ys = synthetic(600, 0)
    a, b = learned_tree.fit(rows, ys, "tree", 0), learned_tree.fit(rows, ys, "tree", 0)
    ra = learned_tree.rules(a)
    assert ra == learned_tree.rules(b) and "h_flush" in ra and "panel_off" in ra
    assert 2 <= a.get_depth() <= 6
    assert learned_tree.rules(learned_tree.fit(rows, ys, "boost", 0)) == ""


def test_cue_table_on_tiny_export(tmp_path):
    root = make_export(tmp_path / "export")
    door_run, handle_run, _ = _runs(tmp_path, root)
    rows, truth = cascade.cue_table(door_run, handle_run, root, "test")
    assert len(rows) == len(truth) == 8
    for r in rows:
        assert list(r) == cascade.FEATURES
        assert math.isnan(r["p_closed"]) and math.isnan(r["panel_off"])        # no state run, no depth in the fixture
        assert r["h_lever"] + r["h_knob"] + r["h_flush"] + r["h_none"] == 1.0
        assert 0.0 <= r["p_slide"] <= 1.0 and r["box_w"] > 0
    assert {t["opening"] for t in truth} == {"push", "slide"}
    preds = {i: {"boxes": np.array([[10, 10, 60, 110], [90, 10, 140, 110], [50, 60, 56, 63], [92, 60, 95, 68]], float),
                 "scores": np.array([0.9, 0.9, 0.8, 0.8]), "labels": np.array([0, 0, 1, 1])} for i in range(4)}
    rows_d, truth_d = cascade.cue_table(door_run, handle_run, root, "test", preds=preds)
    assert len(rows_d) == 8


def test_fit_tree_cli(tmp_path, monkeypatch):
    root = make_export(tmp_path / "export", n=12)
    door_run, handle_run, _ = _runs(tmp_path, root)
    import fit_tree
    monkeypatch.setattr(fit_tree.score, "predict_run", lambda *a, **k: (None, {}, None))
    monkeypatch.setattr(learned_tree, "CV_FOLDS", 2, raising=False)
    out = tmp_path / "tree"
    monkeypatch.setattr(sys, "argv", ["fit_tree.py", str(door_run), str(handle_run), "det", str(root), "--out", str(out)])
    fit_tree.main()
    d = json.loads((out / "learned_tree.json").read_text())
    assert {"true_boxes", "detector_boxes", "features"} <= set(d)
    t = d["true_boxes"]["opening"]
    assert {"tree", "boost", "hand"} <= set(t) and {"acc", "top2", "n"} <= set(t["tree"])
    assert (out / "rules_opening_true_boxes.txt").exists()
