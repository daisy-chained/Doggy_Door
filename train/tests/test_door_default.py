# Unit tests for the door-property default (Madelief 2026-11-07 21:25: "hand first, boosted backup").
#   cascade.decide(..., open_panel=False): the hand mix = depth only on closed doors (recess / hinges); on open doors the opening stays the
#     classifier's unless open_panel=True (the old "panel in front of the wall -> pull" rule, kept as an option).
#   cascade.score / cue_table(..., open_panel=False) pass it on; train/score_cascade.py --open-panel.
#   learned_tree.ranked(first, model, row) -> list of every class of the model: the hand answer first, then the others by the boosted
#     model's probability, most likely first (the failsafe's order of tries).
#   learned_tree.evaluate_backup(firsts, model, rows, labels) -> {"acc", "top2", "n"}: acc = the hand answers, top2 = right answer among
#     the first two of ranked().
#   train/fit_tree.py: per target also "hand_boost" ({"acc", "top2", "n"}) and the boosted model saved as boost_<target>_<boxes>.pkl.
# train/.venv/bin/python -m pytest train/tests/test_door_default.py
import json, pathlib, pickle, sys
import numpy as np
import pytest

TRAIN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))
from doordet import cascade, learned_tree
from test_cascade import _runs
from test_learned_tree import synthetic
from test_props import make_export

BASE = (0.2, "lever", (0.9, 0.1), "right", (0.5, 0.5))   # hinged (lever), classifier says push


def test_open_doors_keep_the_classifier_by_default():
    op = cascade.decide(*BASE, hinge_xs=[], door_cx=140.0, closed=False, geom={"panel_off": 0.3, "near_wall": 0.0})
    assert (op["opening"], op["opening_source"]) == ("push", "classifier")
    op = cascade.decide(*BASE, hinge_xs=[], door_cx=140.0, closed=False, geom={"panel_off": 0.3, "near_wall": 0.0}, open_panel=True)
    assert (op["opening"], op["opening_source"]) == ("pull", "panel")
    rec = cascade.decide(*BASE, hinge_xs=[], door_cx=140.0, closed=True, geom={"panel_off": None, "near_wall": -0.06})
    assert (rec["opening"], rec["opening_source"]) == ("push", "recess")     # closed doors: depth as before


def test_ranked_hand_first_then_boost():
    rows, ys = synthetic(600, 0)
    m = learned_tree.fit(rows, ys, "boost", 0)
    test_rows, _ = synthetic(50, 1)
    _, p, classes = learned_tree.predict(m, test_rows)
    for i, r in enumerate(test_rows):
        for first in classes:
            out = learned_tree.ranked(first, m, r)
            assert out[0] == first and sorted(out) == sorted(classes) and len(set(out)) == len(out)
            rest = [c for c in out[1:]]
            pr = [p[i][classes.index(c)] for c in rest]
            assert pr == sorted(pr, reverse=True)


def test_evaluate_backup():
    rows, ys = synthetic(600, 0)
    m = learned_tree.fit(rows, ys, "boost", 0)
    test_rows, test_ys = synthetic(200, 1)
    always_push = ["push"] * len(test_rows)
    e = learned_tree.evaluate_backup(always_push, m, test_rows, test_ys)
    assert e["n"] == 200
    assert e["acc"] == pytest.approx(np.mean([y == "push" for y in test_ys]))
    want = np.mean([y in learned_tree.ranked("push", m, r)[:2] for r, y in zip(test_rows, test_ys)])
    assert e["top2"] == pytest.approx(want) and e["top2"] >= e["acc"]


def test_score_open_panel_flag(tmp_path):
    root = make_export(tmp_path / "export")
    door_run, handle_run, _ = _runs(tmp_path, root)
    a = cascade.score(door_run, handle_run, root, "test", hinges=True, angle=True, open_panel=True)
    b = cascade.score(door_run, handle_run, root, "test", hinges=True, angle=True)
    assert a["doors"] == b["doors"] == 8
    rows, _ = cascade.cue_table(door_run, handle_run, root, "test", open_panel=True)
    assert len(rows) == 8


def test_fit_tree_hand_boost_and_saved_model(tmp_path, monkeypatch):
    root = make_export(tmp_path / "export", n=12)
    door_run, handle_run, _ = _runs(tmp_path, root)
    import fit_tree
    monkeypatch.setattr(fit_tree.score, "predict_run", lambda *a, **k: (None, {}, None))
    monkeypatch.setattr(learned_tree, "CV_FOLDS", 2, raising=False)
    out = tmp_path / "tree"
    monkeypatch.setattr(sys, "argv", ["fit_tree.py", str(door_run), str(handle_run), "det", str(root), "--out", str(out)])
    fit_tree.main()
    d = json.loads((out / "learned_tree.json").read_text())
    hb = d["true_boxes"]["opening"]["hand_boost"]
    assert {"acc", "top2", "n"} <= set(hb) and hb["top2"] >= hb["acc"]
    assert hb["acc"] == pytest.approx(d["true_boxes"]["opening"]["hand"]["acc"])
    m = pickle.loads((out / "boost_opening_true_boxes.pkl").read_bytes())
    rows, _ = cascade.cue_table(door_run, handle_run, root, "test")
    assert learned_tree.ranked("push", m, rows[0])[0] == "push"
