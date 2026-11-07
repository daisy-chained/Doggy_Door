# Unit tests for the door angle vs the wall from depth, in the door property cascade (Madelief 2026-11-07, "lets see if we can do
# something with the door angle in relation to the surrounding wall", then "Yes, now"). Scratch check (runs/scratch/angle2.py):
# with depth only, open doors push / pull 0.74 (panel in front of the wall = pull), closed doors 0.86 (the door-stop recess).
#   door_angle.geometry(depth_m, K, box_xywh, seed=0) -> {"panel_off", "near_wall"} or None (no wall found):
#     wall  = RANSAC vertical plane (|normal y| <= 0.15, camera ~level, y down; 2 cm inlier band) over the depth points in a band
#             around the box (same rows, box widened by its width on each side, box itself left out), refit on its inliers,
#             normal toward the camera; None if fewer than 100 band points or no plane.
#     panel_off = median signed distance (+ = toward the camera) of box points 0.04..1.2 m off the wall (None if < 20 such points)
#     near_wall = median signed distance of box points within 0.12 m of the wall (None if < 20)
#   cascade.decide(..., geom=None): only with a known state (closed True / False) and a hinged door:
#     closed: hinges seen -> pull ("hinges"); else near_wall < -0.02 -> push, else pull ("recess"); near_wall None -> classifier.
#     open:   panel_off > 0 -> pull, < 0 -> push ("panel"); None -> classifier.
#     closed None or geom None -> as before.
#   cascade.score(..., angle=True): geometry per scored door from <frame>/depth.png (sensor depth, mm) and the dataset's K;
#     missing depth -> geom None. Scores gain acc_push_pull_by_<source> over true-hinged doors called hinged.
# train/.venv/bin/python -m pytest train/tests/test_door_angle.py
import json, pathlib, sys
import numpy as np
import pytest

TRAIN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))
from doordet import cascade, door_angle
from test_cascade import _runs
from test_props import make_export

K = [420.0, 0.0, 320.0, 0.0, 420.0, 240.0, 0.0, 0.0, 1.0]
BOX = [280, 140, 80, 200]   # x, y, w, h


def scene(box_depth, frac=1.0, far=6.0, rng=None):
    """Camera looking straight at a wall 3 m away; inside BOX: `frac` of the columns at box_depth (the panel), the rest the far room."""
    d = np.full((480, 640), 3.0, np.float32)
    x, y, w, h = BOX
    d[y:y + h, x:x + w] = far
    d[y:y + h, x:x + int(w * frac)] = box_depth
    if rng is not None:
        d += rng.normal(0, 0.003, d.shape).astype(np.float32)
    return d


def test_open_pull_panel_in_front():
    g = door_angle.geometry(scene(2.6, 0.6, rng=np.random.default_rng(1)), K, BOX)
    assert g["panel_off"] == pytest.approx(0.4, abs=0.03)


def test_open_push_panel_behind():
    g = door_angle.geometry(scene(3.3, 0.6, rng=np.random.default_rng(2)), K, BOX)
    assert g["panel_off"] == pytest.approx(-0.3, abs=0.03)          # the far room (3 m behind) is beyond 1.2 m: left out


def test_closed_recess_and_flush():
    assert door_angle.geometry(scene(3.06), K, BOX)["near_wall"] == pytest.approx(-0.06, abs=0.01)
    assert door_angle.geometry(scene(3.0), K, BOX)["near_wall"] == pytest.approx(0.0, abs=0.01)


def test_no_wall():
    assert door_angle.geometry(np.zeros((480, 640), np.float32), K, BOX) is None


def test_seeded():
    d = scene(2.6, 0.6, rng=np.random.default_rng(3))
    assert door_angle.geometry(d, K, BOX) == door_angle.geometry(d, K, BOX)


def test_decide_with_geometry():
    base = (0.2, "lever", (0.9, 0.1), "right", (0.5, 0.5))   # classifier says push
    rec = cascade.decide(*base, hinge_xs=[], door_cx=140.0, closed=True, geom={"panel_off": None, "near_wall": -0.06})
    assert (rec["opening"], rec["opening_source"]) == ("push", "recess")
    flush = cascade.decide(*base, hinge_xs=[], door_cx=140.0, closed=True, geom={"panel_off": None, "near_wall": 0.0})
    assert (flush["opening"], flush["opening_source"]) == ("pull", "recess")
    hin = cascade.decide(*base, hinge_xs=[105.0], door_cx=140.0, closed=True, geom={"panel_off": None, "near_wall": -0.06})
    assert (hin["opening"], hin["opening_source"]) == ("pull", "hinges")
    op = cascade.decide(*base, hinge_xs=[105.0], door_cx=140.0, closed=False, geom={"panel_off": 0.3, "near_wall": 0.0})
    assert (op["opening"], op["opening_source"]) == ("pull", "panel")
    op = cascade.decide((0.2), "lever", (0.1, 0.9), "right", (0.5, 0.5), hinge_xs=[], door_cx=140.0, closed=False,
                        geom={"panel_off": -0.2, "near_wall": None})
    assert (op["opening"], op["opening_source"]) == ("push", "panel")
    none = cascade.decide(*base, hinge_xs=[], door_cx=140.0, closed=False, geom={"panel_off": None, "near_wall": None})
    assert (none["opening"], none["opening_source"]) == ("push", "classifier")
    unk = cascade.decide(*base, hinge_xs=[], door_cx=140.0, closed=None, geom={"panel_off": 0.3, "near_wall": 0.0})
    assert (unk["opening"], unk["opening_source"]) == ("push", "classifier")        # no state: geometry not used
    sl = cascade.decide(0.9, None, (0.9, 0.1), None, (0.5, 0.5), hinge_xs=[], door_cx=140.0, closed=True, geom={"panel_off": 0.3, "near_wall": 0.0})
    assert sl["opening"] == "slide"


def test_scores_by_source():
    truth = [{"opening": "push", "hinge_side": "left"}, {"opening": "pull", "hinge_side": "left"}, {"opening": "push", "hinge_side": "left"}]
    mk = lambda o, src: {"door_type": "hinged", "type_source": "handle", "opening": o, "hinge_side": "left", "hinge_source": "rule",
                         "opening_source": src, "hinges_seen": src == "hinges"}
    s = cascade.scores([mk("push", "recess"), mk("push", "recess"), mk("push", "panel")], truth)
    assert s["acc_push_pull_by_recess"] == pytest.approx(0.5) and s["acc_push_pull_by_panel"] == pytest.approx(1.0)


def test_score_with_angle_runs_without_depth(tmp_path):
    root = make_export(tmp_path / "export")          # no depth.png: geometry None, still scores
    door_run, handle_run, _ = _runs(tmp_path, root)
    s = cascade.score(door_run, handle_run, root, "test", hinges=True, state_gt=False, angle=True)
    assert s["doors"] == 8
