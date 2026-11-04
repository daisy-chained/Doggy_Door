# Unit tests for sim/doorsim/radar_doorways.py (no Gazebo). Madelief 2026-11-04: first a geometric doorway finder on the
# radar points (top-down map: wall line + gaps of door width), scored in metres against the doorway truth per distance and
# radar type. Centre error is horizontal only (radar elevation is too coarse for height).
import pathlib, sys
import numpy as np
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import radar_doorways as rd

CAM = [0.0, 0.0, 1.0]


def wall_points(gap=(-0.45, 0.45), x=3.0, span=(-3.0, 3.0), step=0.05, behind=False, noise=0.0, drop=0.0, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(span[0], span[1], step)
    if gap:
        y = y[(y < gap[0]) | (y > gap[1])]
    z = rng.uniform(0.3, 2.0, len(y))
    pts = np.stack([np.full(len(y), x), y, z], 1)
    if behind:  # the far wall of the next room, seen through the doorway
        yb = np.arange(-0.4, 0.4, step)
        pts = np.vstack([pts, np.stack([np.full(len(yb), x + 2.5), yb, np.full(len(yb), 1.0)], 1)])
    pts = pts + rng.normal(0, noise, pts.shape)
    return pts[rng.random(len(pts)) >= drop]


def test_finds_the_doorway():
    got = rd.find(wall_points(), CAM)
    assert len(got) == 1
    d = got[0]
    assert np.linalg.norm(np.array(d["center"][:2]) - [3.0, 0.0]) < 0.1
    assert d["width"] == pytest.approx(0.9, abs=0.15)
    assert d["normal"][0] < -0.9  # wall normal points back to the radar side


def test_points_behind_the_gap_are_fine():
    assert len(rd.find(wall_points(behind=True), CAM)) == 1


def test_too_narrow_or_too_wide_is_no_doorway():
    assert rd.find(wall_points(gap=(-0.15, 0.15)), CAM) == []
    assert rd.find(wall_points(gap=(-1.5, 1.5)), CAM) == []


def test_wall_end_is_no_doorway():
    assert rd.find(wall_points(gap=None, span=(-3.0, 0.0)), CAM) == []


def test_noise_and_dropout():
    got = rd.find(wall_points(noise=0.03, drop=0.3, seed=2), CAM)
    assert len(got) == 1 and np.linalg.norm(np.array(got[0]["center"][:2]) - [3.0, 0.0]) < 0.2


def test_no_points():
    assert rd.find(np.zeros((0, 3)), CAM) == []


def test_ground_truth_and_score():
    meta = {"side": "front", "camera": {"world_position": CAM, "world_rpy": [0, 0, 0]},
            "objects": {"door_a": {"class": 1, "width": 0.9, "height": 2.0,
                                   "approach": {"front": {"point": [3.0, 0.0, 0.0], "direction": [-1.0, 0.0, 0.0]}}},
                        "handle_a": {"class": 2}}}
    gt = rd.ground_truth(meta)
    assert list(gt) == ["door_a"] and gt["door_a"]["center"] == pytest.approx([3.0, 0.0, 1.0])
    rows = rd.score(rd.find(wall_points(), CAM), gt, CAM)
    assert len(rows) == 1 and rows[0]["door"] == "door_a" and rows[0]["found"]
    assert rows[0]["center_m"] < 0.1 and rows[0]["distance_m"] == pytest.approx(3.0, abs=0.05)
    miss = rd.score([], gt, CAM)
    assert miss[0]["found"] is False and miss[0]["center_m"] is None
    far = rd.score([{"center": [3.0, 2.0, 1.0], "width": 0.9, "normal": [-1, 0, 0]}], gt, CAM)
    assert far[0]["found"] is False  # a doorway 2 m off is not this door (match radius rd.MATCH_M)
