# Unit tests for sim/doorsim/radar_grid.py (no Gazebo): milliMap-style occupancy grid doorway finder (Madelief 2026-11-04,
# "just do both": a better finder + a learned one on the radar image). Rays from the sensor to each return mark the cells
# they cross as free; a doorway = a door-wide stretch of the wall line that rays went through (or that has no wall).
import pathlib, sys
import numpy as np
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import radar_doorways as rd, radar_grid as rg
from test_radar_doorways import wall_points, CAM


def with_sensor(pts, pos=CAM):
    return pts, np.tile(np.asarray(pos, float), (len(pts), 1))


def test_doorway_with_rays_through_it():
    pts, sens = with_sensor(wall_points(behind=True))
    got = rg.find(pts, sens, CAM)
    assert len(got) == 1 and np.linalg.norm(np.array(got[0]["center"][:2]) - [3.0, 0.0]) < 0.15
    assert got[0]["width"] == pytest.approx(0.9, abs=0.2) and got[0]["score"] > 0.5


def test_stray_points_in_a_seen_through_gap():
    pts = np.vstack([wall_points(behind=True, seed=s) for s in range(10)] + [np.array([[3.0, 0.1, 1.0], [3.0, -0.2, 1.1]])])
    got = rg.find(*with_sensor(pts), CAM)
    assert len(got) == 1


def test_gap_without_rays_through_still_found_but_weaker():
    seen = rg.find(*with_sensor(wall_points(behind=True)), CAM)
    blind = rg.find(*with_sensor(wall_points(behind=False)), CAM)
    assert len(blind) == 1 and blind[0]["score"] < seen[0]["score"]


def test_views_from_several_positions():
    a, sa = with_sensor(wall_points(behind=True, seed=1), [0.0, -1.0, 1.0])
    b, sb = with_sensor(wall_points(behind=True, seed=2), [0.5, 1.0, 1.0])
    got = rg.find(np.vstack([a, b]), np.vstack([sa, sb]), CAM)
    assert len(got) == 1 and got[0]["normal"][0] < -0.9


def test_no_points():
    assert rg.find(np.zeros((0, 3)), np.zeros((0, 3)), CAM) == []


# --- tuning on val (thresholds were untuned; tune on val, report on test) ---
def test_find_takes_parameter_overrides():
    pts = wall_points()
    assert len(rd.find(pts, CAM)) == 1
    assert rd.find(pts, CAM, params={"MIN_W": 1.0}) == []  # the 0.9 m gap is now too narrow
    assert len(rg.find(*with_sensor(wall_points(behind=True)), CAM, params={"MIN_W": 1.0})) == 0


def test_f1_from_rows():
    rows = [{"door": "a", "found": True, "state": "open", "in_view": True},
            {"door": "b", "found": False, "state": "open", "in_view": True},
            {"door": "c", "found": False, "state": "open", "in_view": False},   # not in view: not counted for recall
            {"door": "d", "found": False, "state": "closed", "in_view": True},  # closed: not counted for recall
            {"door": None, "false_positive": True, "found": True}]
    m = rd.detection_metrics(rows)
    assert m["recall"] == pytest.approx(0.5) and m["precision"] == pytest.approx(0.5) and m["f1"] == pytest.approx(0.5)
