# Unit tests for scoring a SLAM map against the true floor map (run 4 groundwork, Madelief 2026-11-11 ~13:45: "start building slam
# rn, wait for 3 to work before we integrate slam"). Metric choices are Claude decisions (research-based), dev log 2026-11-11.
# No ROS needed: python3 -m pytest sim/tests/test_slam_eval.py
#
# Module sim/doorsim/slam_eval.py (numpy + PIL + yaml).
#   slam_eval.load(yaml_path) -> {"grid": int8 HxW (1 occupied, 0 free, -1 unknown), "res", "origin": (x0, y0)}: ROS map yaml + pgm
#     (row 0 = TOP). Pixel value v (0..255, after `negate` when set): v <= 100 occupied, v >= 250 free (ROS map_saver writes 0 / 254 /
#     205), else unknown.
#   slam_eval.compare(true_m, slam_m, tol=0.1) -> {"coverage", "walls_found", "false_walls", "n_free", "n_wall", "n_slam_occ"}:
#     maps may differ in size, origin (same resolution assumed); compared in world coordinates (cell centres).
#     coverage = of the true free cells, the fraction the SLAM map calls free;
#     walls_found = of the true wall-surface cells (occupied with at least one free 4-neighbour), the fraction with a SLAM occupied
#       cell within tol (m, Chebyshev in cells: ceil(tol / res));
#     false_walls = of the SLAM occupied cells, the fraction with no true occupied cell within tol.
#     Cells outside a map count as unknown (SLAM) / occupied (true). Empty denominators -> 0.0.
#   slam_eval.save(m, stem) writes stem.pgm + stem.yaml in the same format (round trip with load).
import pathlib, sys
import numpy as np
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import slam_eval as se


def box_map(w=40, h=30, res=0.1, origin=(0.0, 0.0)):
    g = np.zeros((h, w), np.int8)
    g[0, :] = g[-1, :] = 1
    g[:, 0] = g[:, -1] = 1
    return {"grid": g, "res": res, "origin": origin}


def test_identical_maps_score_perfect():
    m = box_map()
    r = se.compare(m, m)
    assert r["coverage"] == pytest.approx(1.0) and r["walls_found"] == pytest.approx(1.0) and r["false_walls"] == pytest.approx(0.0)


def test_unknown_half_and_shifted_origin():
    t = box_map()
    s = {"grid": t["grid"].copy(), "res": 0.1, "origin": (0.0, 0.0)}
    s["grid"][:, 20:] = -1                                   # right half never seen
    r = se.compare(t, s)
    assert 0.4 < r["coverage"] < 0.6 and 0.3 < r["walls_found"] < 0.7 and r["false_walls"] == pytest.approx(0.0)
    big = np.full((50, 60), -1, np.int8); big[10:40, 5:45] = t["grid"]   # same map inside a bigger SLAM grid, other origin
    s2 = {"grid": big, "res": 0.1, "origin": (-0.5, -1.0)}
    r2 = se.compare(t, s2)
    assert r2["coverage"] == pytest.approx(1.0) and r2["walls_found"] == pytest.approx(1.0)


def test_false_walls_and_tolerance():
    t = box_map()
    s = {"grid": t["grid"].copy(), "res": 0.1, "origin": (0.0, 0.0)}
    s["grid"][15, 20] = 1                                    # a blob in the open room
    r = se.compare(t, s)
    assert r["false_walls"] == pytest.approx(1 / r["n_slam_occ"])
    s3 = {"grid": t["grid"].copy(), "res": 0.1, "origin": (0.1, 0.0)}   # every wall 1 cell (0.1 m) off: within tol
    assert se.compare(t, s3)["walls_found"] > 0.95


def test_load_save_round_trip(tmp_path):
    m = box_map(); m["grid"][5:8, 5:8] = -1
    se.save(m, tmp_path / "m")
    back = se.load(tmp_path / "m.yaml")
    assert np.array_equal(back["grid"], m["grid"]) and back["res"] == pytest.approx(0.1) and tuple(back["origin"])[:2] == (0.0, 0.0)
