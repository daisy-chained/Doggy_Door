# Unit tests for the fog of war on the run dashboard (Madelief 2026-11-11 ~11:05: "have the map display fog of war for everything
# that hasn't been seen"). Seen = in the robot camera's view: inside the view cone (HFOV 1.3 rad = 74.5 deg), within range (default
# 4.0 m = patrol --max-door-depth) and not behind an occupied map cell. Claude decisions (research-based), dev log 2026-11-11.
# No Gazebo needed: python3 -m pytest sim/tests/test_fog.py
#
# Module sim/doorsim/fog.py (numpy only). Maps as in test_patrol (hinge_look.load_map dicts).
#   fog.new(m, cell=0.25) -> f = {"cell": cell, "origin": (x0, y0) of the map, "seen": bool array (rows, cols)}: cols = ceil(map width
#     in m / cell), rows = ceil(map height / cell); seen[j, i] covers x in [x0 + i cell, x0 + (i + 1) cell), y in [y0 + j cell, ...):
#     row 0 = the BOTTOM (lowest y). All False.
#   fog.look(f, m, cam_xy, yaw, hfov=1.3, rng=4.0) -> number of cells newly marked seen. Rays from cam_xy over [yaw - hfov/2,
#     yaw + hfov/2], spaced so neighbouring rays are at most m["res"] apart at range rng; each ray steps m["res"] / 2 from the camera
#     up to rng and marks the fog cell of every point; it stops at the first occupied map cell (that cell's fog cell is marked: the
#     wall is seen) and at the map edge. The camera's own cell is marked.
#   fog.encode(f) -> {"cell", "origin": [x0, y0], "rows": [str, ...]}: rows[j] = "0" / "1" per column for seen row j (row 0 bottom).
import math, pathlib, sys
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import fog
from test_patrol import two_rooms


def seen_at(f, x, y):
    x0, y0 = f["origin"]
    return bool(f["seen"][int((y - y0) / f["cell"]), int((x - x0) / f["cell"])])


def test_new_shape_all_unseen():
    f = fog.new(two_rooms(), cell=0.25)
    assert f["seen"].shape == (24, 40) and not f["seen"].any() and f["origin"] == (0.0, 0.0)


def test_look_cone_and_wall():
    m = two_rooms(); f = fog.new(m)
    n = fog.look(f, m, (2.0, 1.5), math.pi / 2)
    assert n > 0 and n == f["seen"].sum()
    assert seen_at(f, 2.0, 1.5) and seen_at(f, 2.0, 2.6)       # own cell, ahead
    assert seen_at(f, 2.0, 2.96)                                # the wall itself (its near face)
    assert not seen_at(f, 2.0, 4.0)                             # behind the wall
    assert not seen_at(f, 2.0, 0.5)                             # behind the camera
    assert not seen_at(f, 3.5, 1.6)                             # 90 deg to the side, outside the cone


def test_look_through_doorway_and_range():
    m = two_rooms(); f = fog.new(m)
    fog.look(f, m, (5.0, 1.5), math.pi / 2)
    assert seen_at(f, 5.0, 4.5)                                 # through the open doorway, 3 m away
    f = fog.new(m)
    fog.look(f, m, (1.0, 1.0), 0.0)
    assert seen_at(f, 4.6, 1.0) and not seen_at(f, 5.6, 1.0)    # 3.6 m yes, 4.6 m beyond range 4.0
    f = fog.new(m)
    fog.look(f, m, (1.0, 1.0), 0.0, rng=6.0)
    assert seen_at(f, 5.6, 1.0)


def test_look_again_counts_only_new():
    m = two_rooms(); f = fog.new(m)
    fog.look(f, m, (2.0, 1.5), math.pi / 2)
    assert fog.look(f, m, (2.0, 1.5), math.pi / 2) == 0


def test_encode_rows_bottom_first():
    m = two_rooms(); f = fog.new(m)
    f["seen"][0, 3] = True
    e = fog.encode(f)
    assert e["cell"] == 0.25 and e["origin"] == [0.0, 0.0] and len(e["rows"]) == 24
    assert e["rows"][0] == "000" + "1" + "0" * 36 and set(e["rows"][1]) == {"0"}
