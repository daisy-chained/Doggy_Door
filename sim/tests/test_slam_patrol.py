# Unit tests for run 4 (SLAM patrol). Madelief 2026-11-11 ~17:05: run 4 explores with "the grid on the growing map" (run 3's viewpoint
# grid, planned again on the live SLAM map after every viewpoint; when no grid point is left but unknown space remains, go to the
# nearest map edge), then "Build it and queue it". Technical rules = Claude decisions, dev log 2026-11-11.
# No ROS needed: python3 -m pytest sim/tests/test_slam_patrol.py
#
# Module sim/doorsim/live_map.py (numpy only):
#   live_map.from_grid(width, height, res, origin_xy, data) -> {"occ", "unknown", "res", "origin"}: a ROS nav_msgs/OccupancyGrid's
#     fields (data = row-major list of int8, row 0 = the BOTTOM row at origin y, values 0..100 or -1 unknown) as a hinge_look map dict:
#     row 0 = the TOP row (as hinge_look.load_map); occ = value >= 50 OR unknown (bool HxW); unknown = value < 0 (bool HxW);
#     origin = (x, y) tuple of floats; res float.
#
# Module sim/doorsim/patrol.py, additions:
#   patrol.viewpoints(m, start, radius, spacing=1.5, pass_radius=None, anchor=None): anchor = (ax, ay) -> the grid is fixed in the world
#     instead of to the map corner: x = ax + spacing * i, y = ay + spacing * j for ANY integers i, j (negative too) whose point lies on
#     the map (a SLAM map grows and its origin moves; the grid must not shift). Same filters and greedy order as without an anchor.
#     anchor=None: unchanged.
#   patrol.frontiers(m, start, pass_radius, radius, min_len=0.5, reach_out=1.0) -> [{"x", "y", "size"}]: m has "unknown" too.
#     Frontier cells = known-free cells (not m["occ"]) that are 4-connected to the start cell through known-free cells, and have an
#     unknown cell among their 8 neighbours. Clusters = 8-connected groups of frontier cells; drop clusters with fewer than
#     min_len / res cells. Goal per cluster: of the cells in patrol.reachable(m, start, pass_radius) that are also
#     hinge_look.free(m, x, y, radius) and within reach_out m of some cell of the cluster, the one nearest to the cluster's centroid
#     (cell centres); none -> skip the cluster. (x, y) = the goal cell centre; size = number of cluster cells * res (m).
#     Sorted by straight-line distance of the goal from start.
import math, pathlib, sys
import numpy as np
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(SIM / "tests"))
from doorsim import hinge_look, live_map, patrol
from test_patrol import two_rooms, grid_map, fill


# --- live_map.from_grid ---
def test_from_grid_flips_rows_and_marks_unknown():
    # 3 wide, 2 high; bottom row: free, occupied, unknown; top row: 100, 0, 30
    data = [0, 100, -1, 100, 0, 30]
    m = live_map.from_grid(3, 2, 0.05, (-1.0, 2.0), data)
    assert m["res"] == pytest.approx(0.05) and m["origin"] == (-1.0, 2.0)
    assert m["occ"].tolist() == [[True, False, False], [False, True, True]]
    assert m["unknown"].tolist() == [[False, False, False], [False, False, True]]
    assert m["occ"].dtype == bool and m["unknown"].dtype == bool


def test_from_grid_matches_cell_convention():
    data = [0] * 20 * 10
    data[3 * 20 + 7] = 100                                   # row 3 from the bottom, column 7
    m = live_map.from_grid(20, 10, 0.1, (0.5, -0.5), data)
    x, y = 0.5 + 7.5 * 0.1, -0.5 + 3.5 * 0.1
    assert m["occ"][patrol._cell(m, x, y)] and m["occ"].sum() == 1


# --- viewpoints with a world anchor ---
def test_viewpoints_anchor_grid_fixed_in_world():
    m = two_rooms()
    a = (1.0, 1.0)
    vp = patrol.viewpoints(m, (1.0, 1.0), 0.3, spacing=1.5, anchor=a)
    assert vp and all(abs((x - a[0]) / 1.5 - round((x - a[0]) / 1.5)) < 1e-6 and abs((y - a[1]) / 1.5 - round((y - a[1]) / 1.5)) < 1e-6
                      for x, y in vp)
    assert (1.0, 1.0) in [(round(x, 6), round(y, 6)) for x, y in vp]
    # the same world in a bigger map with another origin (a grown SLAM map): the same points
    big = {"occ": np.ones((m["occ"].shape[0] + 40, m["occ"].shape[1] + 60), bool), "res": m["res"], "origin": (-2.0, -1.0)}
    big["occ"][20:20 + m["occ"].shape[0], 40:40 + m["occ"].shape[1]] = m["occ"]
    vp2 = patrol.viewpoints(big, (1.0, 1.0), 0.3, spacing=1.5, anchor=a)
    assert sorted((round(x, 6), round(y, 6)) for x, y in vp2) == sorted((round(x, 6), round(y, 6)) for x, y in vp)


def test_viewpoints_anchor_negative_side():
    m = grid_map(6.0, 4.0); m["origin"] = (-3.0, -2.0)
    fill_ = lambda x0, y0, x1, y1: fill({"occ": m["occ"], "res": m["res"], "origin": (0.0, 0.0)}, x0 + 3, y0 + 2, x1 + 3, y1 + 2)
    fill_(-3, -2, 3, -1.9); fill_(-3, 1.9, 3, 2); fill_(-3, -2, -2.9, 2); fill_(2.9, -2, 3, 2)
    vp = patrol.viewpoints(m, (0.0, 0.0), 0.3, spacing=1.5, anchor=(0.0, 0.0))
    xs = sorted({round(x, 6) for x, _ in vp})
    assert -1.5 in xs and 1.5 in xs and 0.0 in xs                 # grid points left of the anchor as well


def test_viewpoints_no_anchor_unchanged():
    m = two_rooms()
    assert patrol.viewpoints(m, (1.0, 1.0), 0.3) == patrol.viewpoints(m, (1.0, 1.0), 0.3, anchor=None)


# --- frontiers ---
def half_known():
    """two_rooms, everything above y = 4.0 unknown (lower room + the band 3.05..4.0 of the upper room / pocket known)."""
    m = two_rooms()
    H = m["occ"].shape[0]
    unk = np.zeros_like(m["occ"])
    unk[:H - int(round(4.0 / m["res"])), :] = True
    m["unknown"] = unk
    m["occ"] = m["occ"] | unk
    return m


def test_frontiers_through_doorway_not_pocket():
    m = half_known()
    fr = patrol.frontiers(m, (2.0, 1.5), 0.25, 0.3)
    assert fr, "the upper room's edge is a frontier"
    assert all(set(f) >= {"x", "y", "size"} for f in fr)
    assert any(f["y"] > 3.05 and f["x"] < 7.95 for f in fr)
    for f in fr:
        assert hinge_look.free(m, f["x"], f["y"], 0.3)
        assert patrol.reachable(m, (2.0, 1.5), 0.25)[patrol._cell(m, f["x"], f["y"])]
    # the walled-off pocket (x > 8) is not connected through known-free cells: its edge is no frontier, so no goal near it
    assert sum(f["size"] for f in fr) < 8.0 + 1e-6
    assert all(f["size"] >= 0.5 - 1e-9 for f in fr)


def test_frontiers_none_when_fully_known():
    m = two_rooms(); m["unknown"] = np.zeros_like(m["occ"])
    assert patrol.frontiers(m, (2.0, 1.5), 0.25, 0.3) == []


def test_frontiers_drop_small_and_sort_by_distance():
    m = two_rooms(door_open=False)
    H, W = m["occ"].shape
    unk = np.zeros_like(m["occ"])
    r = lambda y: H - 1 - int(y / m["res"])
    c = lambda x: int(x / m["res"])
    unk[r(1.5) - 1:r(1.5) + 1, c(9.0):c(9.0) + 2] = True    # a 2 x 2 cell blob of unknown in the lower room: edge 12 cells = 0.6 m
    unk[r(2.9):r(0.1) + 1, c(0.1):c(0.3) + 1] = True        # unknown strip along the left wall (2.8 m long)
    m["unknown"] = unk; m["occ"] = m["occ"] | unk
    fr = patrol.frontiers(m, (5.0, 1.5), 0.25, 0.3, min_len=1.0)
    assert len(fr) == 1 and fr[0]["x"] < 2.0                   # the blob's edge (< 1.0 m of cells) is dropped
    fr2 = patrol.frontiers(m, (8.0, 1.5), 0.25, 0.3, min_len=0.3)
    assert len(fr2) == 2 and fr2[0]["x"] > 7.0                 # nearest first from (8, 1.5): the blob
