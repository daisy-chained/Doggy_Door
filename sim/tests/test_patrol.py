# Unit tests for run 3, "no markers" (Madelief 2026-11-11 ~10:45): the robot gets the floor map but no door positions; it patrols
# every reachable space, looks around, turns detector "door" boxes into door candidates on the map, and runs the door step on each.
# Found = right position (0.5 m) AND a right report. Technical rules = Claude decisions, dev log 2026-11-11 reactive door step.
# No Gazebo needed: python3 -m pytest sim/tests/test_patrol.py
#
# Module sim/doorsim/patrol.py (numpy only). Maps are hinge_look.load_map dicts {"occ": bool HxW (True = occupied / unknown),
# "res": m per cell, "origin": (x0, y0)}; cell (x, y) = column int((x - x0) / res), row H - 1 - int((y - y0) / res).
#
#   patrol.reachable(m, start, radius) -> bool HxW: cells from which hinge_look.free(m, x, y, radius) holds (robot fits) AND that
#     connect to the start cell through such cells (4-neighbour flood fill). Start not free -> all False.
#   patrol.viewpoints(m, start, radius, spacing=1.5) -> [(x, y)]: grid points x = x0 + spacing * (i + 0.5), y = y0 + spacing * (j + 0.5)
#     whose cell is in reachable(...), ordered greedily nearest-first starting from `start`.
#   patrol.view_yaws(n=5, start_yaw=0.0) -> [start_yaw + k * 2 pi / n for k in range(n)] (camera HFOV 74.5 deg -> 5 views cover 360).
#   patrol.door_estimate(box, depth, cam, fx, cx, strip=6) -> {"x", "y", "width", "depth"} | None. box = [x1, y1, x2, y2] (pixels),
#     depth = HxW float metres (0 / nan = invalid), cam = (x, y, yaw) of the camera (planar depth along yaw). depth d = median of the
#     valid depth in two strips just OUTSIDE the box (columns x1 - strip .. x1 and x2 .. x2 + strip, rows of the middle half of the box):
#     the wall around the door, not the room behind an open door; no valid strip pixel -> median inside the box; none -> None.
#     Column u = (x1 + x2) / 2, left = d * (cx - u) / fx; x = cam_x + d cos(yaw) - left sin(yaw), y = cam_y + d sin(yaw) + left cos(yaw);
#     width = (x2 - x1) * d / fx.
#   patrol.snap_to_wall(m, x, y, cam_xy, search=0.4) -> (x, y, yaw) | None: the walls here run along x or y. For each direction,
#     look at the lines through (x, y) offset 0.6 .. 1.0 m to BOTH sides along that direction (beside the doorway): along the normal
#     (within +-search) find the occupied cells; the direction with occupied cells on both sides wins, and the point moves onto the
#     mean normal coordinate of those cells. yaw = the normal pointing towards cam_xy. Neither direction -> None.
#   patrol.add_candidate(cands, est, tol=0.6) -> index: est = {"x", "y", "yaw", "width"}; same candidate = within tol AND normal
#     within 45 deg (the same door seen from the other side is another candidate); a match is averaged into it (running mean of x, y,
#     width; "n" counts the merges); otherwise est is appended with "n" = 1.
#   patrol.match(est, doors, tol=0.5) -> (door id, "front" | "back") | None: the real door whose pos is nearest within tol. Side =
#     where the robot looks from: the door's yaw points from its front space into its back space, est["yaw"] points towards the
#     robot, so cos(est yaw - door yaw) < 0 -> "front", else "back".
#   patrol.approach(est, dists=(2.0, 1.0)) -> [{"x", "y", "yaw", "dist"}]: on the normal, dist out from (x, y), facing the door.
#   patrol.snap_to_gap(m, x, y, yaw, search=1.0, gap=(0.3, 1.4)) -> (x, y): open doors are gaps in the wall on the floor map, and the
#     detector box of an open hinged door includes the swung panel, so its centre lands 0.5-0.8 m beside the doorway (patrol check
#     2026-11-11 11:15). The wall line through (x, y) runs across yaw (yaw = the wall normal, a multiple of pi / 2 here). Sample it every
#     m["res"] over along-offsets [-(search + gap[1]), search + gap[1]]; a sample is open when every cell within +-0.15 m across the
#     wall is free. Gaps = maximal runs of open samples that do NOT touch either end of the sampled range, with length (n samples *
#     res) within gap; keep those whose centre is within search of (x, y); return the nearest centre (on the wall line), else (x, y).
#   False-door filters (patrol check 2026-11-11 ~11:50: 8 false doors were panels of doors the robot had just opened, 4 were
#   furniture). Both work WITHOUT the floor map (Madelief: the next stage, run 4, gets no map):
#   patrol.near_opened(est, opened, margin=0.4) -> bool: opened = [{"x", "y", "yaw", "width"}] = where the robot itself opened doors
#     (its own estimates). True when est is on the same wall as one of them: normals parallel (|cos(est yaw - o yaw)| > 0.7, either
#     side of the wall), across-wall offset (along the opened door's normal) < 0.3 m, and along-wall distance <= 1.5 * o width + margin
#     (the swung or slid panel lies within one panel width beside the doorway).
#   patrol.in_front_of_wall(box, depth, strip=6, tol=0.25) -> bool: inside = median valid depth (0 / nan invalid) in the middle half
#     of the box (rows AND columns); wall = median valid depth in the two strips beside the box as door_estimate. True when
#     inside < wall - tol (something standing in front of the wall: furniture, a swung-out panel); a closed door is flush, an open
#     one shows the room behind (deeper). No valid inside or wall pixel -> False (can't tell, keep it).
import math, pathlib, sys
import numpy as np
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import patrol


def grid_map(w_m=10.0, h_m=6.0, res=0.05):
    H, W = int(round(h_m / res)), int(round(w_m / res))
    return {"occ": np.zeros((H, W), bool), "res": res, "origin": (0.0, 0.0)}


def fill(m, x0, y0, x1, y1):
    """Mark the world rect occupied."""
    H = m["occ"].shape[0]
    c0, c1 = int(x0 / m["res"]), int(math.ceil(x1 / m["res"]))
    r0, r1 = H - int(math.ceil(y1 / m["res"])), H - int(y0 / m["res"])
    m["occ"][max(r0, 0):r1, max(c0, 0):c1] = True


def two_rooms(door_open=True):
    """10 x 6 m box, outer walls 0.1 m; a wall along x at y = 3 (2.95..3.05) with a 0.9 m doorway at x 4.55..5.45; a closed-off
    pocket in the top right (x > 8, a wall at x = 8 with no gap) above y = 3."""
    m = grid_map()
    fill(m, 0, 0, 10, 0.1); fill(m, 0, 5.9, 10, 6); fill(m, 0, 0, 0.1, 6); fill(m, 9.9, 0, 10, 6)
    fill(m, 0, 2.95, 4.55, 3.05); fill(m, 5.45, 2.95, 10, 3.05)
    if not door_open:
        fill(m, 4.55, 2.95, 5.45, 3.05)
    fill(m, 7.95, 3.0, 8.05, 6)
    return m


# --- reachable / viewpoints ---
def test_reachable_through_open_door_not_into_pocket():
    m = two_rooms()
    r = patrol.reachable(m, (2.0, 1.5), 0.3)
    cell = lambda x, y: r[m["occ"].shape[0] - 1 - int(y / m["res"]), int(x / m["res"])]
    assert cell(2.0, 1.5) and cell(5.0, 4.5) and cell(5.0, 3.0) and not cell(9.0, 4.5) and not cell(2.0, 3.0)  # doorway yes, wall no


def test_closed_door_blocks():
    m = two_rooms(door_open=False)
    r = patrol.reachable(m, (2.0, 1.5), 0.3)
    assert not r[m["occ"].shape[0] - 1 - int(4.5 / m["res"]), int(5.0 / m["res"])]


def test_start_blocked_all_false():
    m = two_rooms()
    assert not patrol.reachable(m, (2.0, 3.0), 0.3).any()   # start inside the wall


def test_viewpoints_grid_reachable_greedy():
    m = two_rooms()
    vp = patrol.viewpoints(m, (1.0, 1.0), 0.3, spacing=1.5)
    r = patrol.reachable(m, (1.0, 1.0), 0.3)
    H = m["occ"].shape[0]
    assert vp and all(r[H - 1 - int(y / m["res"]), int(x / m["res"])] for x, y in vp)
    assert all(abs((x / 1.5 - 0.5) - round(x / 1.5 - 0.5)) < 1e-6 and abs((y / 1.5 - 0.5) - round(y / 1.5 - 0.5)) < 1e-6 for x, y in vp)
    assert not any(x > 8 and y > 3 for x, y in vp)
    assert any(y > 3 for x, y in vp)                       # the room through the doorway is covered
    assert vp[0] == min(vp, key=lambda p: math.hypot(p[0] - 1.0, p[1] - 1.0))
    for a, b in zip(vp, vp[1:]):                           # greedy: each next point is the nearest of those left
        left = vp[vp.index(b):]
        assert math.hypot(b[0] - a[0], b[1] - a[1]) <= min(math.hypot(p[0] - a[0], p[1] - a[1]) for p in left) + 1e-9


def test_view_yaws():
    y = patrol.view_yaws(5, 0.3)
    assert len(y) == 5 and y[0] == pytest.approx(0.3) and y[1] - y[0] == pytest.approx(2 * math.pi / 5)


# --- door estimate ---
FX, CX = 400.0, 320.0


def test_door_estimate_uses_wall_beside_box():
    depth = np.full((480, 640), 6.0)                       # room behind the open door: 6 m
    depth[:, :200] = 3.0; depth[:, 440:] = 3.0             # wall at 3 m beside the door box (columns 200..440)
    e = patrol.door_estimate([200, 50, 440, 400], depth, (1.0, 2.0, 0.0), FX, CX)
    assert e["depth"] == pytest.approx(3.0)
    assert e["x"] == pytest.approx(4.0) and e["y"] == pytest.approx(2.0)    # u = 320 = cx: straight ahead
    assert e["width"] == pytest.approx(240 * 3.0 / FX)


def test_door_estimate_left_of_centre_and_heading():
    depth = np.full((480, 640), 2.0)
    e = patrol.door_estimate([100, 50, 140, 400], depth, (0.0, 0.0, math.pi / 2), FX, CX)
    left = 2.0 * (CX - 120) / FX                          # 1.0 m to the camera's left; facing +y, left = -x
    assert e["x"] == pytest.approx(-left) and e["y"] == pytest.approx(2.0)


def test_door_estimate_falls_back_inside_then_none():
    depth = np.zeros((480, 640)); depth[:, 300:340] = 2.5
    assert patrol.door_estimate([300, 50, 340, 400], depth, (0, 0, 0), FX, CX)["depth"] == pytest.approx(2.5)
    assert patrol.door_estimate([300, 50, 340, 400], np.zeros((480, 640)), (0, 0, 0), FX, CX) is None


# --- snap to wall ---
def test_snap_to_wall_x_wall():
    m = two_rooms()
    x, y, yaw = patrol.snap_to_wall(m, 5.1, 3.25, (5.0, 1.0))
    assert x == pytest.approx(5.1) and abs(y - 3.0) < 0.05 and math.cos(yaw + math.pi / 2) == pytest.approx(1.0, abs=1e-6)
    x, y, yaw = patrol.snap_to_wall(m, 5.0, 2.8, (5.0, 5.0))
    assert abs(y - 3.0) < 0.05 and math.cos(yaw - math.pi / 2) == pytest.approx(1.0, abs=1e-6)


def test_snap_to_wall_y_wall_and_none():
    m = two_rooms()
    x, y, yaw = patrol.snap_to_wall(m, 7.9, 4.5, (6.0, 4.5))   # the pocket wall at x = 8 (runs along y)
    assert abs(x - 8.0) < 0.05 and y == pytest.approx(4.5) and math.cos(yaw - math.pi) == pytest.approx(1.0, abs=1e-6)
    assert patrol.snap_to_wall(m, 3.0, 1.6, (1.0, 1.6)) is None   # open floor, no wall near


# --- candidates / match / approach ---
def test_add_candidate_merge_and_sides():
    c = []
    assert patrol.add_candidate(c, {"x": 5.0, "y": 3.0, "yaw": -math.pi / 2, "width": 0.8}) == 0
    assert patrol.add_candidate(c, {"x": 5.2, "y": 3.0, "yaw": -math.pi / 2 + 0.2, "width": 1.0}) == 0
    assert c[0]["n"] == 2 and c[0]["x"] == pytest.approx(5.1) and c[0]["width"] == pytest.approx(0.9)
    assert patrol.add_candidate(c, {"x": 5.1, "y": 3.0, "yaw": math.pi / 2, "width": 0.9}) == 1   # other side
    assert patrol.add_candidate(c, {"x": 7.0, "y": 3.0, "yaw": -math.pi / 2, "width": 0.9}) == 2  # too far
    assert len(c) == 3


def test_match_side():
    doors = [{"id": 4, "pos": [5.0, 3.0], "yaw": math.pi / 2}, {"id": 7, "pos": [8.0, 3.0], "yaw": math.pi / 2}]
    assert patrol.match({"x": 5.3, "y": 3.1, "yaw": -math.pi / 2}, doors) == (4, "front")    # robot south, door normal north
    assert patrol.match({"x": 5.3, "y": 3.1, "yaw": math.pi / 2}, doors) == (4, "back")
    assert patrol.match({"x": 6.5, "y": 3.0, "yaw": 0.0}, doors) is None


def test_approach_faces_door():
    ps = patrol.approach({"x": 5.0, "y": 3.0, "yaw": -math.pi / 2})
    assert [p["dist"] for p in ps] == [2.0, 1.0]
    assert ps[0]["x"] == pytest.approx(5.0) and ps[0]["y"] == pytest.approx(1.0) and ps[1]["y"] == pytest.approx(2.0)
    assert math.cos(ps[0]["yaw"] - math.pi / 2) == pytest.approx(1.0)


# --- snap to gap (open doors) ---
def test_snap_to_gap_moves_to_doorway_centre():
    m = two_rooms()
    x, y = patrol.snap_to_gap(m, 5.6, 3.0, -math.pi / 2)
    assert abs(x - 5.0) < 0.06 and y == pytest.approx(3.0)
    x, y = patrol.snap_to_gap(m, 4.2, 3.0, math.pi / 2)
    assert abs(x - 5.0) < 0.06


def test_snap_to_gap_leaves_far_closed_and_plain_walls():
    m = two_rooms()
    assert patrol.snap_to_gap(m, 2.0, 3.0, -math.pi / 2) == (2.0, 3.0)        # doorway 3 m away
    assert patrol.snap_to_gap(two_rooms(door_open=False), 5.6, 3.0, -math.pi / 2) == (5.6, 3.0)   # closed door = wall on the map
    assert patrol.snap_to_gap(m, 8.0, 4.5, math.pi) == (8.0, 4.5)              # wall along y, no gap


def test_snap_to_gap_ignores_wide_openings():
    m = two_rooms()
    m["occ"][:] = False
    from test_patrol import fill
    fill(m, 0, 2.95, 4.0, 3.05); fill(m, 6.0, 2.95, 10, 3.05)                # a 2 m opening, not a door
    assert patrol.snap_to_gap(m, 5.6, 3.0, -math.pi / 2) == (5.6, 3.0)


# --- false-door filters (map-free) ---
OPENED = [{"x": 5.0, "y": 3.0, "yaw": -math.pi / 2, "width": 0.8}]


def test_near_opened_same_wall_beside_doorway():
    assert patrol.near_opened({"x": 6.3, "y": 3.05, "yaw": -math.pi / 2}, OPENED)       # 1.3 m along, <= 1.6
    assert patrol.near_opened({"x": 3.7, "y": 2.9, "yaw": math.pi / 2}, OPENED)         # other side of the wall too
    assert patrol.near_opened({"x": 5.0, "y": 3.0, "yaw": math.pi / 2}, OPENED)         # the opened doorway itself


def test_near_opened_false_cases():
    assert not patrol.near_opened({"x": 6.8, "y": 3.0, "yaw": -math.pi / 2}, OPENED)    # 1.8 m along: too far
    assert not patrol.near_opened({"x": 5.5, "y": 3.6, "yaw": -math.pi / 2}, OPENED)    # 0.6 m off the wall line
    assert not patrol.near_opened({"x": 5.5, "y": 3.0, "yaw": 0.0}, OPENED)             # a wall at right angles
    assert not patrol.near_opened({"x": 5.5, "y": 3.0, "yaw": -math.pi / 2}, [])


def test_in_front_of_wall():
    wall = np.full((480, 640), 3.0)
    flush = wall.copy(); flush[50:400, 200:440] = 3.05                                  # closed door, slightly recessed
    assert not patrol.in_front_of_wall([200, 50, 440, 400], flush)
    room = wall.copy(); room[50:400, 200:440] = 6.0                                     # open door: room behind
    assert not patrol.in_front_of_wall([200, 50, 440, 400], room)
    cab = wall.copy(); cab[50:400, 200:440] = 2.4                                       # cabinet 0.6 m in front of the wall
    assert patrol.in_front_of_wall([200, 50, 440, 400], cab)
    edge = room.copy(); edge[50:400, 200:240] = 2.0                                     # a panel edge at the box side: middle half decides
    assert not patrol.in_front_of_wall([200, 50, 440, 400], edge)


def test_in_front_of_wall_cant_tell():
    d = np.zeros((480, 640)); d[50:400, 200:440] = 2.0                                  # no wall depth beside the box
    assert not patrol.in_front_of_wall([200, 50, 440, 400], d)
