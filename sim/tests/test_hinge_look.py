# Unit tests for the hinge look (option A, Madelief 2026-11-10: "just A"): at a door, guess the hinge edge from the handle (the
# hinge is opposite the handle), then drive to extra spots aimed at that edge and look again. Check before building (dev log
# 2026-11-10 hinge look): one look tops out ~0.6, best at 1-2.5 m and 10-50 deg off head-on; 3 looks found 0.82 vs 0.44.
# No Gazebo / torch needed: python3 -m pytest sim/tests/test_hinge_look.py
#
# Module sim/doorsim/hinge_look.py (numpy + PIL + PyYAML only). Detections everywhere are dicts
#   {"boxes": [[x1, y1, x2, y2], ...] (pixels), "scores": [...], "labels": ["door" | "handle" | "hinge" | "window", ...]}.
#
#   hinge_look.load_map(yaml_path) -> {"occ": bool HxW (True = occupied OR unknown), "res": float, "origin": (x0, y0)}
#     from a map_server yaml (image, resolution, origin, negate 0, occupied_thresh, free_thresh, mode trinary) + its PGM.
#     Pixel value p -> occupancy (255 - p) / 255: > occupied_thresh = occupied, < free_thresh = free, else unknown.
#     Row 0 of the image is the TOP (largest y): cell (x, y) = column int((x - x0) / res), row H - 1 - int((y - y0) / res).
#   hinge_look.free(m, x, y, radius) -> bool: no occupied / unknown cell in the square of half-size ceil(radius / res) cells around
#     (x, y), and the square fully inside the map (outside = not free).
#   hinge_look.pick_door(dets, img_w, score_min=0.5) -> [x1, y1, x2, y2] of the "door" box with score >= score_min whose centre
#     column is closest to img_w / 2; None if there is none.
#   hinge_look.hinge_side_image(door_xyxy, dets, score_min=0.3) -> "left" | "right" | None: the largest "handle" box
#     (score >= score_min) whose centre lies inside the door box decides; handle centre left of the door centre -> hinges "right",
#     else "left" (same rule as train/doordet/props.hinge_from_handle). No such handle -> None.
#   hinge_look.edge_point(door, image_side, robot_yaw) -> (x, y): the door's edge on that side of the IMAGE, for a robot facing the
#     door with heading robot_yaw. door = {"pos": [x, y], "yaw": normal yaw, "width": m}; the edges are pos +/- width / 2 along the
#     wall (perpendicular to the normal); image "left" = the edge on the robot's left (its left = direction (-sin yaw, cos yaw)),
#     the edge itself always on the wall line, also when the robot stands a bit skewed.
#   hinge_look.look_poses(door, edge, stop_xy, free_fn, dist=1.5, angles=(20, 40), fallback=(1.2, 1.0))
#     -> [{"x", "y", "yaw", "angle", "dist", "toward"}]: one pose per angle (degrees off head-on), aimed at the edge
#     (yaw = heading from the pose to the edge). n = unit vector from the door centre towards stop_xy projected on the door normal
#     (the robot's side of the door); u = unit vector from the door centre to the edge. Position = edge + d * (cos a * n + s * sin a * u)
#     with s = +1 ("toward": "out", away from the door centre) tried first, then s = -1 ("in"), first at d = dist, then each fallback
#     distance (out before in at every distance). First candidate with free_fn(x, y) True wins; an angle with no free candidate is left out.
#   hinge_look.edge_column(cam_x, cam_y, cam_yaw, edge, fx, cx) -> float | None: image column of the edge for a camera at
#     (cam_x, cam_y) looking along cam_yaw (ROS camera_link: x forward, y left): column = cx - fx * left / forward;
#     None when forward <= 0.05 m (behind / beside the camera).
#   hinge_look.hinge_seen(dets, column, depth, fx, tol=0.15, score_min=0.5) -> bool: some "hinge" box with score >= score_min whose
#     centre column is within fx * tol / depth pixels of `column` (the edge, give or take tol metres at that depth). column None -> False.
#   hinge_look.edge_forward(cam_x, cam_y, cam_yaw, edge) -> float | None: how far the edge is IN FRONT of the camera (along cam_yaw,
#     = the depth image's z distance); None when <= 0.05 m.
#   hinge_look.hinge_seen(..., depth_img=None, depth_tol=0.3): with a depth image (HxW float metres; 0 / NaN / inf = no depth), a hinge
#     only counts when the median valid depth inside its box (box widened to at least 3 px around its centre column, clipped to the
#     image) is within depth_tol of `depth`. No valid pixel -> that hinge does not count. (Madelief 2026-11-10: a hinge of another door
#     further back, seen through an open door at the same column, must not count; the caller passes depth = edge_forward(...).)
#   hinge_look.pick_door_span(dets, cols, score_min=0.5) -> [x1, y1, x2, y2] | None: the "door" box (score >= score_min) with the most
#     horizontal overlap (pixels) with [min(cols), max(cols)], the image columns of the door's two edges (projected from the robot's
#     pose). No box overlapping that span by > 0 px -> None. (Claude 2026-11-10, offline test: picks the target door, not a door in the
#     next room seen through it.)
#   hinge_look.hinge_side_image(..., depth_img=None, expected=None, depth_tol=1.0): with both depth_img (HxW metres) and expected (the
#     door's depth, edge_forward of the door centre) given, a handle only counts when its median valid box depth (same as for hinges)
#     is within depth_tol of expected; no valid depth -> that handle does not count. Without them: as before. (Claude 2026-11-10:
#     wrong handle guesses 5 -> 2 on the 7 buildings' saved frames; 1.0 m because an open door's handle swings with the leaf.)
#   hinge_look.score_visits(visits) -> dict, offline score of a tour with hinge looks. Each visit: {"truth": bool (true hinge boxes of
#     this door in any of the visit's frames), "first": bool (hinge_seen on the first look), "looks": [bool, ...] (every look of the
#     visit, the first one included)}. Returns {"n_pos", "n_neg", "found_first", "found_any", "false_yes_first", "false_yes_any"}:
#     found_* over truth visits, false_yes_* over the others, any = any look True; 0.0 when a group is empty.
import math, pathlib, sys
import numpy as np
import pytest
from PIL import Image

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import hinge_look as hl


def dets(*items):
    """dets(("door", 0.9, [x1, y1, x2, y2]), ...)"""
    return {"labels": [i[0] for i in items], "scores": [i[1] for i in items], "boxes": [i[2] for i in items]}


# --- map ---
@pytest.fixture
def tiny_map(tmp_path):
    # 20 x 10 cells of 0.1 m, origin (-1, -0.5): free (254) everywhere, a wall (0) in column 10, an unknown (128: occupancy 0.5, between the thresholds) block top-left
    img = np.full((10, 20), 254, np.uint8)
    img[:, 10] = 0
    img[0:2, 0:2] = 128
    Image.fromarray(img).save(tmp_path / "m.pgm")
    (tmp_path / "m.yaml").write_text("image: m.pgm\nresolution: 0.1\norigin: [-1.0, -0.5, 0.0]\nnegate: 0\n"
                                     "occupied_thresh: 0.65\nfree_thresh: 0.25\nmode: trinary\n")
    return hl.load_map(tmp_path / "m.yaml")


def test_load_map(tiny_map):
    m = tiny_map
    assert m["occ"].shape == (10, 20) and m["res"] == pytest.approx(0.1) and tuple(m["origin"]) == pytest.approx((-1.0, -0.5))
    assert m["occ"][:, 10].all() and m["occ"][0:2, 0:2].all()          # wall + unknown count as occupied
    assert not m["occ"][5, 5]


def test_free(tiny_map):
    # column 10 = x in [0.0, 0.1); row 9 is the bottom (y in [-0.5, -0.4))
    assert hl.free(tiny_map, -0.5, 0.0, 0.1)
    assert not hl.free(tiny_map, 0.05, 0.0, 0.1)                     # on the wall
    assert not hl.free(tiny_map, -0.05, 0.0, 0.1)                    # column 9: the wall (column 10) is within one cell
    assert hl.free(tiny_map, -0.35, 0.0, 0.1)
    assert not hl.free(tiny_map, -0.95, 0.45, 0.05)                  # unknown block, top-left
    assert not hl.free(tiny_map, -0.98, 0.0, 0.1)                    # square sticks out of the map
    assert not hl.free(tiny_map, 5.0, 5.0, 0.1)                      # outside


# --- detections -> hinge side ---
def test_pick_door():
    d = dets(("door", 0.9, [0, 0, 100, 400]), ("door", 0.8, [250, 0, 400, 400]), ("door", 0.3, [300, 0, 340, 400]),
             ("handle", 0.9, [310, 200, 330, 220]))
    assert hl.pick_door(d, 640) == [250, 0, 400, 400]                # closest centre column to 320 with score >= 0.5
    assert hl.pick_door(dets(("door", 0.4, [0, 0, 10, 10])), 640) is None
    assert hl.pick_door(dets(), 640) is None


def test_hinge_side_image():
    door = [100, 0, 300, 400]                                          # centre column 200
    assert hl.hinge_side_image(door, dets(("handle", 0.9, [120, 200, 140, 220]))) == "right"
    assert hl.hinge_side_image(door, dets(("handle", 0.9, [260, 200, 280, 220]))) == "left"
    # largest handle inside wins; one outside the door and a low-score one are ignored
    d = dets(("handle", 0.9, [120, 200, 130, 205]), ("handle", 0.9, [250, 190, 290, 230]),
             ("handle", 0.9, [400, 0, 600, 300]), ("handle", 0.1, [101, 0, 199, 399]))
    assert hl.hinge_side_image(door, d) == "left"
    assert hl.hinge_side_image(door, dets(("hinge", 0.9, [120, 200, 140, 220]))) is None


# --- geometry ---
DOOR = {"pos": [0.0, 0.0], "yaw": math.pi / 2, "width": 1.0}         # door in a wall along x, normal +y; edges at x = +/- 0.5


def test_edge_point():
    # robot on the -y side facing +y (yaw pi/2): its left is -x
    assert hl.edge_point(DOOR, "left", math.pi / 2) == pytest.approx((-0.5, 0.0))
    assert hl.edge_point(DOOR, "right", math.pi / 2) == pytest.approx((0.5, 0.0))
    # robot on the +y side facing -y: its left is +x
    assert hl.edge_point(DOOR, "left", -math.pi / 2) == pytest.approx((0.5, 0.0))
    # door normal along x
    d = {"pos": [2.0, 1.0], "yaw": 0.0, "width": 0.8}
    assert hl.edge_point(d, "left", 0.0) == pytest.approx((2.0, 1.4))
    # a slightly skewed robot (tour heading error up to ~0.23 rad) still gets the edge ON the wall line
    assert hl.edge_point(DOOR, "left", math.pi / 2 + 0.25) == pytest.approx((-0.5, 0.0))
    assert hl.edge_point(DOOR, "right", math.pi / 2 - 0.25) == pytest.approx((0.5, 0.0))


def test_look_poses_out_first():
    edge = (0.5, 0.0)
    ps = hl.look_poses(DOOR, edge, (0.0, -1.0), lambda x, y: True)
    assert [p["angle"] for p in ps] == [20, 40]
    for p in ps:
        a = math.radians(p["angle"])
        assert (p["x"], p["y"]) == pytest.approx((0.5 + 1.5 * math.sin(a), -1.5 * math.cos(a)))   # n = -y, out = +x
        assert p["dist"] == pytest.approx(1.5) and p["toward"] == "out"
        assert p["yaw"] == pytest.approx(math.atan2(edge[1] - p["y"], edge[0] - p["x"]))
        assert math.hypot(p["x"] - edge[0], p["y"] - edge[1]) == pytest.approx(1.5)


def test_look_poses_fallbacks():
    edge = (0.5, 0.0)
    # nothing with x > 0.5 is free -> "in" at 1.5 m
    ps = hl.look_poses(DOOR, edge, (0.3, -2.0), lambda x, y: x <= 0.5)
    assert all(p["toward"] == "in" and p["dist"] == pytest.approx(1.5) for p in ps)
    assert ps[0]["x"] == pytest.approx(0.5 - 1.5 * math.sin(math.radians(20)))
    # only points within 1.25 m of the edge are free -> 1.2 m, out
    ps = hl.look_poses(DOOR, edge, (0.0, -1.0), lambda x, y: math.hypot(x - 0.5, y) < 1.25)
    assert all(p["dist"] == pytest.approx(1.2) and p["toward"] == "out" for p in ps)
    # 40 deg never fits -> only 20 deg comes back
    a40 = lambda x, y: abs(math.degrees(math.atan2(abs(x - 0.5), abs(y))) - 40) > 1
    assert [p["angle"] for p in hl.look_poses(DOOR, edge, (0.0, -1.0), a40)] == [20]
    assert hl.look_poses(DOOR, edge, (0.0, -1.0), lambda x, y: False) == []


def test_look_poses_robot_side_and_custom():
    # robot on the +y side: n = +y; custom angles + distance
    ps = hl.look_poses(DOOR, (-0.5, 0.0), (0.2, 3.0), lambda x, y: True, dist=2.0, angles=(30,))
    a = math.radians(30)
    assert len(ps) == 1 and (ps[0]["x"], ps[0]["y"]) == pytest.approx((-0.5 - 2.0 * math.sin(a), 2.0 * math.cos(a)))


def test_edge_column():
    fx, cx = 420.0, 320.0
    assert hl.edge_column(0.0, 0.0, 0.0, (2.0, 0.0), fx, cx) == pytest.approx(320.0)
    assert hl.edge_column(0.0, 0.0, 0.0, (2.0, 0.5), fx, cx) == pytest.approx(320.0 - 420.0 * 0.25)   # to the left
    assert hl.edge_column(1.0, 1.0, math.pi / 2, (1.5, 3.0), fx, cx) == pytest.approx(320.0 + 420.0 * 0.25)  # right of a +y camera
    assert hl.edge_column(0.0, 0.0, 0.0, (-1.0, 0.0), fx, cx) is None
    assert hl.edge_column(0.0, 0.0, 0.0, (0.01, 1.0), fx, cx) is None


def test_hinge_seen():
    fx = 420.0                                                          # tol 0.15 m at 2 m -> 31.5 px
    d = dets(("hinge", 0.9, [345, 100, 351, 140]), ("hinge", 0.4, [318, 100, 322, 140]))   # centres 348 / 320
    assert hl.hinge_seen(d, 320.0, 2.0, fx)                             # 28 px off, inside
    assert not hl.hinge_seen(d, 310.0, 2.0, fx)                         # 38 px off; the close one is under score_min
    assert hl.hinge_seen(d, 310.0, 2.0, fx, score_min=0.3)
    assert not hl.hinge_seen(dets(("handle", 0.9, [318, 100, 322, 140])), 320.0, 2.0, fx)
    assert not hl.hinge_seen(d, None, 2.0, fx)


def test_score_visits():
    v = [{"truth": True, "first": True, "looks": [True, False]},
         {"truth": True, "first": False, "looks": [False, True, False]},
         {"truth": True, "first": False, "looks": [False]},
         {"truth": False, "first": False, "looks": [False, True]},
         {"truth": False, "first": False, "looks": [False, False]}]
    s = hl.score_visits(v)
    assert s["n_pos"] == 3 and s["n_neg"] == 2
    assert s["found_first"] == pytest.approx(1 / 3) and s["found_any"] == pytest.approx(2 / 3)
    assert s["false_yes_first"] == 0.0 and s["false_yes_any"] == pytest.approx(0.5)
    e = hl.score_visits([])
    assert e["found_any"] == 0.0 and e["false_yes_any"] == 0.0 and e["n_pos"] == 0


def test_edge_forward():
    assert hl.edge_forward(0.0, 0.0, 0.0, (2.0, 0.5)) == pytest.approx(2.0)
    assert hl.edge_forward(1.0, 1.0, math.pi / 2, (1.5, 3.0)) == pytest.approx(2.0)
    assert hl.edge_forward(0.0, 0.0, 0.0, (0.01, 1.0)) is None


def test_hinge_seen_depth():
    fx = 420.0
    img = np.full((480, 640), 2.0, np.float32)
    img[:, 340:360] = 4.5                                               # something further back behind columns 340-359
    near = dets(("hinge", 0.9, [318, 100, 322, 140]))                    # centre 320, depth 2.0
    far = dets(("hinge", 0.9, [348, 100, 352, 140]))                     # centre 350, depth 4.5
    assert hl.hinge_seen(near, 320.0, 2.0, fx, depth_img=img)
    assert hl.hinge_seen(near, 320.0, 2.25, fx, depth_img=img)          # within 0.3 m
    assert not hl.hinge_seen(near, 320.0, 2.4, fx, depth_img=img)
    assert hl.hinge_seen(far, 340.0, 2.0, fx, tol=0.2)                  # column only: counts ...
    assert not hl.hinge_seen(far, 340.0, 2.0, fx, tol=0.2, depth_img=img)   # ... but it is 2.5 m further back
    thin = dets(("hinge", 0.9, [339.5, 100, 340.5, 140]))                # 1 px box on the border: widened to 3 px (339-341), median 4.5
    assert not hl.hinge_seen(thin, 340.0, 2.0, fx, depth_img=img)
    holes = img.copy(); holes[100:140, 318:323] = np.nan
    assert not hl.hinge_seen(near, 320.0, 2.0, fx, depth_img=holes)     # no valid depth in the box
    holes[100:110, 318:323] = 0.0; holes[110:140, 318:323] = 2.1
    assert hl.hinge_seen(near, 320.0, 2.0, fx, depth_img=holes)         # zeros ignored, the rest is 2.1


# --- handle guess with the door's known edges + depth (Claude 2026-11-10) ---
def test_pick_door_span():
    d = dets(("door", 0.9, [300, 0, 340, 300]),                         # small door far back, nearest the image centre
             ("door", 0.8, [150, 0, 500, 450]),                         # the target door, spanning the projected edges
             ("door", 0.95, [560, 0, 640, 450]), ("door", 0.3, [140, 0, 520, 450]))
    assert hl.pick_door(d, 640) == [300, 0, 340, 300]                  # old rule: picks the one in the next room
    assert hl.pick_door_span(d, [160.0, 490.0]) == [150, 0, 500, 450]
    assert hl.pick_door_span(d, [490.0, 160.0]) == [150, 0, 500, 450]  # order of the columns does not matter
    assert hl.pick_door_span(d, [600.0, 630.0]) == [560, 0, 640, 450]
    assert hl.pick_door_span(d, [0.0, 100.0]) is None                  # nothing overlaps
    assert hl.pick_door_span(dets(("door", 0.4, [0, 0, 300, 300])), [10.0, 200.0]) is None
    assert hl.pick_door_span(dets(), [10.0, 200.0]) is None


def test_hinge_side_image_depth():
    door = [100, 0, 300, 400]                                          # centre column 200
    img = np.full((480, 640), 1.5, np.float32)
    img[:, 250:300] = 4.0                                               # through the open door: the next room
    far_big = ("handle", 0.9, [255, 190, 295, 230])                     # right half, 4.0 m away, largest
    near_small = ("handle", 0.6, [120, 200, 135, 212])                  # left half, 1.5 m
    d = dets(far_big, near_small)
    assert hl.hinge_side_image(door, d) == "left"                       # no depth: the big far handle wins
    assert hl.hinge_side_image(door, d, depth_img=img, expected=1.5) == "right"
    assert hl.hinge_side_image(door, d, depth_img=img, expected=2.4) == "right"   # within 1.0 m
    assert hl.hinge_side_image(door, d, depth_img=img, expected=2.6) is None      # neither within 1.0 m
    assert hl.hinge_side_image(door, d, depth_img=img, expected=2.6, depth_tol=1.5) == "left"     # both in range: largest wins
    assert hl.hinge_side_image(door, d, depth_img=img) == "left"        # expected missing: no depth check
    holes = img.copy(); holes[200:212, 120:135] = np.nan
    assert hl.hinge_side_image(door, d, depth_img=holes, expected=1.5) is None    # near one has no depth, far one too far


# --- size checks (Claude 2026-11-10, from the 7-building reruns: real hinge boxes <= 0.106 m tall, 2 false hits 0.155 / 0.160 m;
#     real handle boxes >= 0.033 m on their longest side, a 0.023 m blob on a glass door's stile gave 006 d8 a wrong guess) ---
def test_hinge_seen_too_tall():
    fx = 420.0
    img = np.full((480, 640), 2.0, np.float32)
    ok = dets(("hinge", 0.9, [318, 100, 322, 140]))                      # 40 px at 2.0 m = 0.19 m ... too tall at 2 m
    assert not hl.hinge_seen(ok, 320.0, 2.0, fx, depth_img=img)
    assert hl.hinge_seen(ok, 320.0, 2.0, fx, depth_img=img, max_h=0.2)  # limit is a parameter
    assert hl.hinge_seen(ok, 320.0, 2.0, fx)                             # no depth image: no size check (column only)
    real = dets(("hinge", 0.9, [318, 100, 322, 120]))                    # 20 px at 2.0 m = 0.095 m: a real knuckle
    assert hl.hinge_seen(real, 320.0, 2.0, fx, depth_img=img)
    edge = dets(("hinge", 0.9, [318, 100, 322, 129]))                    # 29 px = 0.138 m: just under the default 0.14 m
    assert hl.hinge_seen(edge, 320.0, 2.0, fx, depth_img=img)
    tall_then_real = dets(("hinge", 0.95, [318, 100, 322, 160]), ("hinge", 0.6, [319, 200, 321, 215]))
    assert hl.hinge_seen(tall_then_real, 320.0, 2.0, fx, depth_img=img)  # a too-tall box is skipped, the next one still counts


def test_hinge_side_image_too_small():
    fx = 420.0
    door = [100, 0, 300, 400]                                          # centre column 200
    img = np.full((480, 640), 1.4, np.float32)
    blob = ("handle", 0.8, [264, 135, 269, 142])                        # 5 x 7 px at 1.4 m = 0.017 x 0.023 m: too small
    lever = ("handle", 0.5, [120, 200, 150, 210])                       # 30 px = 0.10 m wide: a real lever
    assert hl.hinge_side_image(door, dets(blob), depth_img=img, expected=1.4) == "left"      # handle right of centre -> hinge left; no fx: no size check
    assert hl.hinge_side_image(door, dets(blob), depth_img=img, expected=1.4, fx=fx) is None
    assert hl.hinge_side_image(door, dets(blob), depth_img=img, expected=1.4, fx=fx, min_size=0.02) == "left"
    assert hl.hinge_side_image(door, dets(blob, lever), depth_img=img, expected=1.4, fx=fx) == "right"   # blob dropped: the lever on the left half -> hinge right
    small_knob = ("handle", 0.5, [270, 200, 278, 208])                  # 8 px = 0.027 m: kept (longest side counts)
    assert hl.hinge_side_image(door, dets(small_knob), depth_img=img, expected=1.4, fx=fx) == "left"
    tall_thin = ("handle", 0.5, [270, 200, 272, 210])                   # 2 x 10 px = 0.007 x 0.033 m: kept, longest side >= 0.025
    assert hl.hinge_side_image(door, dets(tall_thin), depth_img=img, expected=1.4, fx=fx) == "left"
    assert hl.hinge_side_image(door, dets(blob), fx=fx) == "left"     # no depth image: no size check
