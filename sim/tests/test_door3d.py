# Unit tests for sim/doorsim/door3d.py: a door's 3D position from its 2D box + depth (plane fit, glass fallback to the
# wall around it) and scoring against the doorway ground truth (no Gazebo): python3 -m pytest sim/tests
import json, math, pathlib, sys
import numpy as np
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import door3d, doors, frames

W, H = 320, 240
F = 250.0
K = [F, 0.0, W / 2, 0.0, F, H / 2, 0.0, 0.0, 1.0]


def plane_depth(z0, tilt_deg=0.0):
    """Planar z-depth (optical frame) of a plane through (0, 0, z0) turned by tilt about the camera's y axis."""
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    x, y = (u - W / 2) / F, (v - H / 2) / F
    n = np.array([math.sin(math.radians(tilt_deg)), 0.0, -math.cos(math.radians(tilt_deg))])  # toward the camera
    # ray (x, y, 1) * z hits the plane n . (p - p0) = 0
    z = (n[2] * z0) / (n[0] * x + n[1] * y + n[2])
    return z.astype(np.float32)


BOX = {"x_min": 110, "y_min": 50, "x_max": 209, "y_max": 199}  # 100 x 150 px


def test_flat_door_centre_normal_size():
    d = door3d.fit(plane_depth(3.0), K, BOX)
    assert d["source"] == "box"
    cx, cy, cz = d["center"]
    # box centre pixel (160, 125) at z 3
    assert cz == pytest.approx(3.0, abs=0.01)
    assert cx == pytest.approx((160 - W / 2) / F * 3, abs=0.01) and cy == pytest.approx((125 - H / 2) / F * 3, abs=0.01)
    assert np.dot(d["normal"], [0, 0, -1]) > math.cos(math.radians(1))  # faces the camera
    assert d["width"] == pytest.approx(100 / F * 3, rel=0.03) and d["height"] == pytest.approx(150 / F * 3, rel=0.03)


def test_tilted_door_normal():
    d = door3d.fit(plane_depth(3.0, tilt_deg=30), K, BOX)
    want = [math.sin(math.radians(30)), 0, -math.cos(math.radians(30))]
    assert math.degrees(math.acos(min(1, np.dot(d["normal"], want)))) < 2.0


def test_outliers_and_holes_inside_the_box():
    depth = plane_depth(3.0)
    rng = np.random.default_rng(0)
    m = rng.random(depth.shape) < 0.2
    depth[m] = rng.uniform(0.5, 8, m.sum())  # 20 % junk (handles, smoke false returns)
    depth[rng.random(depth.shape) < 0.2] = 0  # and holes
    d = door3d.fit(depth, K, BOX)
    assert d["center"][2] == pytest.approx(3.0, abs=0.03)


@pytest.mark.parametrize("inside", ["holes", "see_through"])
def test_glass_door_falls_back_to_the_wall_around_it(inside):
    depth = plane_depth(3.0)
    y0, y1, x0, x1 = BOX["y_min"], BOX["y_max"] + 1, BOX["x_min"], BOX["x_max"] + 1
    depth[y0:y1, x0:x1] = 0.0 if inside == "holes" else 6.0  # glass: no depth, or the room behind it
    d = door3d.fit(depth, K, BOX)
    assert d["source"] == "ring"
    assert d["center"][2] == pytest.approx(3.0, abs=0.03)
    assert np.dot(d["normal"], [0, 0, -1]) > math.cos(math.radians(3))


def test_no_depth_at_all():
    d = door3d.fit(np.zeros((H, W), np.float32), K, BOX)
    assert d is None


# --- ground truth: the doorway (not the panel, which swings) ---
def v4_meta(cam_pos, yaw, side):
    w = doors.sample_world(np.random.default_rng(0), __import__("yaml").safe_load((SIM / "configs" / "worlds_v4.yaml").read_text()))
    attrs = doors.attributes(w, "t")
    cam = {"world_position": cam_pos, "world_rpy": [0.0, 0.0, yaw]}
    meta = frames.build_frame(attrs, [], [], None, cam, {"side": side})
    return meta, attrs


def test_gt_doorway_in_the_camera_frame():
    meta, attrs = v4_meta([0.0, -1.0, 1.0], 0.0, "front")  # straight in front of door a, looking +x
    g = door3d.gt_doorway(meta, "door_a")
    h = attrs["objects"]["door_a"]["height"]
    # doorway centre on the front wall face (x 2.95), y -1, z = height / 2; optical: x right, y down, z forward
    assert g["center"] == pytest.approx([0.0, -(h / 2 - 1.0), 2.95], abs=1e-6)
    assert g["normal"] == pytest.approx([0, 0, -1], abs=1e-6)  # toward the camera's side
    assert g["width"] == attrs["objects"]["door_a"]["width"] and g["height"] == h
    meta, _ = v4_meta([6.0, -1.0, 1.0], math.pi, "back")  # behind the wall, looking -x
    g = door3d.gt_doorway(meta, "door_a")
    assert g["center"][2] == pytest.approx(6.0 - 3.05, abs=1e-6) and g["normal"] == pytest.approx([0, 0, -1], abs=1e-6)


def test_errors():
    est = {"center": [0.1, 0.0, 3.0], "normal": [0.0, 0.0, -1.0], "width": 0.9, "height": 2.0}
    gt = {"center": [0.0, 0.0, 3.0], "normal": [math.sin(math.radians(10)), 0.0, -math.cos(math.radians(10))], "width": 1.0, "height": 2.1}
    e = door3d.errors(est, gt)
    assert e["center_m"] == pytest.approx(0.1) and e["normal_deg"] == pytest.approx(10.0, abs=1e-6)
    assert e["width_m"] == pytest.approx(-0.1) and e["height_m"] == pytest.approx(-0.1)
    assert e["distance_m"] == pytest.approx(3.0)  # how far the true doorway is, for per-distance tables


# --- open doors: the doorway plane is the wall around the box, the panel only gives the opening angle ---
def open_door_depth(angle_deg, z0=3.0):
    """Wall at z0 everywhere; inside BOX a panel hinged on the box's left edge, swung `angle_deg` toward the camera."""
    depth = plane_depth(z0)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    x = (u - W / 2) / F
    hx = (BOX["x_min"] - W / 2) / F * z0  # hinge line x (camera frame) on the wall
    a = math.radians(angle_deg)
    # panel plane through the hinge line, turned by a about the vertical: points (hx + s cos a, ., z0 - s sin a)
    n = np.array([math.sin(a), 0.0, math.cos(a)])
    z = (n[0] * hx + n[2] * z0) / (n[0] * x + n[2])
    ys, xs = slice(BOX["y_min"], BOX["y_max"] + 1), slice(BOX["x_min"], BOX["x_max"] + 1)
    depth[ys, xs] = z[ys, xs]
    return depth.astype(np.float32)


def test_open_door_uses_the_wall_plane_and_reports_the_opening_angle():
    d = door3d.fit(open_door_depth(40), K, BOX)
    assert d["source"] == "wall"
    assert np.dot(d["normal"], [0, 0, -1]) > math.cos(math.radians(2))  # the wall's normal, not the panel's
    assert d["center"][2] == pytest.approx(3.0, abs=0.03)  # on the wall, not on the panel in front of it
    assert d["open_angle_deg"] == pytest.approx(40, abs=3)


def test_closed_door_keeps_the_box_plane_with_zero_opening_angle():
    d = door3d.fit(plane_depth(3.0), K, BOX)
    assert d["source"] == "box"
    assert d["open_angle_deg"] == pytest.approx(0, abs=2)


def test_glass_has_no_opening_angle():
    depth = plane_depth(3.0)
    depth[BOX["y_min"]:BOX["y_max"] + 1, BOX["x_min"]:BOX["x_max"] + 1] = 0.0
    d = door3d.fit(depth, K, BOX)
    assert d["source"] == "ring" and d["open_angle_deg"] is None


def test_floor_below_the_box_is_not_taken_for_the_wall():
    """Doors stand on the floor: the ring leaves out the band under the box, even when the floor there has most points."""
    depth = plane_depth(3.0)
    y0, y1, x0, x1 = BOX["y_min"], BOX["y_max"] + 1, BOX["x_min"], BOX["x_max"] + 1
    depth[y0:y1, x0:x1] = 0.0  # glass with holes: only the ring can give the plane
    v = np.arange(H) + 0.5
    floor_z = 0.6 * F / np.maximum(v - H / 2, 1e-6)  # floor 0.6 m below the camera (optical y down)
    depth[y1:, :] = floor_z[y1:, None]
    depth[:y0, :] = 0.0  # no depth above the box,
    depth[:, x1:] = 0.0  # nor right of it: the floor band now outweighs the wall left of the box
    d = door3d.fit(depth, K, BOX)
    assert d is not None and np.dot(d["normal"], [0, 0, -1]) > math.cos(math.radians(3))
    assert d["center"][2] == pytest.approx(3.0, abs=0.03)


@pytest.mark.parametrize("box,cut", [(BOX, False), ({**BOX, "x_min": 0}, True), ({**BOX, "y_max": H - 1}, True)])
def test_clipped_box(box, cut):
    assert door3d.clipped(box, (H, W)) is cut


def test_summary_splits_clipped_boxes():
    e = {"center_m": 0.1, "normal_deg": 1.0, "width_m": 0.0, "height_m": 0.0, "distance_m": 2.0}
    rows = [{"glass": False, "state": "closed", "failed": False, "errors": e, "clipped": c} for c in (True, False)]
    s = door3d.summary(rows)
    assert "in view" in s and "clipped" in s


def test_doorway_box_class_fits_from_the_opening(tmp_path):
    """Wide-open panel = thin sliver box; the doorway box still gives the wall's pose (Madelief 2026-11-04)."""
    import yaml
    from PIL import Image
    from doorsim import glass
    w = doors.sample_world(np.random.default_rng(0), yaml.safe_load((SIM / "configs" / "worlds_v4.yaml").read_text()))
    attrs = doors.attributes(w, "t")
    meta = frames.build_frame(attrs, [], [], None, {"world_position": [0.0, -1.0, 1.0], "world_rpy": [0.0, 0.0, 0.0]}, {"side": "front"})
    fdir = tmp_path / "ds" / "frames" / "000000"
    fdir.mkdir(parents=True)
    Image.fromarray(glass.to_png16(np.full((H, W), 2.95, np.float32))).save(fdir / "depth.png")
    m = json.loads(json.dumps(meta))
    m["objects"]["door_a"]["box"] = {"x_min": 150, "y_min": 15, "x_max": 153, "y_max": 204}
    m.update(scene=0, view=0, glass_depth_mode=None)
    (fdir / "meta.json").write_text(json.dumps(m))
    (tmp_path / "ds" / "dataset.json").write_text(json.dumps({"camera_intrinsics": {"width": W, "height": H, "K": K}}))
    dw = {r["door"]: r for r in door3d.score_dataset(tmp_path / "ds", box_class="doorway")}["door_a"]
    assert dw["box_class"] == "doorway" and not dw["failed"]
    assert dw["errors"]["center_m"] < 0.05 and dw["errors"]["normal_deg"] < 2
    dr = {r["door"]: r for r in door3d.score_dataset(tmp_path / "ds")}["door_a"]
    assert dr["box_class"] == "door"
    assert dr["failed"] or dr["errors"]["center_m"] > dw["errors"]["center_m"]
