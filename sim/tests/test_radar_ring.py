# Unit tests for the radar + camera ring platform (Madelief 2026-11-12: "if we are already using radar lets just use that" — a 360°
# radar on top of the robot replaces the lidar, since lidar fails in smoke; "Camera ring" — cameras all around the robot, so a
# viewpoint needs one stop instead of 5 turning views). Specs = Claude decisions (research-based), dev log 2026-11-12.
# No Gazebo needed: python3 -m pytest sim/tests/test_radar_ring.py
#
# Platform YAML (sim/configs/robots/wheeled_ring.yaml = wheeled_low_fast + the two below, no "lidar" key):
#   camera_ring: {count: N}   N >= 2 cameras around the mast head, camera k at yaw 2 pi k / N (k = 0 is the existing front camera,
#     unchanged: camera_link, topics <name>/rgbd ..., frame camera_optical). k >= 1: link camera_link_<k> at the same position
#     (x = camera.forward, z = mast_height) with pose yaw 2 pi k / N, fixed joint camera_joint_<k> from base_link, the same 4 sensors
#     (rgbd "rgbd_<k>", boxes "boxes_<k>", boxes3d "boxes3d_<k>", seg "seg_<k>"; same fov / size / clip as camera 0) on topics
#     <name>/cam<k>/rgbd, <name>/cam<k>/boxes, <name>/cam<k>/boxes3d, <name>/cam<k>/seg, gz_frame_id camera_optical_<k>, update_rate
#     robot.RING_RATE (2 Hz: frames are only grabbed standing still; keeps the render load down).
#   radar: {height, forward, range: [min, max], azimuths, rate, beam_deg, range_res, noise}   the 360° spinning radar (Navtech-like:
#     400 azimuths per turn at 4 Hz; ~2° beam; 4 cm range bins = c / 2B for 4 GHz bandwidth at 77-81 GHz). Link radar_link at
#     (forward, 0, height); gpu_lidar sensor "radar" on topic radar_raw, gz_frame_id radar_link, horizontal samples = azimuths over
#     -pi..pi, update_rate = rate, range min / max = range. Radar sees through smoke: the sim's smoke is image-only, so nothing to
#     add for that. The /scan for Nav2 / SLAM comes from sim/building/radar_node.py (radar_raw -> radar_scan.process -> /scan).
#   load_platform: "lidar" OR "radar" must be there (one of them); camera_ring optional (count int >= 2).
#   robot.static_tfs(p): also (base_link, camera_link_<k>, [forward, 0, mast_height], [0, 0, yaw_k]) and (camera_link_<k>,
#     camera_optical_<k>, [0, 0, 0], [-pi/2, 0, -pi/2]) per ring camera; (base_link, radar_link, [forward, 0, height], [0, 0, 0]).
#   robot.bridge_args(p): /radar_raw (LaserScan, GZ -> ROS) instead of /scan for a radar platform; ring cameras are NOT bridged
#     (door_tour reads them with gz-transport, stop_frames.Grabber).
#   robot.ring_yaws(p) -> [yaw_0 = 0, yaw_1, ...] (N entries; [0.0] without a camera_ring).
#
# Module sim/doorsim/stop_frames.py: TOPICS(name="robot", cam=0) -> cam 0 as before; cam k >= 1 -> /<name>/cam<k>/rgbd/image,
#   /<name>/cam<k>/rgbd/depth_image, /<name>/cam<k>/seg/labels_map, /<name>/cam<k>/boxes, /<name>/cam<k>/boxes3d. Grabber(name, cam=0).
#
# Module sim/doorsim/radar_scan.py (numpy only):
#   radar_scan.process(ranges, angle_inc, rmin, rmax, beam, res, noise=0.0, rng=None) -> float array, same length:
#     ranges = the raw ring (index i at angle min_angle + i * angle_inc, the ring wraps around: last neighbours first). Invalid =
#     not finite or outside [rmin, rmax] -> inf. (1) beam: out[i] = min of the valid raw ranges within +-beam / 2 (rad) of beam i
#     (a strong return fills the whole beam, so obstacles look wider, never narrower); (2) range bins: r -> (floor(r / res) + 0.5) * res;
#     (3) noise: + N(0, noise) per return when noise > 0 (rng = numpy Generator); results outside [rmin, rmax] -> inf. inf stays inf.
import math, pathlib, sys, xml.etree.ElementTree as ET
import numpy as np
import pytest
import yaml

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import radar_scan, robot, stop_frames

RING = SIM / "configs" / "robots" / "wheeled_ring.yaml"


def model(p, name="robot"):
    return ET.fromstring(robot.robot_sdf(p, name)).find("model")


def pose_of(el):
    txt = el.findtext("pose")
    return [float(v) for v in txt.split()] if txt else [0.0] * 6


@pytest.fixture
def ring():
    return robot.load_platform(RING)


def test_ring_platform_file(ring):
    fast = robot.load_platform(SIM / "configs" / "robots" / "wheeled_low_fast.yaml")
    assert "lidar" not in ring and ring["camera_ring"]["count"] == 5
    assert ring["limits"] == fast["limits"] and ring["footprint"] == fast["footprint"] and ring["mast_height"] == fast["mast_height"]
    r = ring["radar"]
    assert r["azimuths"] == 400 and r["rate"] == pytest.approx(4) and r["beam_deg"] == pytest.approx(2.0)
    assert r["range_res"] == pytest.approx(0.04) and r["range"][0] < 0.5 and r["range"][1] >= 12


def test_load_platform_needs_lidar_or_radar(tmp_path, ring):
    d = yaml.safe_load(RING.read_text()); d.pop("radar")
    f = tmp_path / "none.yaml"; f.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError):
        robot.load_platform(f)
    d = yaml.safe_load(RING.read_text()); d["camera_ring"]["count"] = 1
    f.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError):
        robot.load_platform(f)


def test_ring_cameras(ring):
    m = model(ring)
    n = ring["camera_ring"]["count"]
    assert robot.ring_yaws(ring) == pytest.approx([2 * math.pi * k / n for k in range(n)])
    front = m.find("link[@name='camera_link']")
    assert front.find("sensor[@name='rgbd']").findtext("topic") == "robot/rgbd"         # camera 0 unchanged
    for k in range(1, n):
        ln = m.find(f"link[@name='camera_link_{k}']")
        assert ln is not None
        x, y, z, _, _, yaw = pose_of(ln)
        assert (x, y, z) == pytest.approx((ring["camera"]["forward"], 0, ring["mast_height"]))
        assert math.cos(yaw - 2 * math.pi * k / n) == pytest.approx(1)
        j = m.find(f"joint[@name='camera_joint_{k}']")
        assert j is not None and j.get("type") == "fixed" and j.findtext("child") == f"camera_link_{k}"
        sensors = {s.get("name"): s for s in ln.findall("sensor")}
        for base, kind in (("rgbd", "rgbd_camera"), ("boxes", "boundingbox_camera"), ("boxes3d", "boundingbox_camera"),
                           ("seg", "segmentation_camera")):
            s = sensors[f"{base}_{k}"]
            assert s.get("type") == kind and s.findtext("topic") == f"robot/cam{k}/{base}"
            assert s.findtext("gz_frame_id") == f"camera_optical_{k}"
            assert float(s.findtext("update_rate")) == pytest.approx(robot.RING_RATE)
            c = s.find("camera")
            assert float(c.findtext("horizontal_fov")) == pytest.approx(1.3) and c.find("image").findtext("width") == "640"
    assert m.find(f"link[@name='camera_link_{n}']") is None
    assert model(ring, "bot_7").find("link[@name='camera_link_1']").find("sensor[@name='rgbd_1']").findtext("topic") == "bot_7/cam1/rgbd"


def test_radar_sensor(ring):
    m = model(ring)
    assert m.find("link[@name='lidar_link']") is None
    ln = m.find("link[@name='radar_link']")
    r = ring["radar"]
    assert pose_of(ln)[:3] == pytest.approx([r["forward"], 0, r["height"]])
    s = ln.find("sensor[@name='radar']")
    assert s.get("type") == "gpu_lidar" and s.findtext("topic") == "radar_raw" and s.findtext("gz_frame_id") == "radar_link"
    assert float(s.findtext("update_rate")) == pytest.approx(r["rate"])
    h = s.find("lidar").find("scan").find("horizontal")
    assert int(h.findtext("samples")) == r["azimuths"]
    rng = s.find("lidar").find("range")
    assert [float(rng.findtext("min")), float(rng.findtext("max"))] == pytest.approx(r["range"])


def test_ring_tfs_and_bridge(ring):
    tfs = {(p, c): (xyz, rpy) for p, c, xyz, rpy in robot.static_tfs(ring)}
    n = ring["camera_ring"]["count"]
    for k in range(1, n):
        xyz, rpy = tfs[("base_link", f"camera_link_{k}")]
        assert xyz == pytest.approx([ring["camera"]["forward"], 0, ring["mast_height"]])
        assert rpy[:2] == pytest.approx([0, 0]) and math.cos(rpy[2] - 2 * math.pi * k / n) == pytest.approx(1)
        assert tfs[(f"camera_link_{k}", f"camera_optical_{k}")][1] == pytest.approx([-math.pi / 2, 0, -math.pi / 2])
    r = ring["radar"]
    assert tfs[("base_link", "radar_link")][0] == pytest.approx([r["forward"], 0, r["height"]])
    args = robot.bridge_args(ring)
    assert "/radar_raw@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan" in args
    assert not any(a.startswith("/scan@") for a in args)
    assert not any("/cam1/" in a for a in args)


def test_ring_yaws_without_ring():
    assert robot.ring_yaws(robot.load_platform(SIM / "configs" / "robots" / "wheeled_low.yaml")) == [0.0]


def test_stop_frames_topics():
    assert stop_frames.TOPICS("robot") == stop_frames.TOPICS("robot", 0)
    assert stop_frames.TOPICS("robot")["rgb"] == "/robot/rgbd/image"
    t = stop_frames.TOPICS("bot", 3)
    assert t == {"rgb": "/bot/cam3/rgbd/image", "depth": "/bot/cam3/rgbd/depth_image", "seg": "/bot/cam3/seg/labels_map",
                 "boxes": "/bot/cam3/boxes", "boxes3d": "/bot/cam3/boxes3d"}


# --- radar_scan.process ---
INC = 2 * math.pi / 400


def test_beam_widens_a_return():
    r = np.full(400, np.inf); r[100] = 3.0
    out = radar_scan.process(r, INC, 0.2, 15.0, beam=math.radians(2.0), res=0.04)
    hit = np.nonzero(np.isfinite(out))[0]
    assert set(hit) == {99, 100, 101}                          # +-1 deg = +-1.11 steps of 0.9 deg
    assert out[100] == pytest.approx((math.floor(3.0 / 0.04) + 0.5) * 0.04)


def test_beam_takes_nearest_and_wraps():
    r = np.full(400, 5.01); r[0] = 2.01; r[2] = 1.01             # x.x1: away from bin edges (float floor)
    out = radar_scan.process(r, INC, 0.2, 15.0, beam=math.radians(2.0), res=0.04)
    assert out[399] == pytest.approx(2.02) and out[1] == pytest.approx(1.02) and out[0] == pytest.approx(2.02)
    assert out[200] == pytest.approx(5.02)


def test_invalid_and_out_of_range():
    r = np.array([np.nan, 0.05, 20.0, np.inf] + [4.01] * 396)
    out = radar_scan.process(r, INC, 0.2, 15.0, beam=0.0, res=0.04)
    assert np.isinf(out[:4]).all() and out[10] == pytest.approx(4.02)


def test_noise():
    r = np.full(400, 4.01)
    a = radar_scan.process(r, INC, 0.2, 15.0, beam=0.0, res=0.04, noise=0.02, rng=np.random.default_rng(0))
    b = radar_scan.process(r, INC, 0.2, 15.0, beam=0.0, res=0.04, noise=0.02, rng=np.random.default_rng(0))
    assert np.array_equal(a, b) and 0.01 < np.std(a) < 0.03 and abs(np.mean(a) - 4.02) < 0.01
