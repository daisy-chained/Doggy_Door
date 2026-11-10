# Unit tests for the robot platform builder (phase 2e, Madelief 2026-11-09: wheeled base first, Spot later behind the same cmd_vel;
# Jackal-size 0.5 x 0.43 m; camera at 0.4 m and 1.2 m). No Gazebo needed: python3 -m pytest sim/tests/test_robot.py
#
# One robot template + a YAML per platform (sim/configs/robots/*.yaml), so other robots can slot in later.
#   robot.load_platform(path) -> dict (the YAML, checked: missing keys / non-positive sizes / wheels wider than the base -> ValueError)
#   robot.robot_sdf(p, name="robot") -> SDF text <sdf version="1.9"><model name=name>: links base_link, left_wheel, right_wheel,
#     front_caster, rear_caster, camera_link, lidar_link; revolute joints left_wheel_joint / right_wheel_joint (axis y, wheels at
#     y = +/- separation / 2, z = wheel radius); casters on fixed/ball joints touching the floor; base_link collision box =
#     footprint length x width x base height, bottom at the clearance; link poses are in the model frame (model origin on the floor
#     under the base centre). camera_link at z = mast_height, x = camera.forward, a mast visual from the base top up to it.
#     Sensors on camera_link copy the training rig (v3: horizontal_fov 1.3, 640 x 480, clip 0.1-12, 10 Hz): rgbd_camera "rgbd",
#     boundingbox_camera "boxes" (2d), boundingbox_camera "boxes3d" (3d), segmentation_camera "seg" (instance); topics under the
#     model name (<name>/rgbd, <name>/boxes, <name>/boxes3d, <name>/seg), because the building worlds keep the v3 rig on rgbd / boxes; gpu_lidar "scan" (topic scan) on lidar_link. Plugins: gz::sim::systems::DiffDrive (left_joint, right_joint,
#     wheel_separation, wheel_radius, topic cmd_vel, odom_topic odom, tf_topic /tf, frame_id odom, child_frame_id base_link,
#     max_velocity / max_angular_velocity from limits), gz::sim::systems::JointStatePublisher (topic /joint_states). No Label plugin (robot = background).
#   robot.nav2_footprint(p) -> "[[x, y], ...]": the footprint rectangle corners (Nav2 footprint parameter string).
#   robot.static_tfs(p) -> [(parent, child, [x, y, z], [roll, pitch, yaw])]: base_link -> camera_link, camera_link -> camera_optical
#     (ROS optical frame: rpy -pi/2, 0, -pi/2), base_link -> lidar_link. z in base_link = height above the floor (base_link on the floor).
#   robot.bridge_args(p, name="robot") -> ["topic@ros_type<dir>gz_type", ...] for ros_gz_bridge parameter_bridge: /cmd_vel ROS->GZ
#     (]), and GZ->ROS ([): /clock, /odom, /tf, /scan, /<name>/rgbd/image, /<name>/rgbd/depth_image, /<name>/rgbd/camera_info,
#     /joint_states.
import math, pathlib, sys, xml.etree.ElementTree as ET
import pytest
import yaml

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import robot

ROBOTS = SIM / "configs" / "robots"
PLATFORMS = sorted(ROBOTS.glob("*.yaml"))


def pose_of(el):
    txt = el.findtext("pose")
    return [float(v) for v in txt.split()] if txt else [0.0] * 6


def model(p, name="robot"):
    root = ET.fromstring(robot.robot_sdf(p, name))
    assert root.tag == "sdf"
    m = root.find("model")
    assert m is not None and m.get("name") == name
    return m


def link(m, name):
    el = m.find(f"link[@name='{name}']")
    assert el is not None, name
    return el


def plugin(m, cls):
    for pl in m.iter("plugin"):
        if pl.get("name") == cls:
            return pl
    return None


@pytest.fixture(params=PLATFORMS, ids=lambda p: p.stem)
def plat(request):
    return robot.load_platform(request.param)


def test_platform_files_exist():
    names = {p.stem for p in PLATFORMS}
    assert {"wheeled_low", "wheeled_high"} <= names


def test_madelief_numbers():
    low = robot.load_platform(ROBOTS / "wheeled_low.yaml")
    high = robot.load_platform(ROBOTS / "wheeled_high.yaml")
    for p in (low, high):
        assert p["footprint"] == {"length": 0.50, "width": 0.43}
    assert low["mast_height"] == pytest.approx(0.4)
    assert high["mast_height"] == pytest.approx(1.2)


@pytest.mark.parametrize("bad", [
    lambda d: d.pop("footprint"),
    lambda d: d.pop("mast_height"),
    lambda d: d["wheel"].update(radius=0),
    lambda d: d["footprint"].update(width=-0.4),
    lambda d: d["wheel"].update(separation=0.6),   # wheels stick out past a 0.43 m wide base
])
def test_load_platform_rejects_bad_files(tmp_path, bad):
    d = yaml.safe_load((ROBOTS / "wheeled_low.yaml").read_text())
    bad(d)
    f = tmp_path / "bad.yaml"
    f.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError):
        robot.load_platform(f)


def test_links_and_joints(plat):
    m = model(plat)
    for n in ["base_link", "left_wheel", "right_wheel", "front_caster", "rear_caster", "camera_link", "lidar_link"]:
        link(m, n)
    for side, sign in [("left", 1), ("right", -1)]:
        j = m.find(f"joint[@name='{side}_wheel_joint']")
        assert j is not None and j.get("type") == "revolute"
        assert j.findtext("parent") == "base_link" and j.findtext("child") == f"{side}_wheel"
        x, y, z = pose_of(link(m, f"{side}_wheel"))[:3]
        assert y == pytest.approx(sign * plat["wheel"]["separation"] / 2)
        assert z == pytest.approx(plat["wheel"]["radius"])          # wheel touches the floor
        axis = [float(v) for v in j.find("axis").findtext("xyz").split()]
        assert abs(axis[1]) == pytest.approx(1) and axis[0] == 0 and axis[2] == 0


def test_casters_touch_floor(plat):
    m = model(plat)
    r = plat["caster"]["radius"]
    xs = []
    for n in ["front_caster", "rear_caster"]:
        x, y, z = pose_of(link(m, n))[:3]
        assert z == pytest.approx(r)
        xs.append(x)
    assert xs[0] > 0 > xs[1]                                       # one in front, one behind the wheels
    assert max(abs(x) for x in xs) <= plat["footprint"]["length"] / 2


def test_base_box_matches_footprint(plat):
    m = model(plat)
    base = link(m, "base_link")
    size = [float(v) for v in base.find("collision").find("geometry").find("box").findtext("size").split()]
    assert size == pytest.approx([plat["footprint"]["length"], plat["footprint"]["width"], plat["base"]["height"]])
    coll_z = pose_of(base.find("collision"))[2] + pose_of(base)[2]
    assert coll_z - plat["base"]["height"] / 2 == pytest.approx(plat["base"]["clearance"])
    mass = float(base.find("inertial").findtext("mass"))
    assert mass == pytest.approx(plat["base"]["mass"])


def test_camera_height_and_rig(plat):
    m = model(plat)
    cam = link(m, "camera_link")
    x, y, z = pose_of(cam)[:3]
    assert z == pytest.approx(plat["mast_height"])
    assert x == pytest.approx(plat["camera"]["forward"])
    sensors = {s.get("name"): s for s in cam.findall("sensor")}
    assert sensors["rgbd"].get("type") == "rgbd_camera"
    for n in ("rgbd", "boxes", "boxes3d", "seg"):                  # own topics: the building's v3 rig uses rgbd / boxes
        assert sensors[n].findtext("topic") == f"robot/{n}"
    assert model(plat, "bot_7").find("link[@name='camera_link']").find("sensor[@name='rgbd']").findtext("topic") == "bot_7/rgbd"
    assert sensors["boxes"].get("type") == "boundingbox_camera"
    assert sensors["boxes"].find("camera").findtext("box_type") == "2d"
    assert sensors["boxes3d"].find("camera").findtext("box_type") == "3d"
    assert sensors["seg"].get("type") == "segmentation_camera"
    for s in sensors.values():                                     # same intrinsics as the training rig
        c = s.find("camera")
        assert float(c.findtext("horizontal_fov")) == pytest.approx(1.3)
        assert c.find("image").findtext("width") == "640" and c.find("image").findtext("height") == "480"
        assert float(c.find("clip").findtext("near")) == pytest.approx(0.1)
        assert float(c.find("clip").findtext("far")) == pytest.approx(12)
        assert float(s.findtext("update_rate")) == pytest.approx(10)


def test_lidar(plat):
    m = model(plat)
    lid = link(m, "lidar_link")
    assert pose_of(lid)[2] == pytest.approx(plat["lidar"]["height"])
    s = lid.find("sensor[@name='scan']")
    assert s is not None and s.get("type") == "gpu_lidar" and s.findtext("topic") == "scan"
    h = s.find("lidar").find("scan").find("horizontal")
    assert int(h.findtext("samples")) == plat["lidar"]["samples"]
    rng = s.find("lidar").find("range")
    assert [float(rng.findtext("min")), float(rng.findtext("max"))] == pytest.approx(plat["lidar"]["range"])


def test_diff_drive_plugin(plat):
    m = model(plat)
    dd = plugin(m, "gz::sim::systems::DiffDrive")
    assert dd is not None and dd.get("filename") == "gz-sim-diff-drive-system"
    assert dd.findtext("left_joint") == "left_wheel_joint" and dd.findtext("right_joint") == "right_wheel_joint"
    assert float(dd.findtext("wheel_separation")) == pytest.approx(plat["wheel"]["separation"])
    assert float(dd.findtext("wheel_radius")) == pytest.approx(plat["wheel"]["radius"])
    assert dd.findtext("topic") == "cmd_vel"
    # wheel odometry drifts when the wheels slip (tour 2026-11-10: bumped a door leaf, Nav2 then thought it was in a wall),
    # so the diff drive's odom / tf go to unbridged topics; the ground-truth OdometryPublisher below owns /odom and /tf
    assert dd.findtext("odom_topic") not in ("odom", "/odom") and dd.findtext("tf_topic") not in ("tf", "/tf")
    assert float(dd.findtext("max_velocity")) == pytest.approx(plat["limits"]["max_speed"])
    assert float(dd.findtext("max_angular_velocity")) == pytest.approx(plat["limits"]["max_turn"])
    jsp = plugin(m, "gz::sim::systems::JointStatePublisher")
    assert jsp is not None and jsp.findtext("topic") == "/joint_states"   # absolute, so the bridge finds it


def test_ground_truth_odometry(plat):
    m = model(plat)
    op = plugin(m, "gz::sim::systems::OdometryPublisher")
    assert op is not None and op.get("filename") == "gz-sim-odometry-publisher-system"
    assert op.findtext("odom_frame") == "odom" and op.findtext("robot_base_frame") == "base_link"
    assert op.findtext("odom_topic") == "/odom" and op.findtext("tf_topic") == "/tf"
    assert op.findtext("dimensions") == "2"
    assert op.find("gaussian_noise") is None          # truth, no noise (AMCL / SLAM later)


def test_robot_is_background(plat):
    m = model(plat)
    assert plugin(m, "gz::sim::systems::Label") is None


def test_model_name(plat):
    assert model(plat, "bot_7").get("name") == "bot_7"


def test_nav2_footprint(plat):
    pts = yaml.safe_load(robot.nav2_footprint(plat))
    L, W = plat["footprint"]["length"] / 2, plat["footprint"]["width"] / 2
    assert len(pts) == 4
    flat = [v for pt in sorted(map(tuple, pts)) for v in pt]
    want = [v for pt in sorted([(L, W), (L, -W), (-L, W), (-L, -W)]) for v in pt]
    assert flat == pytest.approx(want)


def test_static_tfs(plat):
    tfs = {(p, c): (xyz, rpy) for p, c, xyz, rpy in robot.static_tfs(plat)}
    xyz, rpy = tfs[("base_link", "camera_link")]
    assert xyz == pytest.approx([plat["camera"]["forward"], 0, plat["mast_height"]])
    xyz, rpy = tfs[("camera_link", "camera_optical")]
    assert xyz == pytest.approx([0, 0, 0]) and rpy == pytest.approx([-math.pi / 2, 0, -math.pi / 2])
    xyz, rpy = tfs[("base_link", "lidar_link")]
    assert xyz == pytest.approx([plat["lidar"]["forward"], 0, plat["lidar"]["height"]])


def test_bridge_args(plat):
    args = robot.bridge_args(plat)
    assert "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist" in args
    want = {"/clock": "rosgraph_msgs/msg/Clock[gz.msgs.Clock", "/odom": "nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/tf": "tf2_msgs/msg/TFMessage[gz.msgs.Pose_V", "/scan": "sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "/robot/rgbd/image": "sensor_msgs/msg/Image[gz.msgs.Image", "/robot/rgbd/depth_image": "sensor_msgs/msg/Image[gz.msgs.Image",
            "/robot/rgbd/camera_info": "sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/joint_states": "sensor_msgs/msg/JointState[gz.msgs.Model"}
    for topic, rest in want.items():
        assert f"{topic}@{rest}" in args, topic
    assert "/bot_7/rgbd/image@sensor_msgs/msg/Image[gz.msgs.Image" in robot.bridge_args(plat, "bot_7")


# ---- Nav2 (phase 2e, Madelief 2026-11-07: Nav2 + waypoints over the approach poses; at a door stop at ~2.5 m, look, then ~1 m) ----
#   robot.nav2_params(p, base) -> new dict (base = the parsed nav2_bringup params/nav2_params.yaml, left untouched): every
#     base_frame_id / robot_base_frame "base_footprint" -> "base_link"; every use_sim_time -> True; local + global costmap: no
#     robot_radius, footprint = nav2_footprint(p); MPPI FollowPath vx_max = max_speed, wz_max = max_turn; velocity_smoother
#     max_velocity = [max_speed, 0, max_turn], min_velocity = [-max_speed, 0, -max_turn].
#   robot.door_tour(poses, start) -> [{"door", "side", "space", "stops": [far pose, near pose]}]: one visit per (door, side) with
#     all its poses reachable and clear; stops sorted far -> near by dist; visits ordered greedily: next = the visit whose first
#     stop is nearest (straight line) to the last stop of the previous visit (start = [x, y] for the first).
import copy

NAV2_BASE = pathlib.Path("/opt/ros/jazzy/share/nav2_bringup/params/nav2_params.yaml")


def walk(d, path=()):
    if isinstance(d, dict):
        for k, v in d.items():
            yield path + (k,), k, v
            yield from walk(v, path + (k,))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from walk(v, path + (i,))


@pytest.mark.skipif(not NAV2_BASE.exists(), reason="nav2_bringup not installed")
def test_nav2_params(plat):
    base = yaml.safe_load(NAV2_BASE.read_text())
    keep = copy.deepcopy(base)
    out = robot.nav2_params(plat, base)
    assert base == keep                                           # input untouched
    for path, k, v in walk(out):
        if k in ("base_frame_id", "robot_base_frame"):
            assert v != "base_footprint", path
        if k == "use_sim_time":
            assert v is True, path
    for cm in ("local_costmap", "global_costmap"):
        prm = out[cm][cm]["ros__parameters"]
        assert "robot_radius" not in prm
        assert yaml.safe_load(prm["footprint"]) == yaml.safe_load(robot.nav2_footprint(plat))
    fp = out["controller_server"]["ros__parameters"]["FollowPath"]
    assert fp["vx_max"] == pytest.approx(plat["limits"]["max_speed"])
    assert fp["wz_max"] == pytest.approx(plat["limits"]["max_turn"])
    vs = out["velocity_smoother"]["ros__parameters"]
    s, w = plat["limits"]["max_speed"], plat["limits"]["max_turn"]
    assert vs["max_velocity"] == pytest.approx([s, 0, w]) and vs["min_velocity"] == pytest.approx([-s, 0, -w])


@pytest.mark.skipif(not NAV2_BASE.exists(), reason="nav2_bringup not installed")
def test_nav2_param_lists_have_one_type(plat):
    # ROS params files reject lists that mix ints and floats ("Sequence should be of same type"): Nav2 died on [0.5, 0, 1.0]
    # in the first live tour (2026-11-10). Every list in the written params must be all-int, all-float, all-str or all-bool.
    for path, _, v in walk(robot.nav2_params(plat, yaml.safe_load(NAV2_BASE.read_text()))):
        if isinstance(v, list) and v:
            assert len({type(x) for x in v}) == 1, (path, v)


def P(door, side, dist, x, y, ok=True, clear=True, space="corridor"):
    return {"door": door, "side": side, "space": space, "dist": dist, "x": x, "y": y, "yaw": 0.0, "reachable": ok, "clear": clear}


def test_door_tour_order_and_stops():
    poses = [P(1, "front", 1.0, 1, 1), P(1, "front", 2.5, 1, 2.5),
             P(2, "front", 2.5, 10, 2.5), P(2, "front", 1.0, 10, 1),
             P(3, "front", 2.5, 4, 2.5), P(3, "front", 1.0, 4, 1),
             P(3, "back", 2.5, 4, -2.5, space="r1"), P(3, "back", 1.0, 4, -1, space="r1")]
    tour = robot.door_tour(poses, [0, 2.5])
    assert [(v["door"], v["side"]) for v in tour] == [(1, "front"), (3, "front"), (3, "back"), (2, "front")]
    for v in tour:
        assert [s["dist"] for s in v["stops"]] == [2.5, 1.0]
    assert tour[2]["space"] == "r1"


def test_door_tour_skips_unreachable_or_blocked():
    poses = [P(1, "front", 2.5, 1, 2.5), P(1, "front", 1.0, 1, 1),
             P(1, "back", 2.5, 1, -2.5, ok=False), P(1, "back", 1.0, 1, -1, ok=False),
             P(2, "front", 2.5, 5, 2.5), P(2, "front", 1.0, 5, 1, clear=False)]
    tour = robot.door_tour(poses, [0, 0])
    assert [(v["door"], v["side"]) for v in tour] == [(1, "front")]


def test_door_tour_real_building():
    import json
    f = SIM / "building" / "out" / "bld_train_000.building.json"
    if not f.exists():
        pytest.skip("no generated building")
    b = json.loads(f.read_text())
    tour = robot.door_tour(b["poses"], [1.0, 4.0])
    good = {(q["door"], q["side"]) for q in b["poses"] if q["reachable"] and q.get("clear", True)}
    assert {(v["door"], v["side"]) for v in tour} <= good and tour
