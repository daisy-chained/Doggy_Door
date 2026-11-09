"""Robot platform builder (robot run phase 2e): one SDF template + a YAML per platform (sim/configs/robots/*.yaml).

Frames: base_link x forward, y left, z up; the model origin sits on the floor under the base centre, so every link pose in the
SDF is in that model frame. Wheeled differential-drive base: two drive wheels on revolute joints (axis y), two frictionless ball
casters (front + back) on fixed joints, a camera on a mast and a planar lidar. The camera carries the same four sensors as the
training rig in door_room_v3.sdf (rgbd, 2d boxes, 3d boxes, instance segmentation; fov 1.3, 640 x 480, clip 0.1-12, 10 Hz).
The robot has no Label plugin: it is background for the segmentation camera. Also gives the Nav2 footprint string, the static TFs
(Gazebo does not publish the fixed sensor frames) and the ros_gz_bridge arguments. Only stdlib + PyYAML."""
import copy
import math
import pathlib

import yaml

CASTER_MASS = 0.2      # kg per ball caster (not in the YAML)
SENSOR_MASS = 0.1      # kg for camera_link / lidar_link, so the physics engine has something to work with
MAST_RADIUS = 0.01     # m, mast visual only (no collision)
FOV, IMG_W, IMG_H, CLIP, RATE = 1.3, 640, 480, (0.1, 12), 10   # the training rig (door_room_v3.sdf)

_REQUIRED = {
    "name": None,
    "footprint": ("length", "width"),
    "base": ("height", "clearance", "mass"),
    "wheel": ("radius", "width", "separation", "mass"),
    "caster": ("radius",),
    "mast_height": None,
    "camera": ("forward",),
    "lidar": ("height", "forward", "range", "samples"),
    "limits": ("max_speed", "max_turn"),
}
_POSITIVE = ["footprint.length", "footprint.width", "base.height", "base.mass", "wheel.radius", "wheel.width",
             "wheel.separation", "wheel.mass", "caster.radius", "mast_height", "lidar.height", "lidar.samples",
             "limits.max_speed", "limits.max_turn"]


def _get(p, dotted):
    for k in dotted.split("."):
        p = p[k]
    return p


# --- platform file ---
def load_platform(path):
    """The platform YAML as a dict. ValueError for missing keys, non-positive sizes or wheels wider than the base."""
    p = yaml.safe_load(pathlib.Path(path).read_text())
    if not isinstance(p, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    for key, subs in _REQUIRED.items():
        if key not in p:
            raise ValueError(f"{path}: missing key '{key}'")
        for sub in subs or ():
            if not isinstance(p[key], dict) or sub not in p[key]:
                raise ValueError(f"{path}: missing key '{key}.{sub}'")
    for dotted in _POSITIVE:
        v = _get(p, dotted)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
            raise ValueError(f"{path}: '{dotted}' must be a positive number, got {v!r}")
    if p["base"]["clearance"] < 0:
        raise ValueError(f"{path}: 'base.clearance' must not be negative")
    rng = p["lidar"]["range"]
    if not (isinstance(rng, (list, tuple)) and len(rng) == 2 and 0 < rng[0] < rng[1]):
        raise ValueError(f"{path}: 'lidar.range' must be [min, max] with 0 < min < max, got {rng!r}")
    w = p["wheel"]
    if w["separation"] + w["width"] > p["footprint"]["width"]:
        raise ValueError(f"{path}: wheels stick out past the base (separation {w['separation']} + wheel width {w['width']}"
                         f" > footprint width {p['footprint']['width']})")
    return p


# --- SDF helpers ---
def _f(v):
    return f"{v:.4f}"


def _pose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
    return f"<pose>{' '.join(_f(v) for v in (x, y, z, roll, pitch, yaw))}</pose>"


def _box_inertia(m, sx, sy, sz):
    return (m / 12 * (sy**2 + sz**2), m / 12 * (sx**2 + sz**2), m / 12 * (sx**2 + sy**2))


def _cyl_inertia(m, r, h):
    """Cylinder along its own z axis: (ixx, iyy, izz)."""
    side = m / 12 * (3 * r**2 + h**2)
    return (side, side, m * r**2 / 2)


def _sphere_inertia(m, r):
    i = 0.4 * m * r**2
    return (i, i, i)


def _inertial(mass, inertia, pose=""):
    ixx, iyy, izz = inertia
    return (f"<inertial>{pose}<mass>{_f(mass)}</mass><inertia><ixx>{_f(ixx)}</ixx><ixy>0</ixy><ixz>0</ixz>"
            f"<iyy>{_f(iyy)}</iyy><iyz>0</iyz><izz>{_f(izz)}</izz></inertia></inertial>")


def _friction(mu):
    return f"<surface><friction><ode><mu>{_f(mu)}</mu><mu2>{_f(mu)}</mu2></ode></friction></surface>"


def _material(r, g, b):
    return f"<material><ambient>{_f(r)} {_f(g)} {_f(b)} 1</ambient><diffuse>{_f(r)} {_f(g)} {_f(b)} 1</diffuse></material>"


def _geom(shape):
    return f"<geometry>{shape}</geometry>"


def _cylinder(r, h):
    return f"<cylinder><radius>{_f(r)}</radius><length>{_f(h)}</length></cylinder>"


def _sphere(r):
    return f"<sphere><radius>{_f(r)}</radius></sphere>"


def _joint(name, kind, parent, child, axis=""):
    return f'<joint name="{name}" type="{kind}"><parent>{parent}</parent><child>{child}</child>{axis}</joint>'


# --- links ---
def _base_link(p):
    L, W = p["footprint"]["length"], p["footprint"]["width"]
    h, c, m = p["base"]["height"], p["base"]["clearance"], p["base"]["mass"]
    z = c + h / 2
    box = f"<box><size>{_f(L)} {_f(W)} {_f(h)}</size></box>"
    return (f'<link name="base_link">{_inertial(m, _box_inertia(m, L, W, h), _pose(z=z))}'
            f'<collision name="collision">{_pose(z=z)}{_geom(box)}</collision>'
            f'<visual name="visual">{_pose(z=z)}{_geom(box)}{_material(0.15, 0.15, 0.17)}</visual></link>')


def _wheel_link(p, name, side):
    w = p["wheel"]
    r, width, m = w["radius"], w["width"], w["mass"]
    ixx, iyy, izz = _cyl_inertia(m, r, width)          # cylinder axis is y here: swap y and z
    roll = _pose(roll=math.pi / 2)
    return (f'<link name="{name}">{_pose(y=side * w["separation"] / 2, z=r)}{_inertial(m, (ixx, izz, iyy))}'
            f'<collision name="collision">{roll}{_geom(_cylinder(r, width))}{_friction(1.0)}</collision>'
            f'<visual name="visual">{roll}{_geom(_cylinder(r, width))}{_material(0.05, 0.05, 0.05)}</visual></link>')


def _caster_link(p, name, sign):
    r = p["caster"]["radius"]
    x = sign * (p["footprint"]["length"] / 2 - r)
    return (f'<link name="{name}">{_pose(x=x, z=r)}{_inertial(CASTER_MASS, _sphere_inertia(CASTER_MASS, r))}'
            f'<collision name="collision">{_geom(_sphere(r))}{_friction(0.0)}</collision>'
            f'<visual name="visual">{_geom(_sphere(r))}{_material(0.3, 0.3, 0.3)}</visual></link>')


def _camera(name, topic, kind, extra=""):
    return (f'<sensor name="{name}" type="{kind}"><topic>{topic}</topic><update_rate>{RATE}</update_rate>'
            f'<always_on>true</always_on><gz_frame_id>camera_optical</gz_frame_id><camera>{extra}'
            f'<horizontal_fov>{FOV}</horizontal_fov><image><width>{IMG_W}</width><height>{IMG_H}</height></image>'
            f'<clip><near>{CLIP[0]}</near><far>{CLIP[1]}</far></clip></camera></sensor>')


def _camera_link(p, name="robot"):
    top = p["base"]["clearance"] + p["base"]["height"]
    mast_len = p["mast_height"] - top
    mast = ""
    if mast_len > 0:       # thin cylinder from the base top up to the camera (visual only)
        mast = (f'<visual name="mast">{_pose(z=-mast_len / 2)}{_geom(_cylinder(MAST_RADIUS, mast_len))}'
                f'{_material(0.4, 0.4, 0.4)}</visual>')
    body = "<box><size>0.0400 0.0800 0.0400</size></box>"
    sensors = "".join([
        _camera("rgbd", f"{name}/rgbd", "rgbd_camera"),
        _camera("boxes", f"{name}/boxes", "boundingbox_camera", "<box_type>2d</box_type>"),
        _camera("boxes3d", f"{name}/boxes3d", "boundingbox_camera", "<box_type>3d</box_type>"),
        _camera("seg", f"{name}/seg", "segmentation_camera", "<segmentation_type>instance</segmentation_type>"),
    ])
    return (f'<link name="camera_link">{_pose(x=p["camera"]["forward"], z=p["mast_height"])}'
            f'{_inertial(SENSOR_MASS, _box_inertia(SENSOR_MASS, 0.04, 0.08, 0.04))}'
            f'<visual name="body">{_geom(body)}{_material(0.1, 0.1, 0.1)}</visual>{mast}{sensors}</link>')


def _lidar_link(p):
    lid = p["lidar"]
    lo, hi = lid["range"]
    r, h = 0.04, 0.05
    sensor = (f'<sensor name="scan" type="gpu_lidar"><topic>scan</topic><update_rate>{RATE}</update_rate>'
              f'<always_on>true</always_on><gz_frame_id>lidar_link</gz_frame_id><lidar><scan><horizontal>'
              f'<samples>{int(lid["samples"])}</samples><resolution>1</resolution><min_angle>{_f(-math.pi)}</min_angle>'
              f'<max_angle>{_f(math.pi)}</max_angle></horizontal></scan>'
              f'<range><min>{_f(lo)}</min><max>{_f(hi)}</max><resolution>0.0100</resolution></range></lidar></sensor>')
    return (f'<link name="lidar_link">{_pose(x=lid["forward"], z=lid["height"])}'
            f'{_inertial(SENSOR_MASS, _cyl_inertia(SENSOR_MASS, r, h))}'
            f'<visual name="visual">{_geom(_cylinder(r, h))}{_material(0.1, 0.1, 0.1)}</visual>{sensor}</link>')


def _plugins(p, name):
    w, lim = p["wheel"], p["limits"]
    drive = (f'<plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive">'
             f'<left_joint>left_wheel_joint</left_joint><right_joint>right_wheel_joint</right_joint>'
             f'<wheel_separation>{_f(w["separation"])}</wheel_separation><wheel_radius>{_f(w["radius"])}</wheel_radius>'
             f'<topic>cmd_vel</topic><odom_topic>odom</odom_topic><tf_topic>/tf</tf_topic>'
             f'<frame_id>odom</frame_id><child_frame_id>base_link</child_frame_id>'
             f'<odom_publish_frequency>30</odom_publish_frequency>'
             f'<max_velocity>{_f(lim["max_speed"])}</max_velocity>'
             f'<max_angular_velocity>{_f(lim["max_turn"])}</max_angular_velocity></plugin>')
    joints = ('<plugin filename="gz-sim-joint-state-publisher-system" name="gz::sim::systems::JointStatePublisher">'
              '<topic>/joint_states</topic></plugin>')
    return drive + joints


# --- public builders ---
def robot_sdf(p, name="robot"):
    """SDF text for the robot model (not static; spawn it with a model pose on the floor)."""
    axis = "<axis><xyz>0 1 0</xyz></axis>"
    parts = [
        _base_link(p),
        _wheel_link(p, "left_wheel", +1), _wheel_link(p, "right_wheel", -1),
        _caster_link(p, "front_caster", +1), _caster_link(p, "rear_caster", -1),
        _camera_link(p, name), _lidar_link(p),
        _joint("left_wheel_joint", "revolute", "base_link", "left_wheel", axis),
        _joint("right_wheel_joint", "revolute", "base_link", "right_wheel", axis),
        _joint("front_caster_joint", "fixed", "base_link", "front_caster"),
        _joint("rear_caster_joint", "fixed", "base_link", "rear_caster"),
        _joint("camera_joint", "fixed", "base_link", "camera_link"),
        _joint("lidar_joint", "fixed", "base_link", "lidar_link"),
        _plugins(p, name),
    ]
    body = "\n    ".join(parts)
    return (f'<?xml version="1.0"?>\n<sdf version="1.9">\n  <model name="{name}">\n    <static>false</static>\n'
            f'    {body}\n  </model>\n</sdf>\n')


def nav2_footprint(p):
    """Nav2 footprint parameter string: the footprint rectangle corners in base_link, e.g. "[[0.25, 0.215], ...]"."""
    L, W = p["footprint"]["length"] / 2, p["footprint"]["width"] / 2
    return "[" + ", ".join(f"[{_f(x)}, {_f(y)}]" for x, y in [(L, W), (L, -W), (-L, -W), (-L, W)]) + "]"


def static_tfs(p):
    """[(parent, child, [x, y, z], [roll, pitch, yaw])] for the fixed sensor frames (z = height above the floor)."""
    lid = p["lidar"]
    return [
        ("base_link", "camera_link", [p["camera"]["forward"], 0.0, p["mast_height"]], [0.0, 0.0, 0.0]),
        ("camera_link", "camera_optical", [0.0, 0.0, 0.0], [-math.pi / 2, 0.0, -math.pi / 2]),
        ("base_link", "lidar_link", [lid["forward"], 0.0, lid["height"]], [0.0, 0.0, 0.0]),
    ]


def bridge_args(p, name="robot"):
    """ros_gz_bridge parameter_bridge arguments: /cmd_vel ROS -> GZ, everything else GZ -> ROS."""
    out = ["/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist"]
    for topic, ros, gz in [
        ("/clock", "rosgraph_msgs/msg/Clock", "gz.msgs.Clock"),
        ("/odom", "nav_msgs/msg/Odometry", "gz.msgs.Odometry"),
        ("/tf", "tf2_msgs/msg/TFMessage", "gz.msgs.Pose_V"),
        ("/scan", "sensor_msgs/msg/LaserScan", "gz.msgs.LaserScan"),
        (f"/{name}/rgbd/image", "sensor_msgs/msg/Image", "gz.msgs.Image"),
        (f"/{name}/rgbd/depth_image", "sensor_msgs/msg/Image", "gz.msgs.Image"),
        (f"/{name}/rgbd/camera_info", "sensor_msgs/msg/CameraInfo", "gz.msgs.CameraInfo"),
        ("/joint_states", "sensor_msgs/msg/JointState", "gz.msgs.Model"),
    ]:
        out.append(f"{topic}@{ros}[{gz}")
    return out


# --- Nav2 + door tour ---
def _patch(node):
    """In place, recursively: base_footprint -> base_link frames, use_sim_time True, rectangular MPPI critics."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k in ("base_frame_id", "robot_base_frame") and v == "base_footprint":
                node[k] = "base_link"
            elif k == "use_sim_time":
                node[k] = True
            elif k == "consider_footprint":
                node[k] = True
            else:
                _patch(v)
    elif isinstance(node, list):
        for v in node:
            _patch(v)


def nav2_params(p, base):
    """Copy of the nav2_bringup params for this platform (base is left untouched): base_link frames, sim time, rectangular
    footprint in both costmaps (no robot_radius), MPPI + velocity smoother limits from the platform."""
    out = copy.deepcopy(base)
    _patch(out)
    for cm in ("local_costmap", "global_costmap"):
        prm = out[cm][cm]["ros__parameters"]
        prm.pop("robot_radius", None)
        prm["footprint"] = nav2_footprint(p)
    s, w = p["limits"]["max_speed"], p["limits"]["max_turn"]
    fp = out["controller_server"]["ros__parameters"]["FollowPath"]
    fp["vx_max"], fp["wz_max"] = s, w
    vs = out["velocity_smoother"]["ros__parameters"]
    vs["max_velocity"] = [float(s), 0.0, float(w)]   # one type per list, or ROS rejects the params file
    vs["min_velocity"] = [-float(s), 0.0, -float(w)]
    return out


def door_tour(poses, start):
    """Visits [{"door", "side", "space", "stops"}]: one per (door, side) whose poses are all reachable and clear, stops far -> near
    (dist descending); ordered greedily by straight-line distance from the previous visit's last stop (start = [x, y])."""
    groups = {}
    for q in poses:
        groups.setdefault((q["door"], q["side"]), []).append(q)
    todo = []
    for (door, side), qs in groups.items():
        if all(q["reachable"] and q.get("clear", True) for q in qs):
            stops = sorted(qs, key=lambda q: -q["dist"])
            todo.append({"door": door, "side": side, "space": stops[0]["space"], "stops": stops})
    tour, here = [], (start[0], start[1])
    while todo:
        nxt = min(todo, key=lambda v: (math.hypot(v["stops"][0]["x"] - here[0], v["stops"][0]["y"] - here[1]),
                                       v["door"], v["side"]))
        todo.remove(nxt)
        tour.append(nxt)
        here = (nxt["stops"][-1]["x"], nxt["stops"][-1]["y"])
    return tour
