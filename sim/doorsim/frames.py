"""Frame metadata: attributes + 2D/3D boxes + joint states + camera pose -> one frame dict.

Frames used here:
- world: x toward the wall, y left, z up (metres)
- camera body (Gazebo sensor frame): x forward, y left, z up
- Gazebo's 3D box camera: graphics frame, x right, y up, looking along -z
- saved 3D data: ROS optical frame, x right, y down, z forward
"""
import copy, math, re


def rot(roll, pitch, yaw):
    """Rotation matrix (body -> world) for roll/pitch/yaw in radians (Gazebo's convention)."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return [[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]]


def world_to_cam(v, camera, is_point):
    """A world point or direction -> ROS optical camera frame. camera = {"world_position", "world_rpy"}."""
    R = rot(*camera["world_rpy"])
    if is_point:
        v = [a - b for a, b in zip(v, camera["world_position"])]
    body = [sum(R[i][j] * v[i] for i in range(3)) for j in range(3)]  # R^T v
    return [-body[1], -body[2], body[0]]  # body -> optical


def gl_box_to_optical(center, quat_xyzw):
    """Gazebo 3D box (graphics camera frame) -> ROS optical: turn 180 deg about x.
    position (x, -y, -z), quaternion q_x180 * q."""
    qx, qy, qz, qw = quat_xyzw
    if qx == qy == qz == qw == 0:
        qw = 1.0
    return [center[0], -center[1], -center[2]], [qw, -qz, qy, -qx]


def parse_text_boxes(text, kind):
    """Raw `gz topic -e` text of /boxes (kind "2d") or /boxes3d ("3d") -> list of box dicts.
    Protobuf text leaves out fields that are 0, so missing = 0."""
    def field(block, name, keys):
        m = re.search(name + r"\s*\{([^{}]*)\}", block)
        body = m.group(1) if m else ""
        return [float((re.search(rf"\b{k}:\s*(-?[\d.e+-]+)", body) or [0, 0])[1]) for k in keys]
    out = []
    for b in text.split("annotated_box")[1:]:
        label = int(re.search(r"label:\s*(\d+)", b).group(1))
        if kind == "2d":
            x0, y0 = field(b, "min_corner", "xy"); x1, y1 = field(b, "max_corner", "xy")
            out.append({"label": label, "x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1})
        else:
            out.append({"label": label, "center": field(b, "center", "xyz"),
                        "orientation_xyzw": field(b, "orientation", "xyzw"), "size": field(b, "boxSize", "xyz")})
    return out


def boxes_from_msg(msg, kind):
    """gz.msgs AnnotatedAxisAligned2DBox_V / AnnotatedOriented3DBox_V -> the same dicts as parse_text_boxes."""
    out = []
    for a in msg.annotated_box:
        b = a.box
        if kind == "2d":
            out.append({"label": a.label, "x_min": b.min_corner.x, "y_min": b.min_corner.y,
                        "x_max": b.max_corner.x, "y_max": b.max_corner.y})
        else:
            o = b.orientation
            out.append({"label": a.label, "center": [b.center.x, b.center.y, b.center.z],
                        "orientation_xyzw": [o.x, o.y, o.z, o.w], "size": [b.boxSize.x, b.boxSize.y, b.boxSize.z]})
    return out


def build_frame(attrs, boxes2d, boxes3d, joint_pos=None, camera=None, extra=None):
    """One frame's metadata. boxes2d/3d: lists from parse_text_boxes / boxes_from_msg (label = object id).
    joint_pos: {object name: joint position (unit in its "joint")}. camera: pose for this frame
    (default: the fixed rig pose in attrs). Raises ValueError on a box label that isn't an object id."""
    camera = camera or attrs["camera"]
    objects = copy.deepcopy(attrs["objects"])
    by_id = {o["id"]: o for o in objects.values()}
    for o in by_id.values():
        o["box"] = o["box_3d"] = None  # stay None when the object is not in view
    for b in boxes2d:
        if b["label"] not in by_id:
            raise ValueError(f"box label {b['label']} is not an id in the attributes")
        by_id[b["label"]]["box"] = {k: int(b[k]) for k in ("x_min", "y_min", "x_max", "y_max")}
    for b in boxes3d:
        if b["label"] not in by_id:
            raise ValueError(f"box label {b['label']} is not an id in the attributes")
        c, q = gl_box_to_optical(b["center"], b["orientation_xyzw"])
        by_id[b["label"]]["box_3d"] = {"center": c, "orientation_xyzw": q, "size": list(b["size"])}
    for o in objects.values():
        for key in [k for k in o if k.endswith("_world")]:
            h = o[key]
            o[key[:-6] + "_cam"] = {"point": world_to_cam(h["point"], camera, True),
                                    "direction": world_to_cam(h["direction"], camera, False)}
    for name, pos in (joint_pos or {}).items():
        objects[name]["joint_pos"] = float(pos)
        objects[name]["state"] = "closed" if abs(float(pos)) < 0.01 else "open"  # 0.01 rad or 1 cm
    frame = {"classes": attrs["classes"],
             "camera": {"world_position": list(camera["world_position"]), "world_rpy": list(camera["world_rpy"]),
                        "frame": "ROS optical: x right, y down, z forward"},
             "objects": objects}
    frame.update(extra or {})
    return frame
