# Write the metadata for one frame: attributes.json, the door angles, and each object's 2D + 3D box.
# usage: meta.py out.json boxes.txt boxes3d.txt door_a=<rad> door_b=<rad>
#   boxes*.txt = raw output of `gz topic -e -t /boxes -n 1` (and /boxes3d); the box label is the object id.
# 3D boxes and hinge axes are saved in the ROS optical camera frame: x right, y down, z forward (metres).
import json, math, re, sys, pathlib
attrs = json.loads((pathlib.Path(__file__).parent / "attributes.json").read_text())
frame = {"classes": attrs["classes"], "camera": attrs["camera"], "objects": attrs["objects"]}
by_id = {o["id"]: o for o in frame["objects"].values()}
for o in by_id.values():
    o["box"] = o["box_3d"] = None  # stay None when the object is not in view

def field(block, name, keys):
    # protobuf text leaves out fields that are 0, so missing = 0
    m = re.search(name + r"\s*\{([^{}]*)\}", block)
    body = m.group(1) if m else ""
    return [float((re.search(rf"\b{k}:\s*(-?[\d.e+-]+)", body) or [0, 0])[1]) for k in keys]

def blocks(path):
    for b in pathlib.Path(path).read_text().split("annotated_box")[1:]:
        label = int(re.search(r"label:\s*(\d+)", b).group(1))
        if label not in by_id:
            sys.exit(f"box label {label} is not an id in attributes.json")
        yield by_id[label], b

for o, b in blocks(sys.argv[2]):
    x0, y0 = field(b, "min_corner", "xy"); x1, y1 = field(b, "max_corner", "xy")
    o["box"] = {"x_min": int(x0), "y_min": int(y0), "x_max": int(x1), "y_max": int(y1)}

# Gazebo's 3D boxes use a graphics camera frame (x right, y up, looking along -z).
# To ROS optical = turn 180 deg about x: position (x, -y, -z), quaternion q_x180 * q.
for o, b in blocks(sys.argv[3]):
    cx, cy, cz = field(b, "center", "xyz")
    qx, qy, qz, qw = field(b, "orientation", "xyzw")
    if qx == qy == qz == qw == 0:
        qw = 1.0
    o["box_3d"] = {"center": [cx, -cy, -cz], "orientation_xyzw": [qw, -qz, qy, -qx],
                   "size": field(b, "boxSize", "xyz")}

# Hinge axes are fixed in the world; put them in the camera frame too.
def rot(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return [[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]]
R = rot(*attrs["camera"]["world_rpy"])  # camera body (x forward, y left, z up) in world
def to_cam(v, is_point):
    if is_point:
        v = [a - b for a, b in zip(v, attrs["camera"]["world_position"])]
    body = [sum(R[i][j] * v[i] for i in range(3)) for j in range(3)]  # R^T v
    return [-body[1], -body[2], body[0]]  # body -> optical
for o in by_id.values():
    if "hinge_world" in o:
        h = o["hinge_world"]
        o["hinge_cam"] = {"point": to_cam(h["point"], True), "direction": to_cam(h["direction"], False)}

for arg in sys.argv[4:]:
    name, angle = arg.split("=")
    frame["objects"][name]["angle_rad"] = float(angle)
    frame["objects"][name]["state"] = "closed" if float(angle) == 0 else "open"
pathlib.Path(sys.argv[1]).write_text(json.dumps(frame, indent=2) + "\n")
n2 = sum(o["box"] is not None for o in by_id.values()); n3 = sum(o["box_3d"] is not None for o in by_id.values())
print(f"saved {sys.argv[1]} ({n2}/{len(by_id)} objects with a 2D box, {n3}/{len(by_id)} with a 3D box)")
