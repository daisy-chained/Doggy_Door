# Write the metadata for one frame: attributes, the door joint positions, and each object's 2D + 3D box.
# usage: meta.py <world>.attributes.json out.json boxes.txt boxes3d.txt door_a=<joint pos> door_b=<joint pos>
#   boxes*.txt = raw output of `gz topic -e -t /boxes -n 1` (and /boxes3d); the box label is the object id.
# 3D boxes and hinge/slide axes are saved in the ROS optical camera frame: x right, y down, z forward (metres).
# The frame logic lives in sim/doorsim/frames.py (shared with the dataset generator).
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from doorsim.frames import build_frame, parse_text_boxes

attrs = json.loads(pathlib.Path(sys.argv[1]).read_text())
boxes2d = parse_text_boxes(pathlib.Path(sys.argv[3]).read_text(), "2d")
boxes3d = parse_text_boxes(pathlib.Path(sys.argv[4]).read_text(), "3d")
joints = dict(arg.split("=") for arg in sys.argv[5:])
try:
    frame = build_frame(attrs, boxes2d, boxes3d, joints)
except ValueError as e:
    sys.exit(str(e))
pathlib.Path(sys.argv[2]).write_text(json.dumps(frame, indent=2) + "\n")
objs = frame["objects"].values()
n2 = sum(o["box"] is not None for o in objs); n3 = sum(o["box_3d"] is not None for o in objs)
print(f"saved {sys.argv[2]} ({n2}/{len(objs)} objects with a 2D box, {n3}/{len(objs)} with a 3D box)")
