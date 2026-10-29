# Door room v0 (draft for review)

First test room: one wall with two hinged doors, seen by a fixed sensor rig (RGB-D, 2D + 3D bounding-box cameras, instance segmentation camera) at 1.2 m height, 3 m from the wall.

| | Door A (right in the image) | Door B (left in the image) |
|---|---|---|
| Handle | lever | round knob |
| Hinge (seen from the camera) | right side | left side |
| Opens | away from the camera (push) | toward the camera (pull) |

Classes: `1` door · `2` handle. Walls and floor are background.

In Gazebo, each object gets its **own id** as its label (`11` door A, `12` handle A, `21` door B, `22` handle B; `0` = background, Gazebo allows at most 255). That way every box says which object it belongs to. `attributes.json` maps each id to its object and class, so exporting for training turns the ids back into door / handle.

Everything else is an **attribute**, not a class: handle type, push/pull, hinge side, door type, glass. They live in [`attributes.json`](attributes.json) (one entry per object, handles point to their door). `run_room.sh` saves a `.json` next to each frame with those attributes, each door's angle and open/closed state, and each object's 2D box (`null` when it is out of view). The raw box output is kept as `*_boxes.txt` and `*_boxes3d.txt`.

**3D ground truth** (per object, in the frame `.json`):
- `box_3d`: centre, orientation (quaternion x y z w) and size in metres, from a 3D bounding-box camera on the rig (same ids as the 2D boxes).
- `hinge_cam` (doors): a point on the hinge axis and its direction. The hinge is fixed in the world (`hinge_world` in `attributes.json`, taken from the hinge joints), and `meta.py` puts it in the camera frame.
- Everything 3D is saved in the **ROS optical camera frame**: x right, y down, z forward, so it matches depth images and point clouds. Gazebo's own 3D boxes use a graphics frame (x right, y up, looking along -z); `meta.py` converts.
- The camera pose in the world is in `attributes.json` (`camera`). The rig doesn't move yet; a moving robot will need the camera pose per frame.
- Checked: each 3D centre projected into the image lands within about 2 px of its 2D box centre; open doors stay 0.44 m from their hinge. That way each property can get its own classifier and its own confidence, and new properties don't multiply the number of classes.

```bash
bash sim/door_room/run_room.sh          # boxes + frame + metadata with doors closed, then opened; output in sim/door_room/out/
```

Open or close a door while the sim runs (angle in radians, -1.6 = wide open, 0 = closed):

```bash
gz topic -t /door_a/cmd_pos -m gz.msgs.Double -p 'data: -1.2'
```

Example frames: [closed](example/closed.png) · [open](example/open.png)

## How it's built (and why)
- Each door panel is a link on a **revolute joint** (the hinge) with limits, moved by a `JointPositionController`.
- The hinge axis sits on the panel face on the opening side. With the axis in the middle of the panel, the corner hit the door jamb and the door only opened a little.
- **Handles are separate models**, held to the panel by a `DetachableJoint`. When the handle was part of the door model, the bounding-box camera merged it into the door's box (one box per model), so handles got no box of their own.
- `grab.py` saves a frame as PNG with PIL (no NumPy), see the NumPy note below.

## Known limits / open points
- Handle boxes are small (about 8–20 px at 3 m in 640×480). Something to look at for handle detection (closer views, higher resolution, or a crop step).
- One fixed camera, one lighting setup, no glass or sliding doors, no smoke yet.
- NumPy: a pip-installed NumPy 2.x in `/usr/local/lib/python3.12/dist-packages` hides the system NumPy 1.x that ROS 2 Jazzy's OpenCV/cv_bridge were built for, so `import cv2` fails in ROS terminals. Not fixed here (system change).
