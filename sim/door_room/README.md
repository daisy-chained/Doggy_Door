# Door room v0 (draft for review)

First test room: one wall with two hinged doors, seen by a fixed sensor rig (RGB-D, 2D bounding-box camera, instance segmentation camera) at 1.2 m height, 3 m from the wall.

| | Door A (right in the image) | Door B (left in the image) |
|---|---|---|
| Handle | lever | round knob |
| Hinge | left side | right side |
| Opens | away from the camera (push) | toward the camera (pull) |

Labels (proposal, not decided): `1` door panel · `2` lever handle · `3` knob handle. Walls and floor are background.

```bash
bash sim/door_room/run_room.sh          # boxes + a frame with doors closed, then opened; output in sim/door_room/out/
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
