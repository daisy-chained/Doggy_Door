# Door rooms

Test rooms: one wall with two doors, seen by a fixed sensor rig (RGB-D, 2D + 3D bounding-box cameras, instance segmentation camera) at 1.2 m height, 3 m from the wall. Door sizes follow a common Dutch interior door (about 0.88 × 2.1 m, handle at 1.0 m).

| Room | Door A (right in the image) | Door B (left in the image) |
|---|---|---|
| `door_room_v0` | wooden, hinged, lever, hinge right, push (away from the camera) | wooden, hinged, round knob, hinge left, pull (toward the camera) |
| `door_room_v1` | **glass** (aluminium frame), hinged, lever, hinge right, push | wooden **sliding** door, flush pull, slides left |

Sides and directions are always **seen from the camera side of the wall**. v1 also has door frame trim around both openings (part of the wall = background).

```bash
bash sim/door_room/run_room.sh door_room_v1    # boxes + frame + metadata closed, then every door opened; output in sim/door_room/out/door_room_v1/
```

Open or close a door while the sim runs (hinged: radians, -1.6 = wide open; sliding: metres, 0 .. 0.9; 0 = closed):

```bash
gz topic -t /door_a/cmd_pos -m gz.msgs.Double -p 'data: -1.2'
```

Example frames: v0 [closed](example/closed.png) · [open](example/open.png) — v1 [closed](example/v1/closed.png) · [open](example/v1/open.png)

## Labels and attributes
Classes: `1` door · `2` handle. Walls and floor are background.

In Gazebo, each object gets its **own id** as its label (`11` door A, `12` handle A, `21` door B, `22` handle B; `0` = background, Gazebo allows at most 255). That way every box says which object it belongs to. `<room>.attributes.json` maps each id to its object and class, so exporting for training turns the ids back into door / handle.

Everything else is an **attribute**, not a class: handle type (`lever`, `knob`, `flush_pull`), opening (`push`, `pull`, `slide`), hinge side or slide direction, door type (`hinged`, `sliding`), glass. That way each property can get its own classifier and its own confidence, and new properties don't multiply the number of classes. Handles point to their door. Each door also has a `joint` entry: how `run_room.sh` opens it.

## Frame metadata
`run_room.sh` saves a `.json` next to each frame with, per object: the attributes, `joint_pos` (unit in `joint`) and open/closed `state`, the 2D `box`, and the 3D ground truth (`null` when out of view). The raw box output is kept as `*_boxes.txt` and `*_boxes3d.txt`.

**3D ground truth:**
- `box_3d`: centre, orientation (quaternion x y z w) and size in metres, from a 3D bounding-box camera on the rig (same ids as the 2D boxes).
- `hinge_cam` / `slide_cam` (doors): a point on the hinge axis or slide line, and its direction. These are fixed in the world (`hinge_world` / `slide_world` in the attributes file, taken from the joints), and `meta.py` puts them in the camera frame.
- Everything 3D is saved in the **ROS optical camera frame**: x right, y down, z forward, so it matches depth images and point clouds. Gazebo's own 3D boxes use a graphics frame (x right, y up, looking along -z); `meta.py` converts.
- The camera pose in the world is in the attributes file (`camera`). The rig doesn't move yet; a moving robot will need the camera pose per frame.
- Checked: each 3D centre projected into the image lands within about 2 px of its 2D box centre; open hinged doors stay 0.44 m from their hinge; the sliding door's 3D centre moves 0.85 m left.

## How it's built (and why)
- Hinged doors: the panel is a link on a **revolute joint** (the hinge) with limits, moved by a `JointPositionController`. The hinge axis sits on the panel face on the opening side; with the axis in the middle of the panel, the corner hit the door jamb.
- Sliding door: the panel hangs on the camera side of the wall on a **prismatic joint** along the wall, and is 0.98 m wide so it overlaps the 0.9 m opening.
- **Handles are separate models**, held to the panel by a `DetachableJoint`. When the handle was part of the door model, the bounding-box camera merged it into the door's box (one box per model).
- Glass: the pane is a transparent visual (`transparency 0.85`) with `visibility_flags 2`.
- `grab.py` saves a frame as PNG with PIL.

## Glass and sensors (tested 2026-10-29)
- RGB camera, 2D/3D boxes, segmentation: see the glass door as normal (boxes are there, the pane is see-through in RGB).
- A `gpu_lidar` with `visibility_mask 4294967293` **sees through** the pane (no hit), like a real lidar mostly does.
- Gazebo's **depth camera ignores visibility masks** (tried the RGB-D and a separate depth camera, high and low mask bits), so in depth the glass looks **solid**. Real depth cameras usually give holes or see through glass. Making the depth realistic needs a step after rendering; not built yet.

## Known limits / open points
- Handle boxes are small (about 5–20 px at 3 m in 640×480). Something to look at for handle detection (closer views, higher resolution, or a crop step).
- One fixed camera, one lighting setup, no smoke yet.
