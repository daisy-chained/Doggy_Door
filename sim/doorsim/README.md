# doorsim: dataset generator + helpers

Makes training data from the door rooms in `sim/door_room/`: random **scenes** (how far each door is open, lights), each seen from random **viewpoints** in front of the doors. The camera is teleported frame by frame (pause → move the rig → step one sensor period → grab), so every frame is exact and the same seed gives the same dataset.

```bash
source /opt/ros/jazzy/setup.bash        # system Python, no venv active
python3 sim/doorsim/generate.py sim/configs/door_room_v1_small.yaml            # → datasets/door_room_v1_small/
python3 -m pytest sim/tests                                                     # unit tests, no Gazebo needed
```

## Output
One folder per dataset (`datasets/` is gitignored):

| File | What |
|---|---|
| `dataset.json` | config, seed, world, git commit (+ dirty flag), camera intrinsics, the room's attributes, frame/reject counts |
| `frames/NNNNNN/rgb.png` | colour image |
| `frames/NNNNNN/depth.png` | **sensor-like** depth: uint16 millimetres, `0` = no value. Glass handled by the glass step below |
| `frames/NNNNNN/depth_gt.png` | depth as rendered (glass solid), same format |
| `frames/NNNNNN/seg.png` | object id per pixel (uint8, `0` = background) |
| `frames/NNNNNN/meta.json` | frame metadata (as in the door room README: attributes, 2D/3D boxes, axes in the camera frame) + **camera pose in the world**, scene/view number, lights, `glass_depth_mode` |

Exporting to YOLO/COCO is a separate step (not built yet). Objects with `class: null` (the glass pane) are parts of their `part_of` object: merge them into it for segmentation, no box of their own.

## Config (`sim/configs/*.yaml`)
- `scenes`, `views_per_scene`, `seed`
- `doors`: chance a door is closed, minimum open fraction. The actual joint positions after settling (from each door's joint state) go in the metadata, not the commands
- `lights`: per light its base colour (from the world file) × a random intensity, plus a small tint
- `view`: distance / angle / camera height ranges in front of a random door, aim point on the door
- `reject`: try a viewpoint again when something is closer than `min_depth` (camera inside a door) or the aimed-at door isn't in view
- `glass_depth`: mix of modes, see below

## Glass depth step
Gazebo's depth camera sees glass as solid (it ignores transparency and visibility masks, tested). Real depth cameras mostly see through glass or give no value. For every frame with glass in view, one mode is picked from `glass_depth.mix`:
- `holes`: no depth on the glass pixels (what ClearGrasp / LIDF train on)
- `see_through`: the depth of what's behind the glass. The generator detaches the pane, moves it away, renders depth again, and puts it back at exactly its old pose
- `partial`: smooth random patches of holes / see-through / glass surface

Glass pixels come from the segmentation image: each pane is its own object with its own id.

## Training export
```bash
python3 sim/doorsim/export.py datasets/door_room_v1_small          # → exports/door_room_v1_small/
```
YOLO (`images/`, `labels/`, `data.yaml`) + COCO (`coco/<split>.json`) from one dataset, split **by scene** 70/15/15 (seeded, `--seed`, `--ratios`), so views of one scene never sit in two splits. Images are symlinks to the dataset's `rgb.png`. All boxes kept; each COCO box carries `size_px`, `truncated` (touches the image edge) and the object's `attributes`. Gazebo 2D boxes are inclusive pixel indices (max 639 on a 640 image), so width = x_max − x_min + 1. Details in the `export.py` docstring.

## Files
- `generate.py`: the generator (CLI)
- `export.py`: YOLO + COCO training export (CLI)
- `gzsim.py`: runs a headless world and drives it over gz-transport (pause/step, set pose, doors, lights, glass panes). Every step request also says `pause: true`: without it Gazebo un-pauses the world.
- `sampling.py`: random door positions, lights and viewpoints
- `glass.py`: the glass depth step
- `frames.py`: frame metadata (also used by `sim/door_room/meta.py`)

## Known limits
- Teleporting, not driving: no motion blur, no real robot paths yet (the robot + recordings for scoring come later).
- One room per dataset run. Speed on the laptop: about 1 s per frame including the door settling per scene (24 frames in 23 s for `door_room_v1_small`).
- Gazebo quirks handled in `gzsim.py`: a `multi_step` request never gets a reply, and `set_pose` only replies after the next step while paused. Waiting for those replies cost 5 s and 3 s per call, so both are sent with a short wait and checked by the sim clock / the next render instead.
