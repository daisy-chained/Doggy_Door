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

Objects with `class: null` (the glass pane, and in v2 all clutter) have no box of their own: the glass pane is a part of its `part_of` object (merge it into it for segmentation), clutter is background. The exporter skips class-null objects.

With clutter (`door_room_v2`), `meta.json` also has `visible_fraction` per door / handle (see below) and `clutter`: the items in the scene with position, yaw and colour.

## Config (`sim/configs/*.yaml`)
- `scenes`, `views_per_scene`, `seed`
- `doors`: chance a door is closed, minimum open fraction. The actual joint positions after settling (from each door's joint state) go in the metadata, not the commands
- `lights`: per light its base colour (from the world file) × a random intensity, plus a small tint
- `view`: distance / angle / camera height ranges in front of a random door, aim point on the door
- `reject`: try a viewpoint again when something is closer than `min_depth` (camera inside a door) or the aimed-at door isn't in view
- `glass_depth`: mix of modes, see below
- `clutter` (v2 only): `floor_count` / `wall_count` (items per scene, uniform, inclusive), `zone` (floor area), `keep_out` (floor rectangles kept clear), `gap`, `p_against_wall` (chance a floor item stands with its back to the wall, facing the camera), `wall` (x of the wall face, y / z range, `keep_out_y` spans over the doors), `camera_margin`

## Clutter and visible_fraction
Per scene `clutter.place` picks items from the pool (`clutter.py`) and places them: floor items with a random yaw, not overlapping each other or the keep-out rectangles (they may stand in front of a door and hide part of it), wall items (picture frames, whiteboard) on the wall beside the doors. Shapes get a random colour. Unused items stay parked under the floor.

- **Camera inside clutter:** Gazebo renders straight through an item when the camera is inside it, so a viewpoint within `camera_margin` of an item (footprint and height) counts as rejected and is tried again.
- **`visible_fraction`** (per object with a class, in `meta.json`): the share of the object's pixels (its id + its parts, e.g. the glass pane for door A) still visible, compared with the same view without clutter. When any clutter is in the segmentation image, the generator renders that view again with the clutter parked (after all the frame's images are taken). `null` when the object isn't in the clear view either. The 2D boxes are the visible part only.

## Glass depth step
Gazebo's depth camera sees glass as solid (it ignores transparency and visibility masks, tested). Real depth cameras mostly see through glass or give no value. For every frame with glass in view, one mode is picked from `glass_depth.mix`:
- `holes`: no depth on the glass pixels (what ClearGrasp / LIDF train on)
- `see_through`: the depth of what's behind the glass. The generator detaches the pane, moves it away, renders depth again, and puts it back at exactly its old pose
- `partial`: smooth random patches of holes / see-through / glass surface

Glass pixels come from the segmentation image: each pane is its own object with its own id.

## Lighting, door_room_v3 and scene subsets
`door_room_v3` is v2 in a closed room (`room_shell`: ceiling + walls, no sky), no ambient light, sun without shadows, and a spot light `headlight` on the camera rig (starts off). Built by `sim/door_room/make_v3.py`. Configs with a `lighting` section (`door_room_v3_*.yaml`) draw per scene a mode (`normal` / `dim` / `dark`, log-uniform room light level) and a headlight (on with a chance per mode); per view the room lights can flicker (dip). `meta.json` gets `lighting` (`mode`, `level`, `view_level`, `headlight`).

With `per_scene_rng: true` every scene has its own random streams (scene: doors, clutter, views, glass; light: lighting, lights, flicker), so scenes can be rendered alone and the lighting never changes the scene. Without it the single old stream is kept (v1/v2 reproduce).
```bash
python3 sim/doorsim/generate.py sim/configs/door_room_v3_2k.yaml --split test --light-level 0.05 --headlight 1.0 --out datasets/v3_test_light5
```
`--split test` renders only the scenes `export.py` would put in the test split (`--export-seed`, default 0) and writes them into `dataset.json` as `split_scenes`, so the export keeps them as test. `--light-level L` (+ `--headlight H`, default 0) fixes lighting for every scene (`lighting.fixed`, no flicker).

## Effects tool (smoke, depth, camera noise)
```bash
python3 sim/doorsim/effects.py datasets/door_room_v3_2k --config sim/configs/effects_train.yaml                         # random mode → datasets/<name>_fx
python3 sim/doorsim/effects.py datasets/door_room_v3_2k --config sim/configs/effects_train.yaml --visibility 2 --split test   # fixed level → datasets/<name>_vis2
```
Runs on saved frames (no Gazebo, numpy + PIL). Smoke is computed from `depth_gt.png`: transmission `t = exp(-∫β)` along each pixel's line of sight, `β = 3 / visibility` (contrast 5 % at the visibility distance), optionally denser above a height (hot layer) and patchy (seeded 3D noise). The image is `J t + A (1 − t)` in linear light, with the airlight following the scene's blurred brightness (smoke glows in the headlight cone, stays dark in darkness). The sensor `depth.png` loses pixels in thick smoke or gets false close returns. Every frame then gets camera noise (shot + read noise, auto-gain up to `max_gain`). Random mode: smoke on a share `p` of frames, visibility log-uniform. `--visibility V`: every frame at V with the same layer / patch as other levels (`inf` = noise only). Per-frame random streams are seeded from the config seed and the frame number. `rgb.png` and `depth.png` are new, `seg.png` and `depth_gt.png` are symlinks; `meta.json` gets `effects` and per-object `smoke_transmission`; `dataset.json` gets `effects` (source, config, visibility). Export as usual: `export.json` carries `effects` / `lighting` so `train/score.py` knows the level.

## Training export
```bash
python3 sim/doorsim/export.py datasets/door_room_v1_small          # → exports/door_room_v1_small/
```
YOLO (`images/`, `labels/`, `data.yaml`) + COCO (`coco/<split>.json`) from one dataset, split **by scene** 70/15/15 (seeded, `--seed`, `--ratios`), so views of one scene never sit in two splits. Images are symlinks to the dataset's `rgb.png`. All boxes kept; each COCO box carries `size_px`, `truncated` (touches the image edge) and the object's `attributes`. Gazebo 2D boxes are inclusive pixel indices (max 639 on a 640 image), so width = x_max − x_min + 1. Details in the `export.py` docstring.

## Files
- `generate.py`: the generator (CLI)
- `export.py`: YOLO + COCO training export (CLI); keeps `split_scenes` from `dataset.json` when present
- `effects.py`: smoke, depth and camera-noise effects on a saved dataset (CLI)
- `gzsim.py`: runs a headless world and drives it over gz-transport (pause/step, set pose, doors, lights, glass panes, clutter placement + colours). Every step request also says `pause: true`: without it Gazebo un-pauses the world.
- `sampling.py`: random door positions, lights, lighting modes, flicker, per-scene streams and viewpoints (balance_states / view balance: closed-open and door/side go round, v6)
- `clutter.py`: the clutter pool, its SDF, placement, camera-inside check and `visible_fraction`
- `glass.py`: the glass depth step
- `frames.py`: frame metadata (also used by `sim/door_room/meta.py`)

## Known limits
- Teleporting, not driving: no motion blur, no real robot paths yet (the robot + recordings for scoring come later).
- One room per dataset run. Speed on the laptop: about 1 s per frame including the door settling per scene (24 frames in 23 s for `door_room_v1_small`).
- Gazebo quirks handled in `gzsim.py`: a `multi_step` request never gets a reply, and `set_pose` only replies after the next step while paused. Waiting for those replies cost 5 s and 3 s per call, so both are sent with a short wait and checked by the sim clock / the next render instead.
