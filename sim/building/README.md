# Buildings (robot run, phase 2e)

One floor: a corridor with rooms on both sides, doors from the door-room v5 sampler, written as a Gazebo world plus a Nav2 map.
Code: `sim/doorsim/building.py`, CLI: `sim/building/make_buildings.py`, numbers: `sim/configs/buildings_v1.yaml`, tests: `sim/tests/test_building.py`.

Run (building i uses seed `seed + i`, named `bld_<split>_<seed:03d>`):

    python3 sim/building/make_buildings.py sim/configs/buildings_v1.yaml [--out DIR]   # default DIR = sim/building/out (gitignored)

Files per building: `<name>.sdf` (world), `<name>.building.json` (seed, split, layout, doors with their label ids, approach poses),
`<name>.map.pgm` + `<name>.map.yaml` (Nav2 map_server, 0 = occupied, 254 = free, row 0 = north).

Conventions
- World frame: x east, y north, z up, metres. Spaces: `corridor`, `r0`, `r1`, ... Rects `[x0, y0, x1, y1]` run from wall centre to wall centre; walls are 0.1 thick.
- Doors are built as in v5 (`doors.py`) in a local frame and moved into the building (rotate by `yaw`, translate to `pos`). Local +x = the normal from
  `connects[0]` (front) into `connects[1]` (back). Room <-> corridor doors: corridor is front. Neighbour doors: west room front, east room back.
- Model names: `door_<id>_panel`, `_handle_front`, `_handle_back`, `_glass`, `_hinge_<0..2>`, `door_<id>_frame` (trim / stop / rail); cmd topic `/door_<id>_panel/cmd_pos`.
- Label ids: base = 10 * door id; +1 panel, +2 handle front, +3 glass, +4 handle back, +5..7 hinge knuckles. At most 19 doors (ids < 200).
- Clutter (config section `clutter:`, optional; `building.json` key `clutter`): items from `doorsim/clutter.py` POOL, label = 200 + (pool id - 100). Floor items
  keep off the doors' approach strips and are occupied in the map (an item that would cut apart the clear approach poses of a space is dropped); wall items hang flat
  on a wall face. Static models `clutter_<n>` with visuals + collisions. Drawn after the doors, so doors / layout are the same with or without clutter.
- Open doors: the joint controller starts at `open_frac` * the v5 open value (`<initial_position>`); closed = 0.
