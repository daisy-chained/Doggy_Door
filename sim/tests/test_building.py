# Unit tests for the building generator (phase 2e, Madelief 2026-11-07 22:20), no Gazebo needed: python3 -m pytest sim/tests/test_building.py
# Her design: one floor, corridor + rooms; doors from the v5 sampler (held-out combos only in test buildings, open amount per door per
# building); approach poses on BOTH sides of every door; the Nav2 map is written by the generator (SLAM comes later).
# Frames: world x east, y north, z up, metres. A "space" is the corridor ("corridor") or a room ("r0", "r1", ...); rects are [x0, y0, x1, y1].
#
#   building.sample_layout(rng, cfg) -> {"bounds": [x0, y0, x1, y1], "corridor": [x0, y0, x1, y1], "rooms": [{"id", "rect", "side"}]}
#     corridor along x with rooms on its south ("south", below) and north ("north", above) side, cut at random widths
#     (cfg["room_width"]) and a random depth per side (cfg["room_depth"]); rooms of one side share the corridor wall, tile the
#     corridor length without overlap and touch the corridor. bounds = outer box of all spaces. Walls are WALL_T thick, centred on
#     the shared edges (a space's rect is wall centre to wall centre).
#   building.place_doors(rng, layout, cfg, split="train") -> [door]: every room gets one door to the corridor (a second one with
#     cfg["p_second_door"]), and with cfg["p_neighbour_door"] a door to its east neighbour on the same side. door = {"id" (1..),
#     "spec" (doors.sample_door spec + door_stop, slide_y), "connects": [front space, back space], "pos": [x, y] (door centre on the wall
#     centre line), "yaw" (rad: local +x = normal from the front space into the back space), "state": "closed" | "open",
#     "open_frac" (0 when closed, else in [cfg["min_open"], 1]), "heldout": bool}. A door sits on the wall the two spaces share,
#     its opening + trim at least cfg["corner_margin"] from that wall's ends, no two doors' openings + trim (+ slide path) overlap.
#     Held-out combos (cfg["heldout"]) only when split == "test". At most MAX_DOORS doors (label ids must fit in uint8).
#   building.approach_poses(layout, doors, cfg) -> [pose]: for every door, on both sides, one pose per distance in cfg["approach"]
#     ["distances"] (clamped so the robot (cfg["robot_radius"]) stays inside that space, away from its walls); pose = {"door", "space",
#     "side" ("front" | "back"), "dist", "x", "y", "yaw" (facing the door centre), "reachable"}. reachable = the space can be reached
#     from the corridor through open doors only, each with door_gap(d) >= cfg["min_door_gap"] (missing key = 0: any open door; Claude
#     2026-11-11, research-based, from the tours: no room stop worked with a gap < 0.45 m, ~1/3 at 0.55-0.6, ~1/2 at 0.6-0.65, 81/85
#     above 0.65 -> 0.6 m = 2 x robot_radius, the robot's Nav2 size).
#   building.hinge_xy(d) -> (x, y) world point of a hinged door's hinge axis (doors._layout "hinge", via _world); None for sliding.
#   building.open_position(d) -> joint value of the door fully open as the sim opens it: hinged 1.2 * swing_x * hinge_y (rad), sliding
#     0.9 * travel = round(0.9 * (width + 0.15), 3) (m), i.e. doors._layout(spec, SLOT)["open"]. (Reactive door step, 2026-11-11.)
#   building.door_topic(d) -> gz topic of the door's joint position controller in the building world: "/door_<id>_panel/cmd_pos".
#   building.after_opening(b, opened, cfg) -> new building dict (b unchanged): doors whose id is in `opened` get state "open",
#     open_frac 1.0 (fully open, as the door step opens them); poses = approach_poses(layout, new doors, cfg, clutter) (so rooms behind
#     opened doors become reachable). Round 2 of the tour (Madelief 2026-11-11: two rounds).
#   building.write_map(b, stem) -> writes stem.map.pgm + stem.map.yaml (image: the pgm's file name) exactly as write() does.
#   building.door_gap(d) -> clear width (m) the robot can pass with the panel where it stands; 0 when closed. Hinged (turned th = 1.2 *
#     open_frac rad about a hinge at one jamb, swung out of the wall): width * sin(th) = distance from the far jamb to the panel. Sliding
#     (panel width + 0.08, shifted 0.9 * (width + 0.15) * open_frac): shift - 0.04, clamped to [0, width].
#   building.occupancy(layout, doors, res) -> (grid uint8 H x W, origin [x, y]): 0 = occupied, 254 = free, row 0 = north edge (map
#     image convention, like map_server), origin = world xy of the bottom-left cell. Walls occupied (with their thickness), the doorway of
#     a closed door occupied, of a door with open_frac 1 free.
#   building.building_sdf(b, name) -> SDF text: valid XML, <world name=name>, walls + floor + ceiling + one ceiling light per space,
#     every door's models (panel, glass, handles, hinges as in v5) placed at its pos / yaw, each with a unique Label id < 256;
#     the v3 camera rig is kept (to render frames at approach poses).
#   building.make_building(cfg, seed, split="train") -> {"seed", "split", "layout", "doors", "poses"}; same seed -> same building.
#   building.write(b, out_dir, name) -> writes name.sdf, name.building.json (b + label ids per door), name.map.pgm + name.map.yaml
#     (Nav2 map_server: image, resolution, origin [x, y, 0], negate 0, occupied_thresh 0.65, free_thresh 0.25).
#   sim/building/make_buildings.py <config.yaml> [--out DIR]: count buildings (seeds seed .. seed + count - 1), split per config.
import json, math, pathlib, sys, xml.etree.ElementTree as ET
import numpy as np
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import building

V5 = {"p_sliding": 0.35, "p_glass": 0.35, "handle": {"lever": 0.5, "knob": 0.5}, "width": [0.8, 1.0], "height": [2.0, 2.3],
      "frame_style": {"none": 0.33, "thin": 0.34, "thick": 0.33}, "p_hangs_back": 0.5,
      "colours": {"wood": {"p": 1.0, "rgb": [[0.45, 0.28, 0.14], [0.85, 0.7, 0.5]]}},
      "glass_frame": {"black": {"p": 1.0, "rgb": [[0.05, 0.05, 0.05], [0.15, 0.15, 0.15]]}},
      "trim": {"white": {"p": 0.5, "rgb": [[0.9, 0.9, 0.88], [0.97, 0.97, 0.95]]}, "match": {"p": 0.5}}}
CFG = {"corridor_length": [12.0, 16.0], "corridor_width": [1.8, 2.4], "room_width": [2.6, 4.5], "room_depth": [3.0, 4.5],
       "p_second_door": 0.25, "p_neighbour_door": 0.3, "corner_margin": 0.3, "p_closed": 0.4, "min_open": 0.3,
       "door": V5, "door_stop": True, "heldout": [{"glass": True, "handle_type": "knob"}, {"door_type": "sliding", "glass": True}],
       "approach": {"distances": [2.5, 1.0]}, "robot_radius": 0.3, "map_resolution": 0.05, "min_door_gap": 0.6}
SEEDS = range(12)


def _inside(rect, x, y, margin=0.0):
    return rect[0] + margin - 1e-6 <= x <= rect[2] - margin + 1e-6 and rect[1] + margin - 1e-6 <= y <= rect[3] - margin + 1e-6


def _spaces(layout):
    return {"corridor": layout["corridor"], **{r["id"]: r["rect"] for r in layout["rooms"]}}


def _overlap(a, b):
    return min(a[2], b[2]) - max(a[0], b[0]) > 1e-6 and min(a[3], b[3]) - max(a[1], b[1]) > 1e-6


@pytest.mark.parametrize("seed", SEEDS)
def test_layout_rooms_tile_both_sides(seed):
    lay = building.sample_layout(np.random.default_rng(seed), CFG)
    c = lay["corridor"]
    assert CFG["corridor_length"][0] - 1e-6 <= c[2] - c[0] <= CFG["corridor_length"][1] + 1e-6
    assert CFG["corridor_width"][0] - 1e-6 <= c[3] - c[1] <= CFG["corridor_width"][1] + 1e-6
    rooms = lay["rooms"]
    assert len({r["id"] for r in rooms}) == len(rooms) >= 4
    for side, edge in (("south", c[1]), ("north", c[3])):
        rs = sorted((r["rect"] for r in rooms if r["side"] == side), key=lambda r: r[0])
        assert rs and rs[0][0] == pytest.approx(c[0]) and rs[-1][2] == pytest.approx(c[2])
        for a, b in zip(rs, rs[1:]):
            assert a[2] == pytest.approx(b[0])                                   # no gap, no overlap
        for r in rs:
            assert (r[3] if side == "south" else r[1]) == pytest.approx(edge)    # touches the corridor
            assert CFG["room_depth"][0] - 1e-6 <= r[3] - r[1] <= CFG["room_depth"][1] + 1e-6
    rects = list(_spaces(lay).values())
    assert not any(_overlap(a, b) for i, a in enumerate(rects) for b in rects[i + 1:])
    bx = lay["bounds"]
    assert all(_inside(bx, r[0], r[1]) and _inside(bx, r[2], r[3]) for r in rects)


def _shared_wall(ra, rb):
    """(axis, coord, lo, hi) of the wall two touching rects share: axis "x" = wall along x at y = coord."""
    if abs(ra[3] - rb[1]) < 1e-6 or abs(rb[3] - ra[1]) < 1e-6:
        y = ra[3] if abs(ra[3] - rb[1]) < 1e-6 else ra[1]
        return "x", y, max(ra[0], rb[0]), min(ra[2], rb[2])
    x = ra[2] if abs(ra[2] - rb[0]) < 1e-6 else ra[0]
    return "y", x, max(ra[1], rb[1]), min(ra[3], rb[3])


@pytest.mark.parametrize("seed", SEEDS)
def test_doors_on_shared_walls(seed):
    rng = np.random.default_rng(seed)
    lay = building.sample_layout(rng, CFG)
    doors = building.place_doors(rng, lay, CFG)
    sp = _spaces(lay)
    assert 1 <= len(doors) <= building.MAX_DOORS
    assert [d["id"] for d in doors] == list(range(1, len(doors) + 1))
    for r in lay["rooms"]:
        assert any(r["id"] in d["connects"] and "corridor" in d["connects"] for d in doors)
    spans = []
    for d in doors:
        a, b = d["connects"]
        axis, coord, lo, hi = _shared_wall(sp[a], sp[b])
        x, y = d["pos"]
        along, across = (x, y) if axis == "x" else (y, x)
        assert across == pytest.approx(coord)
        half = d["spec"]["width"] / 2 + building.doors.TRIM[d["spec"]["frame_style"]][0]
        assert lo + CFG["corner_margin"] - 1e-6 <= along - half and along + half <= hi - CFG["corner_margin"] + 1e-6
        # local +x points from the front space into the back space
        nx, ny = math.cos(d["yaw"]), math.sin(d["yaw"])
        ca, cb = [((s[0] + s[2]) / 2, (s[1] + s[3]) / 2) for s in (sp[a], sp[b])]
        assert (cb[0] - ca[0]) * nx + (cb[1] - ca[1]) * ny > 0
        assert abs(nx * (1 if axis == "x" else 0) + ny * (1 if axis == "y" else 0)) < 1e-6   # normal across the wall
        assert d["state"] in ("closed", "open")
        assert d["open_frac"] == 0 if d["state"] == "closed" else CFG["min_open"] - 1e-9 <= d["open_frac"] <= 1
        spans.append((axis, round(coord, 6), along - half, along + half))
    for i, s in enumerate(spans):
        for t in spans[i + 1:]:
            if s[:2] == t[:2]:
                assert s[3] <= t[2] + 1e-6 or t[3] <= s[2] + 1e-6


def test_heldout_only_in_test():
    seen = {"train": 0, "test": 0}
    for split in seen:
        for seed in range(30):
            b = building.make_building(CFG, seed, split)
            seen[split] += sum(d["heldout"] for d in b["doors"])
            for d in b["doors"]:
                assert d["heldout"] == building.doors.is_heldout(d["spec"], CFG["heldout"])
    assert seen["train"] == 0 and seen["test"] > 0


def test_same_seed_same_building():
    a, b = building.make_building(CFG, 7), building.make_building(CFG, 7)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert json.dumps(a, sort_keys=True) != json.dumps(building.make_building(CFG, 8), sort_keys=True)


@pytest.mark.parametrize("seed", SEEDS)
def test_approach_poses_both_sides(seed):
    b = building.make_building(CFG, seed)
    sp, poses = _spaces(b["layout"]), b["poses"]
    for d in b["doors"]:
        mine = [p for p in poses if p["door"] == d["id"]]
        assert {p["side"] for p in mine} == {"front", "back"}
        assert len(mine) == 2 * len(CFG["approach"]["distances"])
        for p in mine:
            assert p["space"] == d["connects"][0 if p["side"] == "front" else 1]
            assert _inside(sp[p["space"]], p["x"], p["y"], CFG["robot_radius"] + building.WALL_T / 2)
            to_door = math.atan2(d["pos"][1] - p["y"], d["pos"][0] - p["x"])
            assert math.cos(p["yaw"] - to_door) == pytest.approx(1.0, abs=1e-6)
            assert 0 < p["dist"] <= max(CFG["approach"]["distances"]) + 1e-6
    # reachable = through open doors from the corridor
    reach, grow = {"corridor"}, True
    while grow:
        grow = False
        for d in b["doors"]:
            a, c = d["connects"]
            if d["state"] == "open" and building.door_gap(d) >= CFG["min_door_gap"] and (a in reach) != (c in reach):
                reach |= {a, c}; grow = True
    for p in poses:
        assert p["reachable"] == (p["space"] in reach)


def _cell(grid, origin, res, x, y):
    col, row = int((x - origin[0]) / res), grid.shape[0] - 1 - int((y - origin[1]) / res)
    return grid[row, col]


@pytest.mark.parametrize("seed", range(6))
def test_occupancy_map(seed):
    b = building.make_building(CFG, seed)
    lay, doors, res = b["layout"], b["doors"], CFG["map_resolution"]
    grid, origin = building.occupancy(lay, doors, res)
    assert grid.dtype == np.uint8 and set(np.unique(grid)) <= {0, 254}
    bx = lay["bounds"]
    assert grid.shape[1] * res >= bx[2] - bx[0] and grid.shape[0] * res >= bx[3] - bx[1]
    c = lay["corridor"]
    assert _cell(grid, origin, res, (c[0] + c[2]) / 2, (c[1] + c[3]) / 2) == 254     # corridor centre free
    wall_x = [x for x in np.linspace(c[0] + 0.3, c[2] - 0.3, 60)                    # corridor north wall, away from doorways
              if not any(abs(d["pos"][1] - c[3]) < 1e-6 and abs(d["pos"][0] - x) < d["spec"]["width"] / 2 + 0.3 for d in doors)]
    assert wall_x and all(_cell(grid, origin, res, x, c[3]) == 0 for x in wall_x)
    for d in doors:
        v = _cell(grid, origin, res, *d["pos"])
        if d["state"] == "closed":
            assert v == 0
    opened = dict(CFG, p_closed=0.0, min_open=1.0)
    b2 = building.make_building(opened, seed)
    g2, o2 = building.occupancy(b2["layout"], b2["doors"], res)
    for d in b2["doors"]:
        assert _cell(g2, o2, res, *d["pos"]) == 254


def test_sdf_valid_and_labels_unique():
    b = building.make_building(CFG, 3, "test")
    text = building.building_sdf(b, "bld_test_003")
    root = ET.fromstring(text)
    world = root.find("world")
    assert world.get("name") == "bld_test_003"
    labels = [int(l.text) for l in root.iter("label")]
    assert len(labels) == len(set(labels)) and max(labels) < 256
    names = {m.get("name") for m in world.findall("model")}
    for d in b["doors"]:
        assert any(n.startswith(f"door_{d['id']}_") or n == f"door_{d['id']}" for n in names)
    assert len(world.findall("light")) >= 1 + len(b["layout"]["rooms"])
    assert any("rig" in (m.get("name") or "") or "camera" in (m.get("name") or "") for m in world.findall("model"))


def test_write_and_cli(tmp_path, monkeypatch):
    b = building.make_building(CFG, 1)
    building.write(b, tmp_path, "bld_001")
    for ext in (".sdf", ".building.json", ".map.pgm", ".map.yaml"):
        assert (tmp_path / f"bld_001{ext}").exists()
    meta = json.loads((tmp_path / "bld_001.building.json").read_text())
    assert {"seed", "split", "layout", "doors", "poses"} <= set(meta)
    assert all("labels" in d for d in meta["doors"])
    y = (tmp_path / "bld_001.map.yaml").read_text()
    assert "bld_001.map.pgm" in y and "resolution: 0.05" in y
    import yaml
    cfg = dict(CFG, seed=10, count=2, split="val")
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    sys.path.insert(0, str(SIM / "building"))
    import make_buildings
    monkeypatch.setattr(sys, "argv", ["make_buildings.py", str(tmp_path / "c.yaml"), "--out", str(tmp_path / "out")])
    make_buildings.main()
    assert sorted(p.name for p in (tmp_path / "out").glob("*.sdf")) == ["bld_val_010.sdf", "bld_val_011.sdf"]


# Added 2026-11-07 22:50 (Claude, after looking at the first maps): an approach pose can land on an open door panel (2-3 per building).
#   pose["clear"]: no occupied map cell within robot_radius. A blocked pose is first moved further back along the door normal (0.1 m steps,
#   staying inside the space), then closer (down to 0.5 m); "dist" is the distance actually used. clear False only if nothing fits.
@pytest.mark.parametrize("seed", SEEDS)
def test_approach_poses_clear_of_obstacles(seed):
    b = building.make_building(CFG, seed)
    res, rr = CFG["map_resolution"], CFG["robot_radius"]
    grid, origin = building.occupancy(b["layout"], b["doors"], res)
    r = int(math.ceil(rr / res))
    blocked = 0
    for p in b["poses"]:
        assert "clear" in p
        c, row = int((p["x"] - origin[0]) / res), grid.shape[0] - 1 - int((p["y"] - origin[1]) / res)
        hit = (grid[max(0, row - r):row + r + 1, max(0, c - r):c + r + 1] == 0).any()
        assert hit == (not p["clear"])
        blocked += hit
    assert blocked <= len(b["poses"]) // 10


# Added 2026-11-08 (Claude, research-based detail of her generator design; logged in the project note): clutter back in the buildings.
#   cfg["clutter"] (optional; absent -> b["clutter"] == [] and the building is exactly as before, the clutter draws come after all other
#   draws): {"rooms": {"floor_count": [lo, hi], "wall_count": [lo, hi]}, "corridor": {same}, "gap": m between floor items,
#   "p_against_wall": chance a floor item stands with its back to a wall of its space, "wall_z": [lo, hi] height range of wall items,
#   "door_margin": m that wall items stay off a door's opening + trim (+ slide path) along the wall}.
#   Items come from clutter.POOL (floor items on the floor, wall items on a wall face); one item at most once per space, reused across
#   spaces. b["clutter"] = [{"name": "clutter_<n>" (n = 0..), "item": pool name, "label": 200 + (pool id - 100), "space", "place":
#   "floor" | "wall", "position": [x, y, z] (centre of the item's box), "yaw" (item front = local -x, as in clutter.py), "colour": rgb or
#   None (shape items get a random colour)}]. Label ids: doors stay below 200 (MAX_DOORS = 19 -> 10 * 19 + 7 = 197), clutter 200..247.
#   Floor items: footprint (clutter.footprint) inside the space (rect shrunk by WALL_T / 2), no two closer than gap, and off every door's
#   keep-out strip on that side: along the wall = the door's opening + trim (+ slide path) grown by robot_radius each side, into the space
#   = max(approach distances) + robot_radius from the wall centre line (so the robot's approach line and a hinged panel's swing stay free).
#   An item that would cut a space's clear approach poses apart (free map grown by robot_radius) is dropped. Floor items are occupied in
#   the map: occupancy(layout, doors, res, clutter=None) marks their footprints; approach poses are computed with the clutter in.
#   SDF: one static model per item, named as in b["clutter"], with visuals + collisions (so the robot can't drive through) and its label.
CCFG = dict(CFG, clutter={"rooms": {"floor_count": [1, 4], "wall_count": [0, 2]}, "corridor": {"floor_count": [0, 2], "wall_count": [0, 3]},
                          "gap": 0.05, "p_against_wall": 0.6, "wall_z": [0.9, 2.0], "door_margin": 0.2})


def _pool():
    from doorsim import clutter
    return {p["name"]: p for p in clutter.POOL}


def _aabb(corners):
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    return [min(xs), min(ys), max(xs), max(ys)]


def _door_span(d, margin):
    """(x0, y0, x1, y1) of the door's opening + trim (+ slide path) along its wall, grown by margin, as a thin box on the wall line."""
    lo, hi = building._extent(d["spec"], 1)
    lo, hi = min(lo, -hi), max(hi, -lo)  # either direction of the local y axis
    if abs(math.cos(d["yaw"])) > 0.5:     # wall along y
        return d["pos"][1] + lo - margin, d["pos"][1] + hi + margin, "y"
    return d["pos"][0] + lo - margin, d["pos"][0] + hi + margin, "x"


def _keep_out(d, space_rect, cfg):
    """Keep-out strip of door d inside the space (rect [x0, y0, x1, y1]), conservative (both slide directions)."""
    lo, hi, along = _door_span(d, cfg["robot_radius"])
    deep = max(cfg["approach"]["distances"]) + cfg["robot_radius"]
    r = space_rect
    if along == "x":
        y = d["pos"][1]
        return [lo, y, hi, y + deep] if r[1] >= y - 1e-6 else [lo, y - deep, hi, y]
    x = d["pos"][0]
    return [x, lo, x + deep, hi] if r[0] >= x - 1e-6 else [x - deep, lo, x, hi]


def test_no_clutter_without_config():
    b = building.make_building(CFG, 2)
    assert b["clutter"] == []
    assert "clutter_" not in building.building_sdf(b, "x")


def test_door_ids_stay_below_clutter_ids():
    assert building.MAX_DOORS == 19 and 10 * building.MAX_DOORS + 7 < 200
    many = dict(CFG, corridor_length=[40.0, 40.0], room_width=[2.6, 2.6], p_second_door=1.0, p_neighbour_door=1.0)
    with pytest.raises(ValueError):  # more rooms than MAX_DOORS: some room gets no door
        building.make_building(many, 0)


@pytest.mark.parametrize("seed", range(8))
def test_clutter_placement(seed):
    b = building.make_building(CCFG, seed)
    pool, sp, cl = _pool(), _spaces(b["layout"]), b["clutter"]
    assert cl and [c["name"] for c in cl] == [f"clutter_{i}" for i in range(len(cl))]
    per = {}
    for c in cl:
        p = pool[c["item"]]
        assert c["label"] == 200 + p["id"] - 100 and 200 <= c["label"] < 256
        assert c["place"] == p["place"] and c["space"] in sp
        assert (c["colour"] is not None) == p["recolour"]
        per.setdefault((c["space"], c["place"]), []).append(c)
    for (space, place), cs in per.items():
        assert len({c["item"] for c in cs}) == len(cs)                       # one item at most once per space
        key = "corridor" if space == "corridor" else "rooms"
        assert len(cs) <= CCFG["clutter"][key][f"{place}_count"][1]
    for c in cl:
        p, r = pool[c["item"]], sp[c["space"]]
        if c["place"] == "floor":
            assert c["position"][2] == pytest.approx(p["size"][2] / 2)
            bb = _aabb(building.clutter.footprint({"position": c["position"], "yaw": c["yaw"]}, p))
            e = building.WALL_T / 2
            assert bb[0] >= r[0] + e - 1e-6 and bb[2] <= r[2] - e + 1e-6 and bb[1] >= r[1] + e - 1e-6 and bb[3] <= r[3] - e + 1e-6
            for d in b["doors"]:
                if c["space"] in d["connects"]:
                    assert not _overlap(bb, _keep_out(d, r, CCFG)), (c, d["id"])
        else:
            z0, z1 = CCFG["clutter"]["wall_z"]
            assert z0 - 1e-6 <= c["position"][2] - p["size"][2] / 2 and c["position"][2] + p["size"][2] / 2 <= z1 + 1e-6
            off = building.WALL_T / 2 + p["size"][0] / 2
            x, y = c["position"][:2]
            on = [w for w, v in (("x0", x - r[0]), ("x1", r[2] - x), ("y0", y - r[1]), ("y1", r[3] - y)) if abs(v - off) < 1e-6]
            assert on, c                                                     # hangs flat on a wall face of its space
            half = p["size"][1] / 2
            for d in b["doors"]:
                lo, hi, along = _door_span(d, CCFG["clutter"]["door_margin"])
                wall_line = d["pos"][1] if along == "x" else d["pos"][0]
                if abs((y if along == "x" else x) - wall_line) < off + 1e-6:
                    a = x if along == "x" else y
                    assert a + half <= lo + 1e-6 or a - half >= hi - 1e-6, (c, d["id"])
    floor = [c for c in cl if c["place"] == "floor"]
    for i, a in enumerate(floor):
        for c in floor[i + 1:]:
            if a["space"] == c["space"]:
                ba = _aabb(building.clutter.footprint({"position": a["position"], "yaw": a["yaw"]}, pool[a["item"]]))
                bc = _aabb(building.clutter.footprint({"position": c["position"], "yaw": c["yaw"]}, pool[c["item"]]))
                g = CCFG["clutter"]["gap"] - 1e-6
                assert not _overlap([ba[0] - g, ba[1] - g, ba[2] + g, ba[3] + g], bc)


def test_clutter_same_seed_and_doors_unchanged():
    a, b = building.make_building(CCFG, 4), building.make_building(CCFG, 4)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    plain = building.make_building(CFG, 4)
    assert json.dumps(plain["doors"], sort_keys=True) == json.dumps(a["doors"], sort_keys=True)
    assert plain["layout"] == a["layout"]


@pytest.mark.parametrize("seed", range(6))
def test_clutter_in_map_and_poses_connected(seed):
    b = building.make_building(CCFG, seed)
    res, rr, pool = CCFG["map_resolution"], CCFG["robot_radius"], _pool()
    grid, origin = building.occupancy(b["layout"], b["doors"], res, clutter=b["clutter"])
    for c in b["clutter"]:
        if c["place"] == "floor":
            assert _cell(grid, origin, res, *c["position"][:2]) == 0
    plain, _ = building.occupancy(b["layout"], b["doors"], res)
    assert (grid == 0).sum() > (plain == 0).sum()
    r = int(math.ceil(rr / res))
    for p in b["poses"]:                                                     # poses see the clutter
        c, row = int((p["x"] - origin[0]) / res), grid.shape[0] - 1 - int((p["y"] - origin[1]) / res)
        assert (grid[max(0, row - r):row + r + 1, max(0, c - r):c + r + 1] == 0).any() == (not p["clear"])
    # clutter never cuts apart clear poses of one space that are connected without it (free map grown by robot_radius; an open door
    # swung into a narrow corridor can already block it on its own, that is left as is)
    from scipy import ndimage

    def comps(g):
        occ, fits = g == 0, np.ones(g.shape, bool)
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                fits &= ~np.roll(np.roll(occ, dr, 0), dc, 1)
        return ndimage.label(fits)[0]

    lab, lab0 = comps(grid), comps(plain)
    cell = lambda p: (grid.shape[0] - 1 - int((p["y"] - origin[1]) / res), int((p["x"] - origin[0]) / res))
    clear = [p for p in b["poses"] if p["clear"]]
    for i, p in enumerate(clear):
        for q in clear[i + 1:]:
            if p["space"] == q["space"] and lab0[cell(p)] == lab0[cell(q)] != 0:
                assert lab[cell(p)] == lab[cell(q)], (p, q)

def test_clutter_sdf_and_json(tmp_path):
    b = building.make_building(CCFG, 5)
    root = ET.fromstring(building.building_sdf(b, "bld_c"))
    models = {m.get("name"): m for m in root.find("world").findall("model")}
    door_labels = [int(l.text) for n, m in models.items() if n.startswith("door_") for l in m.iter("label")]
    assert door_labels and len(door_labels) == len(set(door_labels)) and max(door_labels) < 200
    for c in b["clutter"]:
        m = models[c["name"]]
        assert m.find("static").text.strip() == "true"
        assert int(m.find(".//label").text) == c["label"]
        assert m.findall(".//collision") and m.findall(".//visual")
        pose = [float(v) for v in m.find("pose").text.split()]
        assert pose[:3] == pytest.approx(c["position"], abs=1e-4) and math.cos(pose[5] - c["yaw"]) == pytest.approx(1.0, abs=1e-6)
    building.write(b, tmp_path, "bld_c")
    meta = json.loads((tmp_path / "bld_c.building.json").read_text())
    assert meta["clutter"] == json.loads(json.dumps(b["clutter"]))
    grid = (tmp_path / "bld_c.map.pgm").read_bytes()
    g, o = building.occupancy(b["layout"], b["doors"], CCFG["map_resolution"], clutter=b["clutter"])
    assert grid.endswith(g.tobytes())


def _door(door_type, width, state, frac):
    spec = {"door_type": door_type, "width": width}
    return {"spec": spec, "state": state, "open_frac": frac if state == "open" else 0.0}


@pytest.mark.parametrize("w,f", [(0.8, 0.3), (0.9, 0.36), (0.9, 0.6), (1.0, 1.0)])
def test_door_gap_hinged(w, f):
    assert building.door_gap(_door("hinged", w, "open", f)) == pytest.approx(w * math.sin(1.2 * f), abs=1e-6)
    assert building.door_gap(_door("hinged", w, "closed", 0.0)) == 0.0


@pytest.mark.parametrize("w,f,want", [(0.932, 0.867, 0.9 * 1.082 * 0.867 - 0.04), (0.8, 0.3, 0.9 * 0.95 * 0.3 - 0.04),
                                       (0.9, 1.0, 0.9), (0.8, 0.04, 0.0)])
def test_door_gap_sliding(w, f, want):
    assert building.door_gap(_door("sliding", w, "open", f)) == pytest.approx(want, abs=1e-6)
    assert building.door_gap(_door("sliding", w, "closed", 0.0)) == 0.0


def test_door_gap_small_hinged_too_narrow():
    # 007 door 1: a hinged door 36 % open leaves ~0.38 m, too narrow for the robot (failed in every tour)
    assert building.door_gap(_door("hinged", 0.9, "open", 0.364)) < 0.45


def test_narrow_open_doors_not_reachable():
    # over the seeds some open door is too narrow, and a room whose only doors are closed or too narrow is not reachable
    narrow = 0
    for seed in SEEDS:
        b = building.make_building(CFG, seed)
        for d in b["doors"]:
            if d["state"] == "open" and building.door_gap(d) < CFG["min_door_gap"]:
                narrow += 1
                ways = [e for e in b["doors"] if d["connects"][1] in e["connects"]]
                if all(e["state"] != "open" or building.door_gap(e) < CFG["min_door_gap"] for e in ways):
                    assert not any(p["reachable"] for p in b["poses"] if p["space"] == d["connects"][1])
    assert narrow > 0


def test_min_door_gap_missing_means_any_open_door():
    cfg = {k: v for k, v in CFG.items() if k != "min_door_gap"}
    for seed in SEEDS:
        b = building.make_building(cfg, seed)
        reach, grow = {"corridor"}, True
        while grow:
            grow = False
            for d in b["doors"]:
                a, c = d["connects"]
                if d["state"] == "open" and (a in reach) != (c in reach):
                    reach |= {a, c}; grow = True
        assert all(p["reachable"] == (p["space"] in reach) for p in b["poses"])


@pytest.mark.parametrize("seed", SEEDS[:4])
def test_hinge_xy_open_position_topic(seed):
    b = building.make_building(CFG, seed)
    sdf = building.building_sdf(b, "t")
    for d in b["doors"]:
        assert f"<topic>{building.door_topic(d)}</topic>" in sdf and building.door_topic(d) == f"/door_{d['id']}_panel/cmd_pos"
        w = d["spec"]["width"]
        if d["spec"]["door_type"] == "sliding":
            assert building.hinge_xy(d) is None
            assert building.open_position(d) == pytest.approx(round(0.9 * (w + 0.15), 3))
        else:
            hx, hy = building.hinge_xy(d)
            # the hinge axis sits at one jamb: about half the panel from the door centre along the wall, within ~0.06 m of the wall line
            ux, uy = -math.sin(d["yaw"]), math.cos(d["yaw"])
            along = (hx - d["pos"][0]) * ux + (hy - d["pos"][1]) * uy
            across = (hx - d["pos"][0]) * math.cos(d["yaw"]) + (hy - d["pos"][1]) * math.sin(d["yaw"])
            assert abs(abs(along) - (w - 0.02) / 2) < 0.02 and abs(across) < 0.1
            assert building.open_position(d) == pytest.approx(1.2 * d["spec"]["swing_x"] * d["spec"]["hinge_y"])


def test_after_opening_unlocks_rooms():
    gained = 0
    for seed in SEEDS:
        b = building.make_building(CFG, seed)
        before = json.dumps(b, sort_keys=True)
        ids = [d["id"] for d in b["doors"]]
        nb = building.after_opening(b, ids, CFG)
        assert json.dumps(b, sort_keys=True) == before                      # input untouched
        for d0, d1 in zip(b["doors"], nb["doors"]):
            assert d1["state"] == "open" and d1["open_frac"] == 1.0 and d1["spec"] == d0["spec"]
        assert nb["poses"] == building.approach_poses(nb["layout"], nb["doors"], CFG, nb.get("clutter"))
        assert all(p["reachable"] for p in nb["poses"])                     # every door fully open -> every space reachable
        gained += sum(p["reachable"] for p in nb["poses"]) - sum(p["reachable"] for p in b["poses"])
        same = building.after_opening(b, [], CFG)
        assert same["doors"] == b["doors"]
    assert gained > 0


def test_write_map_same_as_write(tmp_path):
    b = building.make_building(CFG, 3)
    building.write(b, tmp_path / "a", "x")
    building.write_map(b, tmp_path / "b" / "x")
    for ext in (".map.pgm", ".map.yaml"):
        assert (tmp_path / "a" / f"x{ext}").read_bytes() == (tmp_path / "b" / f"x{ext}").read_bytes()
