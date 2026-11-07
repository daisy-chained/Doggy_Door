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
#     from the corridor through open doors only.
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
       "approach": {"distances": [2.5, 1.0]}, "robot_radius": 0.3, "map_resolution": 0.05}
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
            if d["state"] == "open" and (a in reach) != (c in reach):
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
