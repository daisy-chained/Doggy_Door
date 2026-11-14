# Dutch-house test world, round 1 (Madelief 2026-11-14: "Extend our generator"; DoD section 1 + 2e: >= 10 generated Dutch-house
# layouts with fixed seeds). Numbers in sim/configs/houses_v1.yaml (Claude decisions, research-based, dev log 2026-11-14).
# Windows come in round 2. No Gazebo needed: python3 -m pytest sim/tests/test_houses.py
#
# building.sample_house(rng, cfg, floor) -> layout like sample_layout's, plus:
#   {"bounds": [0, 0, W, D], "corridor": hub rect (hall on floor 0, landing on floor 1), "rooms": [{"id", "rect"}], "kind": "house",
#    "floor": 0 | 1, "strip_side": "west" | "east", "stairs": [x0, y0, x1, y1] (an obstacle box on both floors), "plan": [[a, b], ...]
#    (space pairs that get exactly one door, a = front), "door_types": {"a|b": "hinged" | "sliding"} (forced types)}
#   y = 0 is the front facade, y = D the back. The strip (hall / landing) runs front to back along the west or east outer wall,
#   width cfg strip_width; the rest ("main") is the other side.
#   Floor 0: strip = hall (front, y 0 .. hall_d), toilet (behind it, depth toilet_depth), kitchen (the rest, >= kitchen_min).
#     Main = "living" (whole depth) or, with p_ensuite, "living_front" (y 0 .. f D) + "living_back" with a forced sliding door.
#     Plan: hall-living (or hall-living_front), hall-toilet (forced hinged), living-kitchen (or living_back-kitchen),
#     living_front-living_back (forced sliding).
#   Floor 1: strip = bed3 (front), landing, bath (back); main = bed1 (front, y 0 .. f D) + bed2. Plan: landing-bed1, landing-bed2,
#     landing-bed3, landing-bath.
#   Stairs: a straight run against the strip's outer wall (x touches the bounds), width stairs.width, from y = front_gap to
#     front_gap + length. Floor 0: the whole run, and the hall reaches >= hall_free past its top end. Floor 1: only the part inside
#     the landing (the stair void with a railing), same x span, same top end; the landing reaches >= step_off past it.
#   Every planned pair shares a wall of at least min_shared m. Draws are redone (rejection) until every rule holds; the footprint
#   (W, D, strip side + width, stairs) comes from the rng first, so floors 0 and 1 of one seed share it.
# building.make_building(cfg, seed, split="train", floor=0): cfg kind "house" -> sample_house + doors from the plan (exactly one door
#   per planned pair, on the shared wall, connects = the pair, forced types honoured) + clutter (never on the stairs) + "start":
#   [x, y] in the hub, clear of the stairs and walls by robot_radius. The stairs are occupied in the map, a static model "stairs"
#   (visual + collision, no label) in the SDF. Corridor buildings (no "kind") exactly as before.
# make_buildings.py: kind house -> both floors per seed, named hse_<split>_<seed:03d>_f0 / _f1.
# door_tour.py writes "building_path" (its BUILDING argument) into tour.json; score_patrol.py / score_stop3d.py load
#   <building_path>.building.json when it is there (relative paths from the repo root), else the old bld_ name rule.
import json, math, pathlib, subprocess, sys
import numpy as np
import pytest
import yaml

SIM = pathlib.Path(__file__).resolve().parents[1]
REPO = SIM.parent
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(SIM / "building"))
from doorsim import building

HCFG = yaml.safe_load((SIM / "configs" / "houses_v1.yaml").read_text())
H = HCFG["house"]
SEEDS = range(25)


def spaces(lay):
    return {"corridor": lay["corridor"], **{r["id"]: r["rect"] for r in lay["rooms"]}}


def area(r):
    return (r[2] - r[0]) * (r[3] - r[1])


def overlap(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def shared(a, b):
    """Length of the wall two rects share (0 when they don't touch)."""
    if abs(a[2] - b[0]) < 1e-6 or abs(b[2] - a[0]) < 1e-6:
        return max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    if abs(a[3] - b[1]) < 1e-6 or abs(b[3] - a[1]) < 1e-6:
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    return 0.0


def inside(inner, outer, eps=1e-6):
    return inner[0] >= outer[0] - eps and inner[1] >= outer[1] - eps and inner[2] <= outer[2] + eps and inner[3] <= outer[3] + eps


@pytest.fixture(scope="module", params=[0, 1], ids=["ground", "upper"])
def layouts(request):
    return request.param, [building.sample_house(np.random.default_rng(s), HCFG, request.param) for s in SEEDS]


# --- layout ---
def test_footprint_and_tiling(layouts):
    _, lays = layouts
    for lay in lays:
        x0, y0, W, D = lay["bounds"]
        assert (x0, y0) == (0.0, 0.0) and H["width"][0] <= W <= H["width"][1] and H["depth"][0] <= D <= H["depth"][1]
        sp = list(spaces(lay).values())
        assert all(inside(r, lay["bounds"]) for r in sp)
        assert sum(area(r) for r in sp) == pytest.approx(W * D)
        assert all(overlap(a, b) < 1e-9 for i, a in enumerate(sp) for b in sp[i + 1:])
        assert lay["kind"] == "house"


def test_room_sets(layouts):
    floor, lays = layouts
    ensuite = 0
    for lay in lays:
        assert lay["floor"] == floor
        ids = {r["id"] for r in lay["rooms"]}
        if floor == 0:
            assert ids in ({"living", "kitchen", "toilet"}, {"living_front", "living_back", "kitchen", "toilet"})
            ensuite += "living_front" in ids
            t = spaces(lay)["toilet"]
            w, d = sorted([t[2] - t[0], t[3] - t[1]])
            assert w >= 0.85 and d >= 1.2                       # Bbl toilet minimum
        else:
            assert ids == {"bed1", "bed2", "bed3", "bath"}
    if floor == 0:
        assert 0 < ensuite < len(lays)


def test_ensuite_share():
    n = sum("living_front" in {r["id"] for r in building.sample_house(np.random.default_rng(s), HCFG, 0)["rooms"]} for s in range(300))
    assert n / 300 == pytest.approx(H["ground"]["p_ensuite"], abs=0.08)


def test_strip_and_stairs(layouts):
    floor, lays = layouts
    sides = set()
    for lay in lays:
        x0, _, W, _ = lay["bounds"]
        hub, st = lay["corridor"], lay["stairs"]
        sides.add(lay["strip_side"])
        sw = hub[2] - hub[0]
        assert H["strip_width"][0] - 1e-9 <= sw <= H["strip_width"][1] + 1e-9
        assert (hub[0] == 0.0) if lay["strip_side"] == "west" else (hub[2] == pytest.approx(W))
        assert inside(st, hub)
        assert (st[0] == pytest.approx(0.0)) if lay["strip_side"] == "west" else (st[2] == pytest.approx(W))   # against the outer wall
        assert H["stairs"]["width"][0] - 1e-9 <= st[2] - st[0] <= H["stairs"]["width"][1] + 1e-9
        assert sw - (st[2] - st[0]) >= 0.9 - 1e-9                    # passage beside the stairs
        if floor == 0:
            assert H["stairs"]["length"][0] - 1e-9 <= st[3] - st[1] <= H["stairs"]["length"][1] + 1e-9
            assert H["stairs"]["front_gap"][0] - 1e-9 <= st[1] <= H["stairs"]["front_gap"][1] + 1e-9
            assert hub[3] - st[3] >= H["ground"]["hall_free"] - 1e-9
        else:
            assert hub[3] - st[3] >= H["upper"]["step_off"] - 1e-9
    assert sides == {"west", "east"}


def test_floors_share_the_footprint():
    for s in SEEDS:
        g, u = (building.sample_house(np.random.default_rng(s), HCFG, f) for f in (0, 1))
        assert g["bounds"] == u["bounds"] and g["strip_side"] == u["strip_side"]
        assert (g["corridor"][0], g["corridor"][2]) == (u["corridor"][0], u["corridor"][2])
        assert (g["stairs"][0], g["stairs"][2], g["stairs"][3]) == pytest.approx((u["stairs"][0], u["stairs"][2], u["stairs"][3]))
        assert u["stairs"][1] >= g["stairs"][1] - 1e-9


def test_plan(layouts):
    floor, lays = layouts
    for lay in lays:
        sp, plan = spaces(lay), lay["plan"]
        ids = {r["id"] for r in lay["rooms"]}
        if floor == 0:
            want = {("corridor", "toilet")}
            want |= ({("corridor", "living"), ("living", "kitchen")} if "living" in ids else
                     {("corridor", "living_front"), ("living_back", "kitchen"), ("living_front", "living_back")})
            assert lay["door_types"]["corridor|toilet"] == "hinged"
            if "living_front" in ids:
                assert lay["door_types"]["living_front|living_back"] == "sliding"
        else:
            want = {("corridor", r) for r in ("bed1", "bed2", "bed3", "bath")}
        assert {tuple(p) for p in plan} == want
        for a, b in plan:
            assert shared(sp[a], sp[b]) >= H["min_shared"] - 1e-9, (a, b)


# --- whole house ---
@pytest.fixture(scope="module", params=[0, 1], ids=["ground", "upper"])
def houses(request):
    return [building.make_building(HCFG, s, "train", request.param) for s in range(12)]


def on_wall(d, rect):
    x, y = d["pos"]
    return (abs(x - rect[0]) < 1e-6 or abs(x - rect[2]) < 1e-6) and rect[1] - 1e-6 <= y <= rect[3] + 1e-6 or \
           (abs(y - rect[1]) < 1e-6 or abs(y - rect[3]) < 1e-6) and rect[0] - 1e-6 <= x <= rect[2] + 1e-6


def test_doors_follow_the_plan(houses):
    for b in houses:
        lay, sp = b["layout"], spaces(b["layout"])
        pairs = sorted(tuple(d["connects"]) for d in b["doors"])
        assert pairs == sorted(tuple(p) for p in lay["plan"])
        for d in b["doors"]:
            a, c = d["connects"]
            assert on_wall(d, sp[a]) and on_wall(d, sp[c])
            assert HCFG["door"]["width"][0] - 1e-9 <= d["spec"]["width"] <= HCFG["door"]["width"][1] + 1e-9
            forced = lay["door_types"].get(f"{a}|{c}")
            if forced:
                assert d["spec"]["door_type"] == forced
            # local +x = normal from connects[0] into connects[1]
            nx, ny = math.cos(d["yaw"]), math.sin(d["yaw"])
            p = (d["pos"][0] + 0.2 * nx, d["pos"][1] + 0.2 * ny)
            r = sp[c]
            assert r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3]


def test_same_seed_same_house():
    a = building.make_building(HCFG, 3, "train", 1)
    b = building.make_building(HCFG, 3, "train", 1)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def cell(origin, res, shape, x, y):
    return shape[0] - 1 - int((y - origin[1]) / res), int((x - origin[0]) / res)


def test_stairs_occupied_and_start_clear(houses):
    for b in houses:
        lay, res = b["layout"], b["map_resolution"]
        grid, origin = building.occupancy(lay, b["doors"], res, b.get("clutter"))
        st = lay["stairs"]
        r, c = cell(origin, res, grid.shape, (st[0] + st[2]) / 2, (st[1] + st[3]) / 2)
        assert grid[r, c] == 0
        sx, sy = b["start"]
        hub = lay["corridor"]
        rad = HCFG["robot_radius"]
        assert hub[0] + rad <= sx <= hub[2] - rad and hub[1] + rad <= sy <= hub[3] - rad
        assert not (st[0] - rad < sx < st[2] + rad and st[1] - rad < sy < st[3] + rad)
        for cl in b.get("clutter", []):
            if cl["place"] == "floor":
                lo, hi = building._aabb_of(cl)[:2], building._aabb_of(cl)[2:]
                assert lo[0] >= st[2] or hi[0] <= st[0] or lo[1] >= st[3] or hi[1] <= st[1], cl["name"]   # off the stairs


def test_every_room_reachable_with_all_doors_open():
    cfg = {k: v for k, v in HCFG.items() if k != "clutter"}
    for floor in (0, 1):
        for s in range(12):
            b = building.make_building(cfg, s, "train", floor)
            b = building.after_opening(b, [d["id"] for d in b["doors"]], cfg)
            res = b["map_resolution"]
            grid, origin = building.occupancy(b["layout"], b["doors"], res, None)
            lab = building._fit_labels(grid, res, cfg["robot_radius"])
            home = lab[cell(origin, res, grid.shape, *b["start"])]
            assert home > 0
            for rid, r in spaces(b["layout"]).items():
                # some cell of the room is in the start's component
                r0, c0 = cell(origin, res, grid.shape, r[0], r[3])
                r1, c1 = cell(origin, res, grid.shape, r[2], r[1])
                assert (lab[r0:r1 + 1, c0:c1 + 1] == home).any(), (floor, s, rid)


def test_sdf_has_stairs_model(tmp_path):
    import xml.etree.ElementTree as ET
    b = building.make_building(HCFG, 1, "train", 0)
    building.write(b, tmp_path, "hse_train_001_f0")
    text = (tmp_path / "hse_train_001_f0.sdf").read_text()
    world = ET.fromstring(text[text.index("<sdf"):]).find("world")
    m = world.find("model[@name='stairs']")
    assert m is not None and m.find(".//collision") is not None and m.find(".//visual") is not None
    meta = json.loads((tmp_path / "hse_train_001_f0.building.json").read_text())
    assert meta["start"] == b["start"] and meta["layout"]["stairs"] == b["layout"]["stairs"]


def test_corridor_buildings_unchanged_api():
    bcfg = yaml.safe_load((SIM / "configs" / "buildings_v1.yaml").read_text())
    b = building.make_building(bcfg, 5, "train")
    assert "kind" not in b["layout"] and "stairs" not in b["layout"]


def test_make_buildings_cli_houses(tmp_path):
    cfg = dict(HCFG, count=1, seed=7)
    p = tmp_path / "h.yaml"
    p.write_text(yaml.safe_dump(cfg))
    subprocess.run([sys.executable, str(SIM / "building" / "make_buildings.py"), str(p), "--out", str(tmp_path / "out")], check=True)
    for f in ("f0", "f1"):
        for ext in (".sdf", ".building.json", ".map.pgm", ".map.yaml"):
            assert (tmp_path / "out" / f"hse_train_007_{f}{ext}").exists()


def test_score_patrol_uses_building_path(tmp_path):
    import score_patrol
    b = building.make_building(HCFG, 2, "train", 1)
    building.write(b, tmp_path / "bld", "hse_train_002_f1")
    run = tmp_path / "some_run_name"
    run.mkdir()
    (run / "tour.json").write_text(json.dumps({"building_path": str(tmp_path / "bld" / "hse_train_002_f1"), "visits": [],
                                               "patrol": {"candidates": [], "viewpoints": []}}))
    r = score_patrol.score(run)
    assert r["doors"] == len(b["doors"]) and r["missed"] == sorted(d["id"] for d in b["doors"])
