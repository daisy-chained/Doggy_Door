# Unit tests for sim/doorsim/clutter.py and door room v2 (no Gazebo needed): python3 -m pytest sim/tests
import json, math, pathlib, sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest
import yaml

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import clutter

ROOM = SIM / "door_room"
ATTRS_V1 = json.loads((ROOM / "door_room_v1.attributes.json").read_text())
ATTRS_V2 = json.loads((ROOM / "door_room_v2.attributes.json").read_text())
CFG = yaml.safe_load((SIM / "configs" / "door_room_v2_2k.yaml").read_text())
CCFG = CFG["clutter"]
POOL = clutter.POOL


def rect_overlap(a, b):
    return a["x"][0] < b["x"][1] and b["x"][0] < a["x"][1] and a["y"][0] < b["y"][1] and b["y"][0] < a["y"][1]


def aabb(corners):
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    return {"x": [min(xs), max(xs)], "y": [min(ys), max(ys)]}


# --- pool ---
def test_pool_ids_unique_and_free():
    ids = [p["id"] for p in POOL]
    assert len(ids) == len(set(ids))
    v1_ids = {o["id"] for o in ATTRS_V1["objects"].values()}
    assert all(100 <= i <= 254 and i not in v1_ids for i in ids)
    assert {p["role"] for p in POOL} == {"clutter", "lookalike"}
    assert {p["place"] for p in POOL} == {"floor", "wall"}
    assert all(len(p["size"]) == 3 and min(p["size"]) > 0 for p in POOL)
    # look-alikes from Madelief's design: cabinet doors with knobs, drawer pulls, picture frames
    kinds = {p["kind"] for p in POOL if p["role"] == "lookalike"}
    assert {"cabinet", "drawers", "frame"} <= kinds
    # mix: simple shapes (recoloured per scene) and furniture built from shapes
    assert {"box", "cylinder", "sphere"} <= {p["kind"] for p in POOL if p["recolour"]}


def test_world_sdf_one_static_model_per_item_without_collision():
    root = ET.fromstring(f"<sdf version='1.9'><world name='w'>{clutter.world_sdf(POOL)}</world></sdf>")
    models = {m.get("name"): m for m in root.iter("model")}
    assert set(models) == {p["name"] for p in POOL}
    for p in POOL:
        m = models[p["name"]]
        assert m.findtext("static").strip() == "true"
        assert not list(m.iter("collision"))  # visual only: nothing gets pushed or falls
        assert int(m.find("plugin").findtext("label")) == p["id"]
        assert [float(v) for v in m.findtext("pose").split()][:3] == clutter.PARK


# --- door room v2 ---
def test_v2_attributes_are_v1_plus_pool():
    for name, o in ATTRS_V1["objects"].items():
        assert ATTRS_V2["objects"][name] == o
    assert ATTRS_V2["classes"] == ATTRS_V1["classes"]
    for p in POOL:
        o = ATTRS_V2["objects"][p["name"]]
        assert o["id"] == p["id"] and o["class"] is None and o["role"] == p["role"] and o["kind"] == p["kind"]
        assert not any(k.endswith("_world") for k in o)  # frames.build_frame turns *_world into *_cam


def test_v2_sdf_has_v1_world_and_pool():
    root = ET.parse(ROOM / "door_room_v2.sdf").getroot()
    world = root.find("world")
    assert world.get("name") == "door_room_v2"
    names = {m.get("name") for m in world.iter("model")}
    v1 = {m.get("name") for m in ET.parse(ROOM / "door_room_v1.sdf").getroot().find("world").iter("model")}
    assert v1 <= names and {p["name"] for p in POOL} <= names


# --- placement ---
def scenes(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return [clutter.place(rng, POOL, CCFG) for _ in range(n)]


def test_place_counts_and_parking():
    by_name = {p["name"]: p for p in POOL}
    for s in scenes():
        assert set(s) == set(by_name)
        floor = [n for n, v in s.items() if v["in_scene"] and by_name[n]["place"] == "floor"]
        wall = [n for n, v in s.items() if v["in_scene"] and by_name[n]["place"] == "wall"]
        assert CCFG["floor_count"][0] <= len(floor) <= CCFG["floor_count"][1]
        assert CCFG["wall_count"][0] <= len(wall) <= CCFG["wall_count"][1]
        for n, v in s.items():
            if not v["in_scene"]:
                assert v["position"] == clutter.PARK


def test_floor_items_in_zone_on_floor_no_overlap_not_in_keep_out():
    by_name = {p["name"]: p for p in POOL}
    zone = CCFG["zone"]
    for s in scenes():
        boxes = []
        for n, v in s.items():
            p = by_name[n]
            if not v["in_scene"] or p["place"] != "floor":
                continue
            assert v["position"][2] == pytest.approx(p["size"][2] / 2)  # standing on the floor
            bb = aabb(clutter.footprint(v, p))
            assert zone["x"][0] - 1e-9 <= bb["x"][0] and bb["x"][1] <= zone["x"][1] + 1e-9
            assert zone["y"][0] - 1e-9 <= bb["y"][0] and bb["y"][1] <= zone["y"][1] + 1e-9
            for k in CCFG["keep_out"]:
                assert not rect_overlap(bb, k), (n, bb, k)
            boxes.append((n, bb))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                assert not rect_overlap(boxes[i][1], boxes[j][1]), (boxes[i], boxes[j])


def test_wall_items_on_the_wall_beside_the_doors():
    by_name = {p["name"]: p for p in POOL}
    w = CCFG["wall"]
    for s in scenes():
        spans = []
        for n, v in s.items():
            p = by_name[n]
            if not v["in_scene"] or p["place"] != "wall":
                continue
            x, y, z = v["position"]
            assert x + p["size"][0] / 2 == pytest.approx(w["x"])  # back flat on the wall face
            assert v["yaw"] == 0
            half_w, half_h = p["size"][1] / 2, p["size"][2] / 2
            assert w["y"][0] <= y - half_w and y + half_w <= w["y"][1]
            assert w["z"][0] <= z - half_h and z + half_h <= w["z"][1]
            for lo, hi in w["keep_out_y"]:
                assert y + half_w <= lo or y - half_w >= hi  # not over a door opening or its trim
            spans.append((y - half_w, y + half_w))
        spans.sort()
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))  # wall items don't overlap


def test_clutter_can_stand_in_front_of_a_door():
    # Madelief's design: clutter may hide part of a door. Some floor items must land in front of the doors.
    by_name = {p["name"]: p for p in POOL}
    doors = [o["front_world"]["point"] for o in ATTRS_V1["objects"].values() if "front_world" in o]
    hits = 0
    for s in scenes():
        for n, v in s.items():
            if v["in_scene"] and by_name[n]["place"] == "floor":
                x, y, _ = v["position"]
                hits += any(abs(y - d[1]) < 0.45 and d[0] - x < 2.0 for d in doors)
    assert hits > 50


def test_recolour_and_yaw():
    by_name = {p["name"]: p for p in POOL}
    yaws = []
    for s in scenes(100):
        for n, v in s.items():
            p = by_name[n]
            if p["recolour"] and v["in_scene"]:
                assert len(v["colour"]) == 3 and all(0 <= c <= 1 for c in v["colour"])
            elif not p["recolour"]:
                assert v["colour"] is None
            if v["in_scene"] and p["place"] == "floor":
                assert -math.pi <= v["yaw"] <= math.pi
                yaws.append(v["yaw"])
    assert np.std(yaws) > 0.5  # yaw varies


def test_same_seed_same_clutter():
    assert scenes(5, seed=3) == scenes(5, seed=3)
    assert scenes(5, seed=3) != scenes(5, seed=4)


def test_footprint_rotates():
    p = {"size": [0.4, 1.0, 0.5]}
    c0 = aabb(clutter.footprint({"position": [1, 0, 0.25], "yaw": 0.0}, p))
    assert c0["x"] == pytest.approx([0.8, 1.2]) and c0["y"] == pytest.approx([-0.5, 0.5])
    c90 = aabb(clutter.footprint({"position": [1, 0, 0.25], "yaw": math.pi / 2}, p))
    assert c90["x"] == pytest.approx([0.5, 1.5]) and c90["y"] == pytest.approx([-0.2, 0.2])


# --- camera inside clutter (Gazebo renders straight through a box from inside: tested 2026-10-29) ---
def test_camera_blocked():
    item = {"name": "box_x", "size": [0.5, 0.5, 0.6], "place": "floor"}
    placed = {"box_x": {"position": [2.0, 1.0, 0.3], "yaw": 0.0, "in_scene": True, "colour": None}}
    assert clutter.camera_blocked([2.0, 1.0, 0.3], placed, [item], 0.15)
    assert clutter.camera_blocked([2.35, 1.0, 0.5], placed, [item], 0.15)      # inside the margin
    assert not clutter.camera_blocked([2.0, 1.0, 0.8], placed, [item], 0.15)   # above it
    assert not clutter.camera_blocked([1.5, 1.0, 0.3], placed, [item], 0.15)
    placed["box_x"]["in_scene"] = False
    assert not clutter.camera_blocked([2.0, 1.0, 0.3], placed, [item], 0.15)


# --- how much of each door / handle the clutter hides ---
def test_visible_fraction():
    seg_clear = np.zeros((10, 10), np.uint8)
    seg_clear[0:10, 0:4] = 11      # door A frame
    seg_clear[2:8, 1:3] = 13       # door A glass pane: counts as door A
    seg_clear[5, 3] = 12           # handle A
    seg_clear[0:10, 6:10] = 21     # door B
    seg = seg_clear.copy()
    seg[0:5, 0:4] = 150            # clutter hides the top half of door A (frame + glass)
    seg[5, 3] = 150                # and the whole handle
    f = clutter.visible_fraction(seg, seg_clear, ATTRS_V2)
    assert f["door_a"] == pytest.approx(0.5)
    assert f["handle_a"] == 0.0
    assert f["door_b"] == 1.0
    assert f["handle_b"] is None   # not in view at all
    assert "glass_a" not in f and not any(n in f for n in (p["name"] for p in POOL))  # only objects with a class
