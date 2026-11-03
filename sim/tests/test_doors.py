# Unit tests for door variety (door room v4): sim/doorsim/doors.py, sim/door_room/make_v4.py config, the side-aware
# viewpoints / clutter, multi-world splits in the generator and the effects tool (no Gazebo needed): python3 -m pytest sim/tests
import json, math, pathlib, sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest
import yaml

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import clutter, doors, effects, export, generate, sampling

WCFG = yaml.safe_load((SIM / "configs" / "worlds_v4.yaml").read_text())
DCFG = WCFG["door"]
GEN_CFG = yaml.safe_load((SIM / "configs" / "door_room_v4_2k.yaml").read_text())
WALL_X = doors.WALL_X  # wall centre plane, 0.1 thick: faces at WALL_X -/+ 0.05


def specs(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return [doors.sample_door(rng, DCFG, slot) for _ in range(n) for slot in doors.SLOTS]


def world(seed=0, heldout=False):
    return doors.sample_world(np.random.default_rng(seed), WCFG, heldout=heldout)


def find_world(pred, tries=200, heldout=False):
    for s in range(tries):
        w = world(s, heldout)
        if pred(w):
            return w
    raise AssertionError("no world matched")


# --- door specs ---
def test_sample_door_values_and_ranges():
    ss = specs()
    for s in ss:
        assert s["door_type"] in ("hinged", "sliding") and isinstance(s["glass"], bool)
        assert DCFG["width"][0] <= s["width"] <= DCFG["width"][1] and DCFG["height"][0] <= s["height"] <= DCFG["height"][1]
        assert s["frame_style"] in ("none", "thin", "thick")
        assert len(s["colour"]) == 3 and all(0 <= c <= 1 for c in s["colour"])
        assert len(s["frame_colour"]) == 3
        if s["door_type"] == "hinged":
            assert s["handle_type"] in ("lever", "knob")
            assert s["hinge_y"] in (-1, 1) and s["swing_x"] in (-1, 1)  # hinge on the -y / +y edge, swings to -x / +x
        else:
            assert s["handle_type"] == "flush_pull"
            assert s["hangs_on"] in ("front", "back")
    # everything actually varies (Madelief's list: the core 5 + size and colour + frame style)
    for key in ("door_type", "glass", "handle_type", "frame_style"):
        assert len({s[key] for s in ss}) >= 2, key
    hinged = [s for s in ss if s["door_type"] == "hinged"]
    assert {s["hinge_y"] for s in hinged} == {-1, 1} and {s["swing_x"] for s in hinged} == {-1, 1}
    assert len({round(s["width"], 3) for s in ss}) > 50 and len({tuple(s["colour"]) for s in ss}) > 50


def test_sample_door_same_seed_same_door():
    assert specs(20, seed=3) == specs(20, seed=3)
    assert specs(20, seed=3) != specs(20, seed=4)


def test_seen_from_flips_with_the_side():
    for s in specs(100):
        f, b = doors.seen_from(s, "front"), doors.seen_from(s, "back")
        if s["door_type"] == "hinged":
            assert {f["opening"], b["opening"]} == {"push", "pull"}
            assert {f["hinge_side"], b["hinge_side"]} == {"left", "right"}
            # front = camera at x < wall looking +x: right = -y; swinging away from the camera (+x) = push
            assert f["hinge_side"] == ("right" if s["hinge_y"] < 0 else "left")
            assert f["opening"] == ("push" if s["swing_x"] > 0 else "pull")
        else:
            assert f["opening"] == b["opening"] == "slide"
            assert {f["slide_direction"], b["slide_direction"]} == {"left", "right"}
            assert "hinge_side" not in f


def test_heldout_combos():
    combos = WCFG["heldout"]
    assert combos  # e.g. glass + knob, sliding + glass
    for s in specs():
        expect = any(all(s[k] == v for k, v in c.items()) for c in combos)
        assert doors.is_heldout(s, combos) == expect


def test_normal_worlds_have_no_heldout_doors_and_heldout_worlds_do():
    for seed in range(40):
        w = world(seed)
        assert set(w["doors"]) == set(doors.SLOTS)
        assert not any(doors.is_heldout(d, WCFG["heldout"]) for d in w["doors"].values())
        h = world(seed, heldout=True)
        assert any(doors.is_heldout(d, WCFG["heldout"]) for d in h["doors"].values())


def test_sliding_doors_slide_away_from_the_other_doorway():
    # slot a sits at y = -1, slot b at y = +1: room to slide only on the outer side
    for s in specs():
        if s["door_type"] == "sliding":
            assert s["slide_y"] == (-1 if doors.SLOT_Y[s["slot"]] < 0 else 1)


def test_make_worlds_splits_by_world_heldout_only_in_test():
    ws = doors.make_worlds(WCFG)
    assert len(ws) == WCFG["count"] + WCFG["heldout_count"]
    assert len({w["name"] for w in ws}) == len(ws)
    held = [w for w in ws if w["heldout"]]
    assert len(held) == WCFG["heldout_count"] and all(w["split"] == "test" for w in held)
    normal = [w for w in ws if not w["heldout"]]
    counts = {sp: sum(w["split"] == sp for w in normal) for sp in export.SPLITS}
    assert all(counts[sp] >= 1 for sp in export.SPLITS)
    assert counts["train"] > counts["val"] and counts["train"] > counts["test"]
    assert doors.make_worlds(WCFG) == ws  # fixed seed


# --- attributes ---
def test_attributes_ids_classes_and_sides():
    w = find_world(lambda w: any(d["glass"] for d in w["doors"].values()))
    a = doors.attributes(w, "door_room_v4_test")
    objs = a["objects"]
    assert a["classes"] == {"1": "door", "2": "handle"}
    for k, slot in enumerate(doors.SLOTS, 1):
        d, spec = objs[f"door_{slot}"], w["doors"][slot]
        assert d["id"] == 10 * k + 1 and d["class"] == 1
        assert d["door_type"] == spec["door_type"] and d["glass"] == spec["glass"] and d["heldout"] == w["doors_heldout"][slot]
        assert d["seen_from"] == {"front": doors.seen_from(spec, "front"), "back": doors.seen_from(spec, "back")}
        assert set(d["approach"]) == {"front", "back"}
        assert d["approach"]["front"]["direction"][0] < 0 < d["approach"]["back"]["direction"][0]
        # a handle on each face, its own id
        hf, hb = objs[f"handle_{slot}"], objs[f"handle_{slot}_back"]
        assert hf["id"] == 10 * k + 2 and hb["id"] == 10 * k + 4
        assert hf["class"] == hb["class"] == 2 and hf["door"] == hb["door"] == f"door_{slot}"
        assert hf["handle_type"] == hb["handle_type"] == spec["handle_type"]
        assert (hf["face"], hb["face"]) == ("front", "back")
        if spec["glass"]:
            g = objs[f"glass_{slot}"]
            assert g["id"] == 10 * k + 3 and g["class"] is None and g["part"] == "glass_pane" and g["part_of"] == f"door_{slot}"
            assert g["detach_topic"] == f"/glass_{slot}/detach"
        else:
            assert f"glass_{slot}" not in objs
    ids = [o["id"] for o in objs.values()]
    assert len(ids) == len(set(ids)) and all(1 <= i <= 255 for i in ids)
    assert any(o.get("role") == "lookalike" for o in objs.values())  # clutter pool entries kept


def test_hinged_joint_swings_to_the_spec_side():
    for seed in range(30):
        w = world(seed)
        objs = doors.attributes(w, "t")["objects"]
        for slot, spec in w["doors"].items():
            if spec["door_type"] != "hinged":
                continue
            o = objs[f"door_{slot}"]
            assert o["joint"]["type"] == "revolute" and o["joint"]["unit"] == "rad"
            lo, hi = o["joint"]["range"]
            assert lo <= o["joint"]["open"] <= hi and abs(o["joint"]["open"]) > 0.5
            hp = o["hinge_world"]["point"]
            assert math.copysign(1, hp[1] - doors.SLOT_Y[slot]) == spec["hinge_y"]
            # lock edge (hinge -> door centre along y) turned by the open angle moves to the swing side in x
            lock = doors.SLOT_Y[slot] - hp[1]
            th = o["joint"]["open"]
            x_after = -lock * math.sin(th)
            assert math.copysign(1, x_after) == spec["swing_x"]


def test_sliding_joint_moves_outward():
    for seed in range(30):
        w = world(seed)
        objs = doors.attributes(w, "t")["objects"]
        for slot, spec in w["doors"].items():
            if spec["door_type"] != "sliding":
                continue
            o = objs[f"door_{slot}"]
            assert o["joint"]["type"] == "prismatic" and o["joint"]["unit"] == "m"
            assert o["joint"]["range"][0] == 0 and o["joint"]["range"][1] >= spec["width"] - 1e-9
            assert np.sign(o["slide_world"]["direction"][1]) == spec["slide_y"]
            # hung on its face: panel on the front face sits at x < wall, on the back face at x > wall
            assert (o["slide_world"]["point"][0] < WALL_X) == (spec["hangs_on"] == "front")


def test_keep_out_covers_swing_and_slide_paths():
    for seed in range(30):
        w = world(seed)
        objs = doors.attributes(w, "t")["objects"]
        for slot, spec in w["doors"].items():
            ko = objs[f"door_{slot}"]["keep_out"]
            assert set(ko) == {"front", "back"}
            if spec["door_type"] == "hinged":
                side = "back" if spec["swing_x"] > 0 else "front"
                r = max(ko[side], key=lambda k: k["x"][1] - k["x"][0])
                assert r["x"][1] - r["x"][0] >= spec["width"] - 1e-6  # the swing arc's depth
            else:
                r = ko[spec["hangs_on"]][0]
                assert r["y"][1] - r["y"][0] >= 2 * spec["width"] - 1e-6  # opening + where the panel slides to


# --- world file ---
def parse(sdf):
    root = ET.fromstring(sdf)
    return root.find("world"), {m.get("name"): m for m in root.find("world").findall("model")}


def label(model):
    for p in model.findall("plugin"):
        if p.find("label") is not None:
            return int(p.find("label").text)


def test_world_sdf_models_labels_and_lights():
    w = find_world(lambda w: any(d["glass"] for d in w["doors"].values()))
    a = doors.attributes(w, "door_room_v4_test")
    world_el, models = parse(doors.world_sdf(w, "door_room_v4_test"))
    assert world_el.get("name") == "door_room_v4_test"
    for name, o in a["objects"].items():
        assert name in models, name
        assert label(models[name]) == o["id"], name
    for m in ("floor", "back_wall", "rig", "room_shell"):
        assert m in models
    lights = {l.get("name") for l in world_el.iter("light")}
    assert {"sun", "ceiling_light", "ceiling_light_back", "headlight"} <= lights
    # a joint position controller per door on /<door>/cmd_pos, glass held by a detachable joint with the attribute topics
    for slot in doors.SLOTS:
        m = models[f"door_{slot}"]
        topics = [p.find("topic").text for p in m.findall("plugin") if p.find("topic") is not None]
        assert f"/door_{slot}/cmd_pos" in topics
        joint = a["objects"][f"door_{slot}"]["joint"]
        j = [e for e in m.findall("joint") if e.get("name") == joint["name"]][0]
        lim = j.find("axis/limit")
        assert [float(lim.find("lower").text), float(lim.find("upper").text)] == pytest.approx(joint["range"])
        children = {p.find("child_model").text for p in m.findall("plugin") if p.find("child_model") is not None}
        assert {f"handle_{slot}", f"handle_{slot}_back"} <= children
        if w["doors"][slot]["glass"]:
            assert f"glass_{slot}" in children


def test_world_room_reaches_far_enough_behind_the_wall():
    # back-side viewpoints go up to view.distance from the wall face: the room must be that deep on both sides
    _, models = parse(doors.world_sdf(world(0), "t"))
    xs = []
    for v in models["room_shell"].iter("visual"):
        x = float(v.find("pose").text.split()[0]); sx = float(v.find("geometry/box/size").text.split()[0])
        xs += [x - sx / 2, x + sx / 2]
    far = GEN_CFG["view"]["distance"][1]
    assert min(xs) <= WALL_X - 0.05 - far and max(xs) >= WALL_X + 0.05 + far


def wall_boxes(models):
    """(y0, y1, z0, z1) of the back wall's visual boxes (thick enough to be wall, not trim)."""
    out = []
    for v in models["back_wall"].iter("visual"):
        size = v.find("geometry/box/size")
        if size is None:
            continue
        sx, sy, sz = map(float, size.text.split())
        if sx < 0.09:
            continue  # trim / rail
        px, py, pz = map(float, v.find("pose").text.split()[:3])
        out.append((py - sy / 2, py + sy / 2, pz - sz / 2, pz + sz / 2))
    return out


def test_wall_openings_match_door_size():
    for seed in range(15):
        w = world(seed)
        _, models = parse(doors.world_sdf(w, "t"))
        boxes = wall_boxes(models)
        for slot, spec in w["doors"].items():
            y, hw, h = doors.SLOT_Y[slot], spec["width"] / 2, spec["height"]
            # nothing of the wall inside the opening
            for y0, y1, z0, z1 in boxes:
                assert not (y0 < y + hw - 1e-6 and y - hw + 1e-6 < y1 and z0 < h - 1e-6), (slot, seed)
            # and the wall closes right above it (lintel)
            assert any(y0 <= y - hw + 1e-6 and y + hw - 1e-6 <= y1 and abs(z0 - h) < 1e-6 for y0, y1, z0, z1 in boxes)


# --- per-frame side ---
def test_apply_side_sets_the_camera_side_properties():
    w = find_world(lambda w: w["doors"]["a"]["door_type"] == "hinged")
    objs = doors.attributes(w, "t")["objects"]
    for side in ("front", "back"):
        o = json.loads(json.dumps(objs))
        doors.apply_side(o, side)
        for k, v in objs["door_a"]["seen_from"][side].items():
            assert o["door_a"][k] == v
    assert doors.side_of([WALL_X - 1, 0, 1]) == "front" and doors.side_of([WALL_X + 1, 0, 1]) == "back"


def test_view_both_sides():
    w = world(1)
    a = doors.attributes(w, "t")
    for p_back, want in ((0.0, {"front"}), (1.0, {"back"}), (0.5, {"front", "back"})):
        cfg = dict(GEN_CFG["view"], p_back=p_back)
        rng, seen = np.random.default_rng(0), set()
        for _ in range(100):
            v = sampling.view(rng, a, cfg)
            seen.add(v["side"])
            assert v["side"] == doors.side_of(v["world_position"])
            yaw = v["world_rpy"][2]
            assert (math.cos(yaw) > 0) == (v["side"] == "front")  # looking at the wall
        assert seen == want


def test_view_old_worlds_still_front_only():
    old = json.loads((SIM / "door_room" / "door_room_v3.attributes.json").read_text())
    cfg = yaml.safe_load((SIM / "configs" / "door_room_v3_small.yaml").read_text())["view"]
    v = sampling.view(np.random.default_rng(0), old, cfg)
    assert v["side"] == "front" and v["world_position"][0] < 3


# --- clutter on both sides ---
def test_clutter_both_sides_clear_of_doors():
    ccfg = GEN_CFG["clutter"]
    pool = clutter.POOL
    for seed in range(20):
        w = world(seed)
        objs = doors.attributes(w, "t")["objects"]
        keep = doors.keep_out(objs)
        placed = clutter.place(np.random.default_rng(seed), pool, ccfg, keep)
        sides = set()
        for p in pool:
            pl = placed[p["name"]]
            if not pl["in_scene"]:
                continue
            side = doors.side_of(pl["position"])
            sides.add(side)
            if p["place"] == "wall":
                assert abs(abs(pl["position"][0] - WALL_X) - (0.05 + p["size"][0] / 2)) < 0.02
                # facing the camera on its side: yaw 0 faces -x (front), pi faces +x (back)
                assert abs(abs(pl["yaw"]) - (math.pi if side == "back" else 0.0)) < 1e-6
            if p["place"] == "floor":
                bb = clutter._aabb(clutter.footprint(pl, p))
                for k in keep[side]:
                    assert not clutter._overlap(bb, (k["x"][0], k["x"][1], k["y"][0], k["y"][1])), (seed, p["name"])
                assert abs(bb[0] - WALL_X) >= 0.05 - 1e-6 and abs(bb[1] - WALL_X) >= 0.05 - 1e-6  # not in the wall
                assert (bb[1] <= WALL_X) if side == "front" else (bb[0] >= WALL_X)
    # over many scenes both sides get clutter
    both = set()
    for seed in range(20):
        placed = clutter.place(np.random.default_rng(seed), pool, ccfg, doors.keep_out(doors.attributes(world(seed), "t")["objects"]))
        both |= {doors.side_of(pl["position"]) for pl in placed.values() if pl["in_scene"]}
    assert both == {"front", "back"}


def test_clutter_old_config_unchanged():
    cfg = yaml.safe_load((SIM / "configs" / "door_room_v3_small.yaml").read_text())["clutter"]
    a = clutter.place(np.random.default_rng(5), clutter.POOL, cfg)
    b = clutter.place(np.random.default_rng(5), clutter.POOL, cfg, None)
    assert a == b and all(pl["position"][0] < 3 for pl in a.values() if pl["in_scene"])


# --- generator: many worlds ---
WORLDS = [{"name": f"w{i}", "split": sp, "heldout": i == 3} for i, sp in enumerate(["train", "train", "val", "test", "test"])]


def test_world_scenes_and_splits():
    plan = generate.world_scenes(WORLDS, 2)
    assert [(w["name"], s) for w, s in plan] == [("w0", [0, 1]), ("w1", [2, 3]), ("w2", [4, 5]), ("w3", [6, 7]), ("w4", [8, 9])]
    assert generate.world_scenes(WORLDS, 2, "test") == [(WORLDS[3], [6, 7]), (WORLDS[4], [8, 9])]
    sp = generate.world_split_scenes(WORLDS, 2)
    assert sp == {"train": [0, 1, 2, 3], "val": [4, 5], "test": [6, 7, 8, 9]}


def test_v4_configs_load():
    for name in ("door_room_v4_small", "door_room_v4_2k"):
        c = yaml.safe_load((SIM / "configs" / f"{name}.yaml").read_text())
        assert c["worlds"].endswith("worlds.json") and c["scenes_per_world"] >= 1
        assert "ceiling_light_back" in c["lights"] and 0 < c["view"]["p_back"] < 1
        assert set(c["clutter"]["sides"]) == {"front", "back"}


# --- effects: per-world lamps + the dataset's own split ---
def test_effects_world_file_per_frame():
    ds = {"world": "door_room_v4", "world_files": {"door_room_v4_000": "door_room/v4/door_room_v4_000.sdf"}}
    assert effects.world_file(ds, {"world": "door_room_v4_000"}) == SIM / "door_room" / "v4" / "door_room_v4_000.sdf"
    assert effects.world_file({"world": "door_room_v3"}, {}) == SIM / "door_room" / "door_room_v3.sdf"


def test_effects_split_uses_the_datasets_split():
    metas = {f"{i:06d}": {"scene": s} for i, s in enumerate([0, 1, 2, 3, 4, 5])}
    ds = {"split_scenes": {"train": [0, 1], "val": [2], "test": [3, 4, 5]}}
    keep, given = effects.split_frames(ds, metas, "test", (0.7, 0.15, 0.15), 0)
    assert sorted(m["scene"] for m in keep.values()) == [3, 4, 5] and given == ds["split_scenes"]
    keep, given = effects.split_frames({}, metas, "test", (0.7, 0.15, 0.15), 0)
    assert given == export.split_scenes(range(6), (0.7, 0.15, 0.15), 0)
    keep, given = effects.split_frames(ds, metas, None, (0.7, 0.15, 0.15), 0)
    assert len(keep) == 6 and given is None


def test_effects_transmission_uses_frame_objects():
    # glass ids differ per world: the frame's own objects decide what belongs to a door
    t = np.full((4, 4), 0.5, np.float32)
    seg = np.zeros((4, 4), np.uint8); seg[0, :] = 11; seg[1, :] = 13
    objs = {"door_a": {"id": 11, "class": 1}, "glass_a": {"id": 13, "class": None, "part_of": "door_a"}}
    t[1, :] = 0.1
    assert effects.object_transmission(t, seg, {"objects": objs})["door_a"] == pytest.approx(0.3)


# --- export: helper fields stay out of the COCO attributes ---
def test_export_drops_v4_helper_fields():
    for k in ("seen_from", "approach", "keep_out"):
        assert k in export.DROP
