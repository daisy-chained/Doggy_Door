# Double doors, round 1 (Madelief 2026-11-13: "start with a then when it works B" + "2. a"): a pair of hinged leaves in one opening,
# reported as ONE door of type "double" (one door box); the leaf with the handle ("active" leaf) is the one the robot opens; hinge
# side = that leaf's hinge. Bi-parting sliding pairs come later. Technical choices = Claude decisions (research-based, project note):
# both leaves swing the same way, the other leaf is bolted shut (no handle on it), total width 1.2-1.6 m.
# No Gazebo needed: python3 -m pytest sim/tests/test_double_doors.py
#
# sim/doorsim/doors.py:
#   sample_door: NEW optional cfg["double"] = {"p": float, "width": [lo, hi]}. When the key exists, AFTER every other draw (also
#     after glazing): r = rng.random(), wd = rng.uniform(*width) (both always drawn, so the stream doesn't depend on the outcome);
#     a hinged door becomes double when force["door_type"] == "double", or when force has no door_type and r < p (a forced
#     "hinged" stays hinged: balanced worlds deal the type): spec["door_type"] = "double", spec["width"] = wd
#     (rounded to 3 decimals). Everything else is the hinged spec (hinge_y = side of the ACTIVE leaf's hinge, swing_x, handle_type,
#     glass, glazing). force={"door_type": "double"} draws a hinged door first. Without the key nothing changes (v7 / v8 hashes).
#   is_double(spec) -> bool. seen_from(spec, side) for a double door = the hinged formula (the active leaf).
#   _layout(spec, slot) for a double door: like hinged, plus leaves = 2; pw = (w - 0.03) / 2 (one leaf); leaf_y = [active leaf centre,
#     other leaf centre] = [y + hinge_y * w / 4, y - hinge_y * w / 4] (±0.01); hinge = the active leaf's hinge point at
#     y + hinge_y * (w - 0.02) / 2, hinge_b = the other leaf's at y - hinge_y * (w - 0.02) / 2 (same x, z as hinged);
#     lock_y = y + hinge_y * 0.09 (handle on the active leaf next to the meeting stiles); open / range as hinged.
#   attributes: door object door_type "double"; "joint" = the active leaf's hinge (name "hinge"); "joint_b" = {"name": "hinge_b",
#     "type": "revolute", "unit": "rad", "open": -open, "range": its range}; six knuckles hinge_<slot>_0..5 (ids 10 k + 5 + i,
#     class 3, three per leaf); handles as for hinged doors.
#   world_sdf: the door model has links "panel" and "panel_b", joints "hinge" and "hinge_b", position controllers on
#     /door_<slot>/cmd_pos and /door_<slot>/cmd_pos_b; a glass / partly glazed double door gets a pane per leaf (models glass_<slot>
#     and glass_<slot>_b, both label 10 k + 3, each held by its own leaf).
# train/doordet/props.py: HEADS door_type = ["hinged", "sliding", "double"]; labels(): door_type "double" passes through.
# sim/configs/worlds_v9.yaml + door_room_v9_6k.yaml: v8 + double doors (solid doubles in training; double + glass held out).
import hashlib, json, pathlib, sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest
import yaml

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(SIM.parent / "train"))
from doorsim import doors

V7 = yaml.safe_load((SIM / "configs" / "worlds_v7.yaml").read_text())
V8 = yaml.safe_load((SIM / "configs" / "worlds_v8.yaml").read_text())
V9 = yaml.safe_load((SIM / "configs" / "worlds_v9.yaml").read_text())


def _hash(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()


def test_old_configs_unchanged():
    rng = np.random.default_rng(0)
    ss = [doors.sample_door(rng, V7["door"], s) for _ in range(50) for s in doors.SLOTS]
    assert _hash(ss) == "6ed230f9820a81b1885bf24d7af794d256a6a629e30b7767e6551ee01e40c145"
    assert _hash(doors.make_worlds(V7)) == "022b7cb918e6f7851d6ddda85e8000d0279043ae1be266cc656539298fc6c7c1"
    assert all(d["door_type"] != "double" for w in doors.make_worlds(V8) for d in w["doors"].values())


def test_sample_double():
    rng = np.random.default_rng(0)
    ss = [doors.sample_door(rng, V9["door"], s) for _ in range(400) for s in doors.SLOTS]
    dbl = [s for s in ss if s["door_type"] == "double"]
    lo, hi = V9["door"]["double"]["width"]
    assert dbl and all(lo <= s["width"] <= hi for s in dbl)
    assert all(doors.is_double(s) and s["handle_type"] in V9["door"]["handle"] for s in dbl)
    assert all(s["hinge_y"] in (-1, 1) and s["swing_x"] in (-1, 1) for s in dbl)
    assert not any(s["door_type"] == "double" for s in ss if "slide_y" in s)
    assert all(doors.sample_door(np.random.default_rng(k), V9["door"], "a", force={"door_type": "hinged"})["door_type"] == "hinged" for k in range(40))
    f = doors.sample_door(np.random.default_rng(5), V9["door"], "a", force={"door_type": "double", "glass": False})
    assert f["door_type"] == "double" and not f["glass"]


def _double(hinge=1, swing=1, glass=False, glazing="none", width=1.4):
    s = doors.sample_door(np.random.default_rng(3), V9["door"], "a", force={"door_type": "double", "glass": glass, "handle_type": "lever"})
    s.update(hinge_y=hinge, swing_x=swing, glazing=glazing, width=width, door_stop=True)
    return s


def test_seen_from_like_hinged():
    for h in (-1, 1):
        for sw in (-1, 1):
            d = _double(h, sw)
            hd = dict(d, door_type="hinged")
            for side in ("front", "back"):
                assert doors.seen_from(d, side) == doors.seen_from(hd, side)


def test_layout_two_leaves():
    for h in (-1, 1):
        d = _double(hinge=h, width=1.4)
        g = doors._layout(d, "a")
        y = doors.SLOT_Y["a"]
        assert g["leaves"] == 2 and g["pw"] == pytest.approx((1.4 - 0.03) / 2)
        assert g["leaf_y"][0] == pytest.approx(y + h * 1.4 / 4, abs=0.01) and g["leaf_y"][1] == pytest.approx(y - h * 1.4 / 4, abs=0.01)
        assert g["hinge"][1] == pytest.approx(y + h * (1.4 - 0.02) / 2) and g["hinge_b"][1] == pytest.approx(y - h * (1.4 - 0.02) / 2)
        assert g["hinge"][0] == pytest.approx(g["hinge_b"][0]) and g["lock_y"] == pytest.approx(y + h * 0.09)


def _world(spec_a, spec_b=None):
    w = doors.sample_world(np.random.default_rng(0), V8)
    other = spec_b or dict(doors.sample_door(np.random.default_rng(9), V8["door"], "b", force={"door_type": "hinged", "glass": False, "handle_type": "lever"}), door_stop=True)
    w["doors"] = {"a": spec_a, "b": dict(other, slot="b")}
    w["doors_heldout"] = {k: doors.is_heldout(v, V9["heldout"]) for k, v in w["doors"].items()}
    w.pop("window", None)
    return w


def test_attributes_double():
    d = _double()
    w = _world(d)
    objs = doors.attributes(w, "door_room_v9_test")["objects"]
    door = objs["door_a"]
    assert door["door_type"] == "double" and door["joint"]["name"] == "hinge" and door["joint_b"]["name"] == "hinge_b"
    assert door["joint_b"]["open"] == pytest.approx(-door["joint"]["open"])
    knuckles = [f"hinge_a_{i}" for i in range(6)]
    assert all(objs[k]["class"] == 3 and objs[k]["id"] == 15 + i for i, k in enumerate(knuckles))
    assert objs["handle_a"]["class"] == 2 and objs["handle_a_back"]["class"] == 2
    ids = [o["id"] for o in objs.values()]
    assert len(ids) == len(set(ids)) and max(ids) <= 255


def _models(w):
    root = ET.fromstring(doors.world_sdf(w, "door_room_v9_test"))
    return {m.get("name"): m for m in root.find("world").findall("model")}


def _label(m):
    for p in m.findall("plugin"):
        if p.find("label") is not None:
            return int(p.find("label").text)


def test_world_sdf_double():
    for glass, glazing in ((False, "none"), (True, "none"), (False, "panel")):
        w = _world(_double(glass=glass, glazing=glazing))
        ms = _models(w)
        door = ms["door_a"]
        assert {l.get("name") for l in door.findall("link")} >= {"panel", "panel_b"}
        assert {j.get("name") for j in door.findall("joint")} >= {"hinge", "hinge_b"}
        topics = {p.find("topic").text for p in door.findall("plugin") if p.find("topic") is not None}
        assert {"/door_a/cmd_pos", "/door_a/cmd_pos_b"} <= topics
        for i in range(6):
            assert _label(ms[f"hinge_a_{i}"]) == 15 + i
        if glass or glazing != "none":
            assert _label(ms["glass_a"]) == _label(ms["glass_a_b"]) == 13
            kids = {p.find("child_model").text for p in door.findall("plugin") if p.find("child_model") is not None}
            assert {"glass_a", "glass_a_b"} <= kids
        else:
            assert "glass_a" not in ms and "glass_a_b" not in ms


def test_props_heads_double():
    pytest.importorskip("torch")
    from doordet import props
    assert props.HEADS["door"]["door_type"] == ["hinged", "sliding", "double"]
    lab = props.labels({"opening": "push", "hinge_side": "left", "door_type": "double", "glass": False}, "door")
    assert lab["door_type"] == "double" and lab["opening"] == "push" and lab["hinge_side"] == "left"


def test_v9_worlds():
    ws = doors.make_worlds(V9)
    normal = [d for w in ws if not w["heldout"] for d in w["doors"].values()]
    held = [d for w in ws if w["heldout"] for d in w["doors"].values()]
    assert any(d["door_type"] == "double" for d in normal)
    assert not any(d["door_type"] == "double" and d["glass"] for d in normal)            # double + glass = held out
    assert any(d["door_type"] == "double" and d["glass"] for d in held)
    assert {"door_type": "double", "glass": True} in V9["heldout"]
    for w in ws:                                                                          # the two doorways never overlap
        a, b = w["doors"]["a"], w["doors"]["b"]
        assert (doors.SLOT_Y["b"] - b["width"] / 2) - (doors.SLOT_Y["a"] + a["width"] / 2) >= 0.3
    g = yaml.safe_load((SIM / "configs" / "door_room_v9_6k.yaml").read_text())
    assert g["world"] == "door_room_v9" and g["worlds"] == "door_room/v9/worlds.json"
