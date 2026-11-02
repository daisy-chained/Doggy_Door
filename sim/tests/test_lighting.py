# Unit tests for lighting modes (sampling.lighting / flicker / scene_rngs), door_room_v3 (closed room + headlight)
# and the generator's scene subset (no Gazebo): python3 -m pytest sim/tests
import json, math, pathlib, sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest
import yaml

SIM = pathlib.Path(__file__).resolve().parents[1]
ROOM = SIM / "door_room"
sys.path.insert(0, str(SIM))
from doorsim import export, generate, sampling

CFG_V3 = yaml.safe_load((SIM / "configs" / "door_room_v3_2k.yaml").read_text())
LCFG = CFG_V3["lighting"]


# --- lighting modes ---
def test_lighting_modes_shares_and_levels():
    draws = [sampling.lighting(np.random.default_rng(i), LCFG) for i in range(900)]
    for mode, m in LCFG["modes"].items():
        got = [d for d in draws if d["mode"] == mode]
        assert abs(len(got) / 900 - m["p"]) < 0.06
        assert all(m["level"][0] <= d["level"] <= m["level"][1] for d in got)
        share_head = np.mean([d["headlight"] > 0 for d in got])
        assert abs(share_head - LCFG["headlight"]["p"][mode]) < 0.1
    assert all(d["headlight"] == 0 or LCFG["headlight"]["level"][0] <= d["headlight"] <= LCFG["headlight"]["level"][1] for d in draws)


def test_lighting_fixed_overrides_everything():
    cfg = {**LCFG, "fixed": {"level": 0.05, "headlight": 1.0}}
    for i in range(20):
        d = sampling.lighting(np.random.default_rng(i), cfg)
        assert d == {"mode": "fixed", "level": 0.05, "headlight": 1.0}
        assert sampling.flicker(np.random.default_rng(i), cfg, d["level"]) == 0.05  # no flicker at a hard limit


def test_flicker_only_dims():
    levels = [sampling.flicker(np.random.default_rng(i), LCFG, 0.5) for i in range(500)]
    dimmed = [l for l in levels if l < 0.5]
    assert all(l <= 0.5 for l in levels)
    assert abs(len(dimmed) / 500 - LCFG["flicker"]["p"]) < 0.06
    lo, hi = LCFG["flicker"]["depth"]
    assert all(0.5 * (1 - hi) - 1e-9 <= l <= 0.5 * (1 - lo) + 1e-9 for l in dimmed)


def test_lights_scaled_by_level():
    lights_cfg = CFG_V3["lights"]
    full = sampling.lights(np.random.default_rng(3), lights_cfg)
    half = sampling.lights(np.random.default_rng(3), lights_cfg, level=0.5)
    for n in full:
        assert half[n]["diffuse"] == pytest.approx([c * 0.5 for c in full[n]["diffuse"]])


def test_scene_rngs_independent_of_lighting_and_other_scenes():
    a_scene, a_light = sampling.scene_rngs(7, 12)
    b_scene, b_light = sampling.scene_rngs(7, 12)
    a_light.random(50)  # using the light stream more must not change the scene stream
    assert a_scene.random() == b_scene.random()
    c_scene, _ = sampling.scene_rngs(7, 13)
    assert sampling.scene_rngs(7, 12)[0].random() != c_scene.random()


# --- door_room_v3 ---
def world(name):
    return ET.parse(ROOM / f"{name}.sdf").getroot().find("world")


def test_v3_is_v2_closed_with_headlight():
    w3, w2 = world("door_room_v3"), world("door_room_v2")
    assert w3.get("name") == "door_room_v3"
    names3 = {m.get("name") for m in w3.iter("model")}
    assert {m.get("name") for m in w2.iter("model")} <= names3 and "room_shell" in names3
    scene = w3.find("scene")
    assert [float(x) for x in scene.find("ambient").text.split()][:3] == [0, 0, 0]
    assert [float(x) for x in scene.find("background").text.split()][:3] == [0, 0, 0]
    rig = next(m for m in w3.iter("model") if m.get("name") == "rig")
    head = [l for l in rig.iter("light") if l.get("name") == "headlight"]
    assert len(head) == 1 and head[0].get("type") == "spot"
    sun = next(l for l in w3.iter("light") if l.get("name") == "sun")
    assert sun.find("cast_shadows").text.strip() == "false"  # the ceiling would block it otherwise


def test_v3_shell_closes_the_room():
    """Ceiling above every camera / clutter position and walls around them: no sky in view."""
    shell = next(m for m in world("door_room_v3").iter("model") if m.get("name") == "room_shell")
    boxes = []
    for v in shell.iter("visual"):
        p = [float(x) for x in v.find("pose").text.split()][:3]
        s = [float(x) for x in v.find("geometry/box/size").text.split()]
        boxes.append([(c - d / 2, c + d / 2) for c, d in zip(p, s)])
    top = max(b[2][1] for b in boxes)
    ceiling = [b for b in boxes if b[2][0] >= 2.6 and b[0][0] <= -1.5 and b[0][1] >= 6.0 and b[1][0] <= -3.0 and b[1][1] >= 3.0]
    assert ceiling and top >= 2.7
    for y in (-3.0, 3.0):  # side walls reach the ceiling
        assert any(b[1][0] <= y <= b[1][1] and b[2][1] >= 2.6 and b[0][0] <= -1.5 and b[0][1] >= 6.0 for b in boxes)
    assert any(b[0][1] <= -1.5 and b[2][1] >= 2.6 for b in boxes)  # wall behind the cameras


def test_v3_attributes_equal_v2():
    a2 = json.loads((ROOM / "door_room_v2.attributes.json").read_text())
    a3 = json.loads((ROOM / "door_room_v3.attributes.json").read_text())
    assert a3["objects"] == a2["objects"] and a3["classes"] == a2["classes"]
    assert "door_room_v3" in a3["_note"]


# --- generator: which scenes ---
def test_scenes_to_render_all_or_one_split():
    assert generate.scenes_to_render(10) == list(range(10))
    test = generate.scenes_to_render(10, split="test", export_seed=0)
    assert test == sorted(export.split_scenes(range(10), (0.7, 0.15, 0.15), 0)["test"])


def test_v3_configs():
    assert CFG_V3["world"] == "door_room_v3" and CFG_V3["per_scene_rng"] is True
    assert CFG_V3["scenes"] == 100 and CFG_V3["views_per_scene"] == 20
    assert LCFG["fixed"] is None
    assert sum(m["p"] for m in LCFG["modes"].values()) == pytest.approx(1.0)
    small = yaml.safe_load((SIM / "configs" / "door_room_v3_small.yaml").read_text())
    assert small["world"] == "door_room_v3" and small["lighting"] == LCFG
