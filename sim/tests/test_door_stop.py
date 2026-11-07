# Unit tests for the real push / pull cues on hinged doors (Madelief 2026-11-07, "get started on both"): in v4 a closed hinged
# panel sits centred in the wall with no stop and no visible hinges, so push and pull look the same from both sides. With
# spec["door_stop"] (worlds config key door_stop: true, door room v5):
#   - the panel is flush with the wall face on its swing side (the pull side): panel faces at WALL_X + 0.01 s .. WALL_X + 0.05 s
#     (s = swing_x), hinge axis on that face (WALL_X + 0.05 s);
#   - a door stop (jambs l / r + head t, 0.015 deep, 0.03 into the opening) on the other side (push side), against the panel:
#     x from WALL_X - 0.005 s to WALL_X + 0.01 s, so from the push side the stop sits 0.045 and the panel 0.06 behind the face;
#   - three metal hinge knuckles (cylinders) on the hinge axis, on the panel link (they turn with the door).
# Sliding doors and door_stop false (v4) stay as they were.
# python3 -m pytest sim/tests/test_door_stop.py
import copy, pathlib, sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest
import yaml

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import doors

WX = doors.WALL_X


def world(door_stop=True, seed=0, want=None):
    """A v4-config world whose door a is hinged (want = extra spec checks), door_stop set on every door spec."""
    cfg = yaml.safe_load((SIM / "configs" / "worlds_v4.yaml").read_text())
    for w in doors.make_worlds(cfg):
        a = w["doors"]["a"]
        if a["door_type"] == "hinged" and (want is None or want(a)):
            w = copy.deepcopy(w)
            for spec in w["doors"].values():
                spec["door_stop"] = door_stop
            return w
    raise AssertionError("no world with a hinged door a")


def models(w):
    root = ET.fromstring(doors.world_sdf(w, "t").encode())
    return {m.get("name"): m for m in root.find("world").findall("model")}


def pose(el):
    return [float(v) for v in el.find("pose").text.split()]


def box_extent(vis):
    """(lo, hi) per axis of a visual box in its link frame (no rotation)."""
    p = pose(vis)[:3]
    size = [float(v) for v in vis.find("geometry/box/size").text.split()]
    return [(p[i] - size[i] / 2, p[i] + size[i] / 2) for i in range(3)]


@pytest.mark.parametrize("swing", [-1, 1])
def test_panel_flush_with_the_swing_face_and_hinge_on_it(swing):
    w = world(want=lambda a: a["swing_x"] == swing)
    g = doors._layout(w["doors"]["a"], "a")
    assert g["xc"] == pytest.approx(WX + 0.03 * swing)
    assert g["hinge"][0] == pytest.approx(WX + 0.05 * swing)
    m = models(w)["door_a"]
    assert pose(m)[0] == pytest.approx(WX + 0.03 * swing)
    # handles sit on the moved panel's faces
    hf, hb = pose(models(w)["handle_a"]), pose(models(w)["handle_a_back"])
    assert hf[0] == pytest.approx(WX + 0.03 * swing - 0.02) and hb[0] == pytest.approx(WX + 0.03 * swing + 0.02)


def test_without_door_stop_v4_geometry_unchanged():
    w = world(door_stop=False)
    g = doors._layout(w["doors"]["a"], "a")
    assert g["xc"] == pytest.approx(WX)
    assert g["hinge"][0] == pytest.approx(WX + 0.02 * w["doors"]["a"]["swing_x"])
    text = doors.world_sdf(w, "t")
    assert "stop_a" not in text and "knuckle" not in text
    # an old spec without the key behaves as door_stop false
    old = copy.deepcopy(w)
    for spec in old["doors"].values():
        spec.pop("door_stop")
    assert doors.world_sdf(old, "t") == text


@pytest.mark.parametrize("swing", [-1, 1])
def test_door_stop_on_the_push_side_against_the_panel(swing):
    w = world(want=lambda a: a["swing_x"] == swing)
    spec = w["doors"]["a"]
    g = doors._layout(spec, "a")
    wall = models(w)["back_wall"]
    assert pose(wall)[0] == pytest.approx(WX)
    vis = {v.get("name"): v for v in wall.iter("visual")}
    stops = {k: vis[f"stop_a_{k}"] for k in ("l", "r", "t")}
    push_face = WX - 0.05 * swing
    for k, v in stops.items():
        (x0, x1), (y0, y1), (z0, z1) = box_extent(v)
        x0, x1 = x0 + WX, x1 + WX  # link frame -> world (wall model at WALL_X)
        near = x0 if swing > 0 else x1  # the stop face toward the push side
        far = x1 if swing > 0 else x0
        assert abs(near - push_face) == pytest.approx(0.045, abs=1e-6)  # stop 0.045 behind the push-side face
        assert far == pytest.approx(WX + 0.01 * swing, abs=1e-6)        # touching the panel's push-side face
        if k in ("l", "r"):
            assert z0 == pytest.approx(0, abs=1e-6) and z1 == pytest.approx(spec["height"], abs=1e-6)
            inner = min(abs(y0 - g["y"]), abs(y1 - g["y"]))
            outer = max(abs(y0 - g["y"]), abs(y1 - g["y"]))
            assert outer == pytest.approx(spec["width"] / 2, abs=1e-6)        # on the jamb
            assert inner == pytest.approx(spec["width"] / 2 - 0.03, abs=1e-6)  # 0.03 into the opening ...
            assert inner < g["pw"] / 2                                       # ... so it overlaps the panel edge
        else:
            assert z1 == pytest.approx(spec["height"], abs=1e-6) and z0 == pytest.approx(spec["height"] - 0.03, abs=1e-6)
            assert z0 < g["zc"] + g["ph"] / 2
    # the panel's push-side face is 0.06 behind the push-side wall face
    assert abs((g["xc"] - 0.02 * swing) - push_face) == pytest.approx(0.06)
    # the door b stop exists only when door b is hinged
    assert any(n.startswith("stop_b_") for n in vis) == (w["doors"]["b"]["door_type"] == "hinged")


def test_hinge_knuckles_on_the_axis_on_the_panel():
    w = world()
    spec = w["doors"]["a"]
    g = doors._layout(spec, "a")
    panel = next(l for l in models(w)["door_a"].iter("link") if l.get("name") == "panel")
    kn = [v for v in panel.iter("visual") if v.get("name", "").startswith("knuckle")]
    assert len(kn) == 3
    zs = []
    for v in kn:
        assert v.find("geometry/cylinder") is not None
        r = float(v.find("geometry/cylinder/radius").text)
        assert 0.005 <= r <= 0.012
        x, y, z = pose(v)[:3]  # panel link frame: origin at the panel centre (model xc, y, zc)
        assert x == pytest.approx(0.02 * spec["swing_x"], abs=1e-6)             # on the swing face (the axis)
        assert abs(y) >= g["pw"] / 2 and spec["hinge_y"] * y > 0                 # at the hinge edge
        zs.append(z + g["zc"])
        diffuse = [float(c) for c in v.find("material/diffuse").text.split()[:3]]
        assert diffuse == pytest.approx(list(doors.METAL_RGB), abs=1e-3)
    zs.sort()
    assert zs[0] == pytest.approx(0.25, abs=0.02) and zs[-1] == pytest.approx(spec["height"] - 0.25, abs=0.02)


def test_sliding_doors_get_no_stop_or_knuckles():
    w = world()
    w["doors"]["b"] = next(d for wo in doors.make_worlds(yaml.safe_load((SIM / "configs" / "worlds_v4.yaml").read_text()))
                           for d in wo["doors"].values() if d["door_type"] == "sliding" and d["slot"] == "b")
    w["doors"]["b"]["door_stop"] = True
    ms = models(w)
    names = [v.get("name") for v in ms["back_wall"].iter("visual")]
    assert not any(n.startswith("stop_b_") for n in names)
    assert not any(v.get("name", "").startswith("knuckle") for v in ms["door_b"].iter("visual"))
    assert doors._layout(w["doors"]["b"], "b")["xc"] == pytest.approx(doors._layout({**w["doors"]["b"], "door_stop": False}, "b")["xc"])


def test_v5_config_turns_it_on_and_writes_to_v5():
    cfg = yaml.safe_load((SIM / "configs" / "worlds_v5.yaml").read_text())
    v4 = yaml.safe_load((SIM / "configs" / "worlds_v4.yaml").read_text())
    assert cfg["door_stop"] is True and cfg["out"] == "v5"
    rest = {k: v for k, v in cfg.items() if k not in ("door_stop", "out")}
    assert rest == {k: v for k, v in v4.items() if k != "out"}  # same doors, same seed: only the cues differ
    ws = doors.make_worlds(cfg)
    assert all(d["door_stop"] for w in ws for d in w["doors"].values())
    assert ws[0]["name"] == "door_room_v5_000"  # world names follow the out version (v4 stays door_room_v4_*)
    assert doors.make_worlds(v4)[0]["name"] == "door_room_v4_000"
    # same random doors as v4 (door_stop must not change the draws)
    w4 = doors.make_worlds(v4)
    strip = lambda w: {s: {k: v for k, v in d.items() if k != "door_stop"} for s, d in w["doors"].items()}
    assert [strip(w) for w in ws] == [strip(w) for w in w4]
