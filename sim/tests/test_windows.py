# Windows (door room v7, Madelief 2026-11-08: "build those first", so the robot can tell that something is a window, not a door).
# Each world gets one fixed window in the door wall (x = WALL_X), placed in a free stretch of wall: never over a door opening + its
# trim + margin, never over a sliding door's slide path, inside the room. Window draws use their own rng stream, so the doors of a
# config with `window` are the same as without it (v7 = the v6 doors + a window each).
#   world["window"] = {"y", "width", "sill", "top", "frame_width", "frame_colour", "mullions" (none / vertical / cross),
#                      "handle" (None / "front" / "back"), "low" (bool: floor-length window, looks like a glass door)}
#   attributes: "window" id 31 class 4 (+ classes["4"] = "window"), "window_glass" id 33 class None part glass_pane part_of window
#   (detach / attach topics like door glass), "window_handle" id 32 class None part window_handle part_of window (if a handle).
#   world_sdf: the wall gets a hole for the window (wall boxes around it), a "window" model (frame, label 31) holding the
#   "window_glass" model (label 33, same glass look + visibility_flags 2 as door glass) by a DetachableJoint, and a handle model.
# No Gazebo needed: python3 -m pytest sim/tests/test_windows.py
import pathlib, re, sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest
import yaml

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import doors, export

V6 = yaml.safe_load((SIM / "configs" / "worlds_v6.yaml").read_text())
V7 = yaml.safe_load((SIM / "configs" / "worlds_v7.yaml").read_text())
WCFG = V7["window"]


@pytest.fixture(scope="module")
def w7():
    return doors.make_worlds(V7)


def test_v7_config_is_v6_plus_window():
    assert V7["out"] == "v7" and "window" not in V6
    same = {k: v for k, v in V7.items() if k not in ("out", "window")}
    assert same == {k: v for k, v in V6.items() if k != "out"}


def test_no_window_without_config():
    assert all("window" not in w for w in doors.make_worlds(V6))


def test_doors_same_as_v6(w7):
    w6 = doors.make_worlds(V6)
    assert [w["doors"] for w in w7] == [w["doors"] for w in w6]
    assert [w["split"] for w in w7] == [w["split"] for w in w6]
    assert doors.make_worlds(V7) == w7  # fixed seed


def _blocked(world, margin):
    """y spans the window must stay out of: door openings + trim + margin, sliding paths + margin."""
    spans = []
    for slot, spec in world["doors"].items():
        g = doors._layout(spec, slot)
        spans.append((g["y"] - g["w"] / 2 - g["tw"] - margin, g["y"] + g["w"] / 2 + g["tw"] + margin))
        if spec["door_type"] == "sliding":
            spans.append((g["path"][0] - margin, g["path"][1] + margin))
    return spans


def test_windows_sit_in_free_wall(w7):
    have = [w for w in w7 if w.get("window")]
    assert len(have) >= 0.9 * len(w7)
    m = WCFG["margin"]
    for w in have:
        win = w["window"]
        y0, y1 = win["y"] - win["width"] / 2 - win["frame_width"], win["y"] + win["width"] / 2 + win["frame_width"]
        assert -3.0 + m - 1e-9 <= y0 and y1 <= 3.0 - m + 1e-9
        for a, b in _blocked(w, m):
            assert y1 <= a + 1e-9 or y0 >= b - 1e-9, (w["name"], (y0, y1), (a, b))
        assert WCFG["min_width"] - 1e-9 <= win["width"] <= WCFG["width"][1] + 1e-9
        assert win["top"] <= doors.ROOM_HEIGHT - 0.1 + 1e-9 and win["top"] - win["sill"] >= 0.5
        assert WCFG["top"][0] - 1e-9 <= win["top"] <= WCFG["top"][1] + 1e-9
        lo = WCFG["sill"]["low" if win["low"] else "normal"]["z"]
        assert lo[0] - 1e-9 <= win["sill"] <= lo[1] + 1e-9
        assert win["mullions"] in ("none", "vertical", "cross") and win["handle"] in (None, "front", "back")


def test_window_mix(w7):
    have = [w["window"] for w in w7 if w.get("window")]
    low = sum(x["low"] for x in have) / len(have)
    assert 0.15 <= low <= 0.45  # config p 0.3
    assert {x["mullions"] for x in have} == {"none", "vertical", "cross"}
    assert {x["handle"] for x in have} == {None, "front", "back"}
    assert len({round(x["y"], 2) for x in have}) > 10  # not one fixed spot


def _with_window(w7):
    return next(w for w in w7 if w.get("window") and w["window"]["handle"])


def test_attributes_have_window(w7):
    w = _with_window(w7)
    a = doors.attributes(w, w["name"])
    o = a["objects"]
    assert a["classes"]["4"] == "window"
    assert o["window"]["id"] == 31 and o["window"]["class"] == 4
    for k in ("width", "sill", "top", "low", "mullions"):
        assert o["window"][k] == w["window"][k]
    g = o["window_glass"]
    assert g["id"] == 33 and g["class"] is None and g["part"] == "glass_pane" and g["part_of"] == "window"
    assert g["detach_topic"] == "/window_glass/detach" and g["attach_topic"] == "/window_glass/attach"
    h = o["window_handle"]
    assert h["id"] == 32 and h["class"] is None and h["part_of"] == "window"
    ids = [x["id"] for x in o.values()]
    assert len(ids) == len(set(ids))


def test_attributes_without_window_unchanged():
    w = doors.make_worlds(V6)[0]
    a = doors.attributes(w, w["name"])
    assert "4" not in a["classes"] and not any(n.startswith("window") for n in a["objects"])


def test_no_handle_no_handle_object(w7):
    w = next(w for w in w7 if w.get("window") and w["window"]["handle"] is None)
    assert "window_handle" not in doors.attributes(w, w["name"])["objects"]


# --- world file ---
def _model(root, name):
    m = [x for x in root.iter("model") if x.get("name") == name]
    assert len(m) == 1, name
    return m[0]


def _wall_boxes(root):
    """(y0, y1, z0, z1) of the back wall's collision boxes (wall frame: model pose x = WALL_X, y / z as world)."""
    out = []
    for c in _model(root, "back_wall").iter("collision"):
        x, y, z = map(float, c.find("pose").text.split()[:3])
        sx, sy, sz = map(float, c.find("geometry/box/size").text.split())
        out.append((y - sy / 2, y + sy / 2, z - sz / 2, z + sz / 2))
    return out


def _covered(boxes, y, z):
    return any(y0 - 1e-6 <= y <= y1 + 1e-6 and z0 - 1e-6 <= z <= z1 + 1e-6 for y0, y1, z0, z1 in boxes)


def test_world_sdf_has_window_and_hole(w7):
    for w in [x for x in w7 if x.get("window")][:12]:
        win = w["window"]
        root = ET.fromstring(doors.world_sdf(w, w["name"]).split("?>", 1)[1])
        boxes = _wall_boxes(root)
        y0, y1 = win["y"] - win["width"] / 2, win["y"] + win["width"] / 2
        for yy in np.linspace(y0 + 0.01, y1 - 0.01, 5):  # the opening is free
            for zz in np.linspace(max(win["sill"], 0.0) + 0.01, win["top"] - 0.01, 5):
                assert not _covered(boxes, yy, zz), (w["name"], yy, zz)
        fy0, fy1 = y0 - win["frame_width"] - 0.03, y1 + win["frame_width"] + 0.03
        mid = (win["sill"] + win["top"]) / 2
        assert _covered(boxes, fy0, mid) and _covered(boxes, fy1, mid)  # wall beside the window
        assert _covered(boxes, win["y"], min(win["top"] + win["frame_width"] + 0.03, doors.ROOM_HEIGHT - 0.01))  # above
        if win["sill"] > 0.15:
            assert _covered(boxes, win["y"], win["sill"] - win["frame_width"] - 0.03)  # below the sill
        m = _model(root, "window")
        assert "<label>31</label>" in ET.tostring(m, encoding="unicode")
        dj = [p for p in m.iter("plugin") if p.get("name") == "gz::sim::systems::DetachableJoint"]
        assert len(dj) == 1 and dj[0].find("child_model").text == "window_glass"
        assert dj[0].find("detach_topic").text == "/window_glass/detach" and dj[0].find("attach_topic").text == "/window_glass/attach"
        gl = ET.tostring(_model(root, "window_glass"), encoding="unicode")
        assert "<label>33</label>" in gl and "<transparency>" in gl and "<visibility_flags>2</visibility_flags>" in gl
        if win["handle"]:
            assert "<label>32</label>" in ET.tostring(_model(root, "window_handle"), encoding="unicode")
        else:
            assert not [x for x in root.iter("model") if x.get("name") == "window_handle"]


def test_world_sdf_without_window_unchanged():
    w = doors.make_worlds(V6)[0]
    text = doors.world_sdf(w, w["name"])
    assert 'name="window' not in text
    saved = SIM / "door_room" / "v6" / f"{w['name']}.sdf"
    if saved.exists():
        assert text == saved.read_text()


def test_doorway_class_after_window():
    assert export.doorway_id({1: "door", 2: "handle", 3: "hinge", 4: "window"}) == 5
