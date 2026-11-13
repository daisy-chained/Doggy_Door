# More door variety (door room v8; Madelief 2026-11-13: confidence on new door types, option B, "All of them"). Models are ~81 %
# sure but ~43 % right on door types never seen in training (new COMBINATIONS of known properties). Training on more combinations
# helps a model generalise to unseen ones (Madan et al. 2022, Nature Machine Intelligence). Held-out combos stay unseen.
#   B1 more handle types: "pull_bar" (vertical bar on two standoffs), "flush_pull" (recessed pull, now also on hinged doors),
#      "push_plate" (a push plate on the face the door is pushed from and a pull bar on the face it is pulled from);
#      sliding doors can get a pull bar too.
#   B2 partly glazed doors: a SOLID door (glass False) with a vision "strip" or an upper glass "panel"; glass label stays False.
# Technical choices = Claude decisions (research-based, project note). No Gazebo needed: python3 -m pytest sim/tests/test_door_variety.py
#
# sim/doorsim/doors.py:
#   sample_door(rng, cfg, slot, force=None): NEW optional cfg keys, each drawn only when present (so a config without them gives
#     exactly the same doors and the same random stream as before; v7 fingerprint below):
#       cfg["handle"] may also name "pull_bar", "flush_pull", "push_plate" (hinged doors, drawn as before by weight);
#       cfg["sliding_handle"] = {type: weight}: drawn for sliding doors (else "flush_pull" without a draw);
#       cfg["glazing"] = {"none" | "strip" | "panel": weight}: drawn for every door AFTER all existing draws; glass doors always
#         get "none" (the draw is still made, so the stream does not depend on glass); spec["glazing"] set only when the key exists.
#     force may set "handle_type" for hinged doors as before (also the new types).
#   face_handle(spec, face) -> handle type on that face ("front" | "back"): "push_plate" doors -> "push_plate" on the face from
#     which seen_from(spec, face)["opening"] == "push", "pull_bar" on the other; every other door -> spec["handle_type"].
#   glass_rect(spec, g) -> (dy, dz, w, h): the glass pane in the panel's link frame (centre offset along y and z, size).
#     Full glass door: (0, 0.02, pw - 0.2, ph - 0.24) as before. "strip": a vertical strip, w 0.12..0.2, h >= 0.6, toward the lock
#     edge; "panel": the upper part, w = pw - 0.3, h 0.5..0.9 * ph / 2, above the middle. Partial panes leave >= 0.1 m of solid
#     panel on every side.
#   _panel_visuals(spec, g): partly glazed doors -> solid boxes around the pane only (no box covers the pane), in spec["colour"].
#   _glass_model: built for glass doors AND partly glazed doors, pane size / place from glass_rect (same label 10 k + 3, same
#     detachable joint and topics).
#   _handle_visuals(kind, bar): new kinds "pull_bar" and "push_plate" (own geometry, not the lever's).
#   _handle_model(..., face): uses face_handle(spec, face).
#   attributes(world, name): handle objects get handle_type = face_handle(spec, face); the door object gets "glazing" when the
#     spec has it; glass_<slot> exists for glass and partly glazed doors; the door's "glass" stays spec["glass"].
#   is_heldout: unchanged (glass + knob, sliding + glass); partly glazed doors are not glass doors.
# train/doordet/props.py: HEADS["handle"]["handle_type"] = ["lever", "knob", "flush_pull", "pull_bar", "push_plate"] (old ids kept).
# sim/configs/worlds_v8.yaml + door_room_v8_6k.yaml: the v8 worlds / render config (6000 frames, as v7).
import hashlib, json, pathlib, sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest
import yaml

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import doors

V7 = yaml.safe_load((SIM / "configs" / "worlds_v7.yaml").read_text())
V8 = yaml.safe_load((SIM / "configs" / "worlds_v8.yaml").read_text())
NEW_HINGED = {"lever", "knob", "pull_bar", "flush_pull", "push_plate"}


def specs(cfg, n=400, seed=0):
    rng = np.random.default_rng(seed)
    return [doors.sample_door(rng, cfg["door"], s) for _ in range(n) for s in doors.SLOTS]


def test_v7_doors_unchanged():
    rng = np.random.default_rng(0)
    ss = [doors.sample_door(rng, V7["door"], s) for _ in range(50) for s in doors.SLOTS]
    assert hashlib.sha256(json.dumps(ss, sort_keys=True).encode()).hexdigest() == "6ed230f9820a81b1885bf24d7af794d256a6a629e30b7767e6551ee01e40c145"
    w = doors.make_worlds(V7)
    assert hashlib.sha256(json.dumps(w, sort_keys=True, default=str).encode()).hexdigest() == "022b7cb918e6f7851d6ddda85e8000d0279043ae1be266cc656539298fc6c7c1"
    assert all("glazing" not in s for s in ss)


def test_v8_handles_and_glazing():
    ss = specs(V8)
    hinged = {s["handle_type"] for s in ss if s["door_type"] == "hinged"}
    sliding = {s["handle_type"] for s in ss if s["door_type"] == "sliding"}
    assert hinged == NEW_HINGED and sliding == {"flush_pull", "pull_bar"}
    assert all(s["glazing"] in ("none", "strip", "panel") for s in ss)
    assert all(s["glazing"] == "none" for s in ss if s["glass"])
    solid = [s["glazing"] for s in ss if not s["glass"]]
    assert {"none", "strip", "panel"} == set(solid)
    for kind in ("strip", "panel"):  # roughly the configured share among solid doors
        assert abs(solid.count(kind) / len(solid) - V8["door"]["glazing"][kind] / sum(V8["door"]["glazing"].values())) < 0.06


def test_force_new_handle():
    s = doors.sample_door(np.random.default_rng(1), V8["door"], "a", force={"door_type": "hinged", "handle_type": "push_plate"})
    assert s["door_type"] == "hinged" and s["handle_type"] == "push_plate"


def _hinged(handle, swing=1, hinge=1, glass=False, glazing="none"):
    s = doors.sample_door(np.random.default_rng(3), V8["door"], "a", force={"door_type": "hinged", "glass": glass, "handle_type": handle})
    s.update(swing_x=swing, hinge_y=hinge, glazing=glazing, door_stop=True)
    return s


def test_face_handle_push_plate_on_push_face():
    for swing in (-1, 1):
        s = _hinged("push_plate", swing=swing)
        for face in ("front", "back"):
            want = "push_plate" if doors.seen_from(s, face)["opening"] == "push" else "pull_bar"
            assert doors.face_handle(s, face) == want
        assert {doors.face_handle(s, f) for f in ("front", "back")} == {"push_plate", "pull_bar"}
    s = _hinged("lever")
    assert doors.face_handle(s, "front") == doors.face_handle(s, "back") == "lever"


def test_is_heldout_unchanged_for_new_values():
    combos = V8["heldout"]
    assert doors.is_heldout(_hinged("knob", glass=True), combos)
    assert not doors.is_heldout(_hinged("pull_bar", glass=True), combos)
    assert not doors.is_heldout(_hinged("knob", glazing="panel"), combos)


def _inside(r, pw, ph, border=0.1):
    dy, dz, w, h = r
    return abs(dy) + w / 2 <= pw / 2 - border + 1e-9 and abs(dz) + h / 2 <= ph / 2 - border + 1e-9


def test_glass_rect():
    full = _hinged("lever", glass=True)
    g = doors._layout(full, "a")
    assert doors.glass_rect(full, g) == pytest.approx((0, 0.02, g["pw"] - 0.2, g["ph"] - 0.24))
    st = _hinged("lever", glazing="strip"); g = doors._layout(st, "a")
    dy, dz, w, h = doors.glass_rect(st, g)
    assert 0.12 <= w <= 0.2 and h >= 0.6 and _inside((dy, dz, w, h), g["pw"], g["ph"])
    lock_side = -st["hinge_y"]                           # the lock edge is opposite the hinge (world y sign = link y sign here)
    assert dy * lock_side > 0
    pa = _hinged("lever", glazing="panel"); g = doors._layout(pa, "a")
    dy, dz, w, h = doors.glass_rect(pa, g)
    assert w == pytest.approx(g["pw"] - 0.3) and 0.5 * g["ph"] / 2 <= h <= 0.9 * g["ph"] / 2 and dz > 0
    assert _inside((dy, dz, w, h), g["pw"], g["ph"])


def _boxes(visuals):
    out = []
    for v in visuals:
        el = ET.fromstring(v)
        size = el.find("geometry/box/size")
        if size is None:
            continue
        p = [float(x) for x in el.find("pose").text.split()]
        sx, sy, sz = (float(x) for x in size.text.split())
        out.append((p[1], p[2], sy, sz))
    return out


@pytest.mark.parametrize("kind", ["strip", "panel"])
def test_partly_glazed_panel_leaves_the_pane_open(kind):
    s = _hinged("lever", glazing=kind); g = doors._layout(s, "a")
    dy, dz, w, h = doors.glass_rect(s, g)
    bs = _boxes(doors._panel_visuals(s, g))
    assert len(bs) >= 4
    for by, bz, bw, bh in bs:                                # no solid box overlaps the pane (shrunk by 1 mm)
        oy = min(by + bw / 2, dy + w / 2 - 1e-3) - max(by - bw / 2, dy - w / 2 + 1e-3)
        oz = min(bz + bh / 2, dz + h / 2 - 1e-3) - max(bz - bh / 2, dz - h / 2 + 1e-3)
        assert oy <= 0 or oz <= 0
    area = sum(bw * bh for _, _, bw, bh in bs)               # the solid parts cover the rest of the panel
    assert area == pytest.approx(g["pw"] * g["ph"] - w * h, rel=0.02)


def test_new_handle_visuals():
    lever = doors._handle_visuals("lever", 1)
    for kind in ("pull_bar", "push_plate"):
        v = doors._handle_visuals(kind, 1)
        assert v and v != lever and all(ET.fromstring(x).tag == "visual" for x in v)
    assert doors._handle_visuals("pull_bar", 1) != doors._handle_visuals("push_plate", 1)


def _world(slot_a, slot_b):
    w = doors.sample_world(np.random.default_rng(0), V8)
    w["doors"] = {"a": slot_a, "b": dict(slot_b, slot="b")}
    w["doors_heldout"] = {k: doors.is_heldout(v, V8["heldout"]) for k, v in w["doors"].items()}
    w.pop("window", None)
    return w


def test_attributes_and_world_for_new_doors():
    a_spec, b_spec = _hinged("push_plate", glazing="panel"), _hinged("pull_bar", glass=True)
    w = _world(a_spec, b_spec)
    attrs = doors.attributes(w, "door_room_v8_test")["objects"]
    assert attrs["door_a"]["glazing"] == "panel" and attrs["door_a"]["glass"] is False
    assert attrs["handle_a"]["handle_type"] == doors.face_handle(a_spec, "front")
    assert attrs["handle_a_back"]["handle_type"] == doors.face_handle(a_spec, "back")
    assert {attrs["handle_a"]["handle_type"], attrs["handle_a_back"]["handle_type"]} == {"push_plate", "pull_bar"}
    assert "glass_a" in attrs and "glass_b" in attrs
    root = ET.fromstring(doors.world_sdf(w, "door_room_v8_test"))
    models = {m.get("name"): m for m in root.find("world").findall("model")}
    pane = models["glass_a"].find("link/visual/geometry/box/size").text.split()
    g = doors._layout(a_spec, "a")
    _, _, gw, gh = doors.glass_rect(a_spec, g)
    assert [float(x) for x in pane][1:] == pytest.approx([gw, gh])
    door_children = {p.find("child_model").text for p in models["door_a"].findall("plugin") if p.find("child_model") is not None}
    assert "glass_a" in door_children


def test_v8_worlds():
    ws = doors.make_worlds(V8)
    assert len(ws) == V8["count"] + V8["heldout_count"]
    normal = [d for w in ws if not w["heldout"] for d in w["doors"].values()]
    assert not any(doors.is_heldout(d, V8["heldout"]) for d in normal)
    assert {d["handle_type"] for d in normal} >= {"pull_bar", "push_plate"}
    assert {d["glazing"] for d in normal} == {"none", "strip", "panel"}
    g = yaml.safe_load((SIM / "configs" / "door_room_v8_6k.yaml").read_text())
    assert g["world"] == "door_room_v8" and g["worlds"] == "door_room/v8/worlds.json"
    assert len(ws) * g["scenes_per_world"] * g["views_per_scene"] == 6000
