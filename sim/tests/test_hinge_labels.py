# Unit tests for hinge labels, step B of the door property cascade (Madelief 2026-11-07, "A then B"): "is this a hinged door?
# then are its hinges on my side?" needs the detector to find hinges like it finds handles. In door room v5 (door_stop) the three
# hinge knuckles were visuals on the door panel (same Gazebo label as the door), so they had no boxes. Now:
#   - each knuckle is its own model hinge_<slot>_<i> (i = 0, 1, 2; one cylinder, same pose / size / metal as before), with its own
#     label 10 k + 5 + i (door a: 15, 16, 17; door b: 25, 26, 27), held on the panel by a DetachableJoint like the handles, so it
#     turns with the door; the panel link no longer has knuckle visuals.
#   - attributes: objects hinge_<slot>_<i> = {"id", "class": 3, "door": "door_<slot>", "heldout"}; classes gain "3": "hinge" (only in
#     worlds that have hinges; v4 worlds stay exactly as they were).
#   - Gazebo's 2d boxes are visible-only, so a hinge gets a box only from the side it is on (the pull side of a closed door).
#   - export: hinge = category 3 / YOLO row 2; --doorway takes the next free id (still 3 on datasets without hinges).
#   - thermal heats hinges like handles (metal); the radar treats them as metal and hides them from the see-through lidar.
# python3 -m pytest sim/tests/test_hinge_labels.py
import copy, json, pathlib, sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest
import yaml
from PIL import Image

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import doors, export, radar, thermal


def v5_world(want=None):
    cfg = yaml.safe_load((SIM / "configs" / "worlds_v5.yaml").read_text())
    for w in doors.make_worlds(cfg):
        if want is None or want(w):
            return w
    raise AssertionError("no matching world")


def models(w):
    root = ET.fromstring(doors.world_sdf(w, "t").encode())
    return {m.get("name"): m for m in root.find("world").findall("model")}


def pose(el):
    return [float(v) for v in el.find("pose").text.split()]


def labels_of(m):
    return [int(p.find("label").text) for p in m.findall("plugin") if p.get("name") == "gz::sim::systems::Label"]


both_hinged = lambda w: all(d["door_type"] == "hinged" for d in w["doors"].values())
one_sliding = lambda w: sorted(d["door_type"] for d in w["doors"].values()) == ["hinged", "sliding"]


def test_each_knuckle_is_its_own_labelled_model_on_the_panel():
    w = v5_world(both_hinged)
    ms = models(w)
    for k, slot in enumerate(doors.SLOTS, 1):
        spec, g = w["doors"][slot], doors._layout(w["doors"][slot], slot)
        door = ms[f"door_{slot}"]
        panel = next(l for l in door.iter("link") if l.get("name") == "panel")
        assert not [v for v in panel.iter("visual") if v.get("name", "").startswith("knuckle")]   # moved off the panel
        attached = [p.find("child_model").text for p in door.findall("plugin") if p.get("name") == "gz::sim::systems::DetachableJoint"]
        zs = []
        for i in range(3):
            name = f"hinge_{slot}_{i}"
            m = ms[name]
            assert name in attached                                   # turns with the panel
            assert labels_of(m) == [10 * k + 5 + i]
            vis = list(m.iter("visual"))
            assert len(vis) == 1 and vis[0].find("geometry/cylinder") is not None
            assert float(vis[0].find("geometry/cylinder/radius").text) == pytest.approx(0.008)
            assert float(vis[0].find("geometry/cylinder/length").text) == pytest.approx(0.09)
            diffuse = [float(c) for c in vis[0].find("material/diffuse").text.split()[:3]]
            assert diffuse == pytest.approx(list(doors.METAL_RGB), abs=1e-3)
            # world position of the knuckle = where it sat on the panel before: on the hinge axis, at the hinge edge
            x, y, z = pose(m)[:3]
            vx, vy, vz = pose(vis[0])[:3] if vis[0].find("pose") is not None else (0, 0, 0)
            x, y, z = x + vx, y + vy, z + vz
            assert x == pytest.approx(g["xc"] + 0.02 * spec["swing_x"], abs=1e-6)
            assert y == pytest.approx(g["y"] + spec["hinge_y"] * (g["pw"] / 2 + 0.005), abs=1e-6)
            zs.append(z)
        assert sorted(zs) == pytest.approx([0.25, g["h"] / 2, g["h"] - 0.25], abs=1e-6)


def test_attributes_list_the_hinges_as_class_3():
    w = v5_world(one_sliding)
    a = doors.attributes(w, "t")
    assert a["classes"] == {"1": "door", "2": "handle", "3": "hinge"}
    hinged = next(s for s, d in w["doors"].items() if d["door_type"] == "hinged")
    sliding = next(s for s, d in w["doors"].items() if d["door_type"] == "sliding")
    k = doors.SLOTS.index(hinged) + 1
    for i in range(3):
        o = a["objects"][f"hinge_{hinged}_{i}"]
        assert o["id"] == 10 * k + 5 + i and o["class"] == 3 and o["door"] == f"door_{hinged}"
        assert o["heldout"] == w["doors_heldout"][hinged]
    assert not any(n.startswith(f"hinge_{sliding}_") for n in a["objects"])       # sliding doors have no hinges
    assert not any(n.startswith(f"hinge_{sliding}_") for n in models(w))
    ids = [o["id"] for o in a["objects"].values() if "id" in o]
    assert len(ids) == len(set(ids)) and max(ids) <= 255


def test_v4_worlds_unchanged():
    cfg = yaml.safe_load((SIM / "configs" / "worlds_v4.yaml").read_text())
    w = doors.make_worlds(cfg)[0]
    text = doors.world_sdf(w, "t")
    assert "hinge_a_" not in text and "hinge_b_" not in text and "knuckle" not in text
    a = doors.attributes(w, "t")
    assert a["classes"] == {"1": "door", "2": "handle"}
    assert not any(n.startswith("hinge_") for n in a["objects"])


def test_thermal_and_radar_treat_hinges_as_metal_handles():
    assert thermal.kind("hinge_a_0") == "handle"
    w = v5_world(both_hinged)
    a = doors.attributes(w, "t")
    mats = radar.materials(a)
    assert mats[a["objects"]["hinge_a_1"]["id"]] == "metal"
    hidden = ET.fromstring(radar.inject(doors.world_sdf(w, "t")).encode())
    for m in hidden.find("world").findall("model"):
        if m.get("name").startswith("hinge_"):
            assert all(v.find("visibility_flags") is not None and v.find("visibility_flags").text.strip() == "2" for v in m.iter("visual"))


def _dataset(tmp_path, classes, objects):
    src = tmp_path / "ds"
    f = src / "frames" / "000000"
    f.mkdir(parents=True)
    Image.fromarray(np.zeros((40, 60, 3), np.uint8)).save(f / "rgb.png")
    (f / "meta.json").write_text(json.dumps({"scene": 0, "view": 0, "objects": objects}))
    (src / "dataset.json").write_text(json.dumps({"attributes": {"classes": classes},
                                                  "camera_intrinsics": {"width": 60, "height": 40, "K": [50, 0, 30, 0, 50, 20, 0, 0, 1]},
                                                  "split_scenes": {"train": [0], "val": [], "test": []}}))
    return src


def test_export_writes_hinge_boxes_as_class_3(tmp_path):
    box = lambda x0, y0, x1, y1: {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1}
    objs = {"door_a": {"id": 11, "class": 1, "box": box(5, 5, 30, 38), "opening": "pull"},
            "handle_a": {"id": 12, "class": 2, "box": box(25, 20, 28, 22)},
            "hinge_a_0": {"id": 15, "class": 3, "door": "door_a", "box": box(5, 8, 6, 14)},
            "hinge_a_1": {"id": 16, "class": 3, "door": "door_a"}}          # not visible: no box, no row
    src = _dataset(tmp_path, {"1": "door", "2": "handle", "3": "hinge"}, objs)
    export.export(src, tmp_path / "out")
    coco = json.loads((tmp_path / "out" / "coco" / "train.json").read_text())
    assert {c["id"]: c["name"] for c in coco["categories"]} == {1: "door", 2: "handle", 3: "hinge"}
    hinge = [a for a in coco["annotations"] if a["category_id"] == 3]
    assert len(hinge) == 1 and hinge[0]["object"] == "hinge_a_0" and hinge[0]["attributes"]["door"] == "door_a"
    rows = (tmp_path / "out" / "labels" / "train" / "000000.txt").read_text().split("\n")
    assert sum(r.startswith("2 ") for r in rows) == 1


def test_doorway_takes_the_next_free_id():
    names = {1: "door", 2: "handle", 3: "hinge"}
    assert export.doorway_id(names) == 4
    assert export.doorway_id({1: "door", 2: "handle"}) == 3           # old datasets: unchanged
