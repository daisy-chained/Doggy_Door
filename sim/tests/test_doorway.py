# Unit tests for sim/doorsim/doorway.py: the doorway (frame opening) box, projected from the door room v4 geometry and
# cut to the part the depth shows (Madelief 2026-11-04: "doorway" class next to "door" = the panel): python3 -m pytest sim/tests
import json, pathlib, sys
import numpy as np
import pytest
import yaml
from PIL import Image

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import doors, doorway, export, frames, glass

W, H = 320, 240
F = 250.0
K = [F, 0.0, W / 2, 0.0, F, H / 2, 0.0, 0.0, 1.0]


def v4_meta(cam_pos=(0.0, -1.0, 1.0), yaw=0.0, side="front"):
    w = doors.sample_world(np.random.default_rng(0), yaml.safe_load((SIM / "configs" / "worlds_v4.yaml").read_text()))
    attrs = doors.attributes(w, "t")
    meta = frames.build_frame(attrs, [], [], None, {"world_position": list(cam_pos), "world_rpy": [0.0, 0.0, yaw]}, {"side": side})
    return meta, attrs


def expected_px():
    """door_a straight ahead at 2.95 m (seed 0 world: width 0.803, height 2.244, camera 1 m above the floor)."""
    meta, _ = v4_meta()
    o = meta["objects"]["door_a"]
    d = 2.95
    half = F * o["width"] / 2 / d
    return (W / 2 - half, H / 2 - F * (o["height"] - 1.0) / d, W / 2 + half, H / 2 + F * 1.0 / d)


def test_doorway_box_straight_ahead():
    meta, _ = v4_meta()
    got = doorway.box(meta, "door_a", K, np.zeros((H, W), np.float32))  # no depth returns: nothing in front of it
    x0, y0, x1, y1 = expected_px()
    b = got["box"]
    assert b["x_min"] == pytest.approx(x0, abs=3) and b["x_max"] == pytest.approx(x1, abs=3)
    assert b["y_min"] == pytest.approx(y0, abs=3) and b["y_max"] == pytest.approx(y1, abs=3)
    assert got["visible_fraction"] == pytest.approx(1.0, abs=0.02)


def test_closed_panel_in_the_doorway_still_counts_as_visible():
    meta, _ = v4_meta()
    got = doorway.box(meta, "door_a", K, np.full((H, W), 2.97, np.float32))  # panel 2 cm behind the wall face
    assert got["visible_fraction"] == pytest.approx(1.0, abs=0.02)


def test_something_in_front_cuts_the_box():
    meta, _ = v4_meta()
    depth = np.zeros((H, W), np.float32)
    depth[:, : W // 2] = 1.0  # a box 1 m in front of the camera hides the left half
    got = doorway.box(meta, "door_a", K, depth)
    assert got["box"]["x_min"] == pytest.approx(W / 2, abs=3)
    assert got["visible_fraction"] == pytest.approx(0.5, abs=0.05)


def test_partly_outside_the_image_is_clipped():
    meta, _ = v4_meta(cam_pos=(1.9, -1.0, 1.0))  # 1.05 m away: the doorway is taller than the image
    b = doorway.box(meta, "door_a", K, np.zeros((H, W), np.float32))["box"]
    assert b["y_min"] == 0 and b["y_max"] == H - 1


@pytest.mark.parametrize("yaw", [np.pi, np.pi / 2])
def test_not_in_view(yaw):
    meta, _ = v4_meta(yaw=yaw)
    assert doorway.box(meta, "door_a", K, np.zeros((H, W), np.float32)) is None


def test_hidden_completely():
    meta, _ = v4_meta()
    assert doorway.box(meta, "door_a", K, np.full((H, W), 0.5, np.float32)) is None


# --- exporter --doorway ---
def make_v4_dataset(root, n_scenes=6):
    meta, attrs = v4_meta()
    root.mkdir(parents=True)
    for s in range(n_scenes):
        d = root / "frames" / f"{s:06d}"
        d.mkdir(parents=True)
        (d / "rgb.png").write_bytes(b"png")
        Image.fromarray(glass.to_png16(np.zeros((H, W), np.float32))).save(d / "depth.png")
        m = json.loads(json.dumps(meta))
        m["objects"]["door_a"]["box"] = {"x_min": 150, "y_min": 15, "x_max": 160, "y_max": 204}
        m["objects"]["door_a"]["smoke_transmission"] = 0.02
        m.update(scene=s, view=0, glass_depth_mode=None)
        (d / "meta.json").write_text(json.dumps(m))
    (root / "dataset.json").write_text(json.dumps({"world": "door_room_v4", "seed": 0, "frames": n_scenes, "attributes": attrs,
                                                   "camera_intrinsics": {"width": W, "height": H, "K": K}}))
    return root


def test_export_adds_doorway_class(tmp_path):
    src = make_v4_dataset(tmp_path / "ds")
    out = tmp_path / "ex"
    export.export(src, out, doorway=True, ignore_below=0.05)
    coco = json.loads((out / "coco" / "test.json").read_text())
    assert {"id": 3, "name": "doorway"} in coco["categories"]
    dw = [a for a in coco["annotations"] if a["category_id"] == 3]
    assert dw and dw[0]["object"] == "door_a_doorway" and dw[0]["attributes"]["door"] == "door_a"
    assert dw[0]["attributes"]["smoke_transmission"] == 0.02  # the door's own value (same place, same smoke)
    assert dw[0]["iscrowd"] == 0  # test split keeps counting it
    train = json.loads((out / "coco" / "train.json").read_text())
    assert all(a["iscrowd"] == 1 for a in train["annotations"] if a["object"] == "door_a_doorway")  # ignored like its door
    # door_b's panel has no box (not in view) but part of its doorway is: still a doorway, never ignored (no transmission)
    assert any(a["object"] == "door_b_doorway" and a["iscrowd"] == 0 for a in train["annotations"])
    labels = (out / "labels" / "test").glob("*.txt")
    assert any(l.split()[0] == "2" for f in labels for l in f.read_text().splitlines())  # YOLO class 2 = doorway
    assert "  2: doorway" in (out / "data.yaml").read_text()
    assert json.loads((out / "export.json").read_text())["doorway"] is True


def test_export_without_doorway_is_unchanged(tmp_path):
    src = make_v4_dataset(tmp_path / "ds")
    export.export(src, tmp_path / "ex")
    coco = json.loads((tmp_path / "ex" / "coco" / "test.json").read_text())
    assert all(c["id"] != 3 for c in coco["categories"])
