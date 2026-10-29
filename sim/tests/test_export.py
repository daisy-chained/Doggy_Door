# Unit tests for sim/doorsim/export.py on a fake dataset folder (no Gazebo needed): python3 -m pytest sim/tests
import json, pathlib, sys
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import export

ATTRS_V1 = json.loads((SIM / "door_room" / "door_room_v1.attributes.json").read_text())
W, H = 640, 480


def frame_meta(scene, view, boxes):
    """boxes: {object name: (x_min, y_min, x_max, y_max) or None}, like generate.py writes them."""
    objects = json.loads(json.dumps(ATTRS_V1["objects"]))
    for name, o in objects.items():
        b = boxes.get(name)
        o["box"] = None if b is None else dict(zip(("x_min", "y_min", "x_max", "y_max"), b))
        o["box_3d"] = None
    objects["door_a"]["joint_pos"], objects["door_a"]["state"] = -0.5, "open"
    return {"classes": ATTRS_V1["classes"], "camera": {"world_position": [0, 0, 1], "world_rpy": [0, 0, 0]},
            "objects": objects, "scene": scene, "view": view, "glass_depth_mode": None}


def make_dataset(root, n_scenes=10, views=3):
    """Fake generator output: dataset.json + frames/NNNNNN/{rgb.png, meta.json}."""
    root.mkdir(parents=True)
    i = 0
    for s in range(n_scenes):
        for v in range(views):
            d = root / "frames" / f"{i:06d}"
            d.mkdir(parents=True)
            (d / "rgb.png").write_bytes(b"png")
            boxes = {"door_a": (100, 50, 300, 450), "handle_a": (280, 240, 296, 252),
                     "door_b": (500, 60, 640, 470)}  # door_b touches the right edge
            (d / "meta.json").write_text(json.dumps(frame_meta(s, v, boxes)))
            i += 1
    (root / "dataset.json").write_text(json.dumps({
        "world": "door_room_v1", "seed": 0, "git": {"commit": "abc", "dirty": False}, "frames": i,
        "attributes": ATTRS_V1, "camera_intrinsics": {"width": W, "height": H, "K": [500, 0, 320, 0, 500, 240, 0, 0, 1]}}))
    return root


@pytest.fixture
def exported(tmp_path):
    src = make_dataset(tmp_path / "ds")
    out = tmp_path / "export"
    export.export(src, out, seed=0)
    return src, out


# --- split ---
def test_split_scenes_by_ratio_and_seed():
    a = export.split_scenes(list(range(20)), (0.7, 0.15, 0.15), seed=1)
    assert [len(a[k]) for k in ("train", "val", "test")] == [14, 3, 3]
    assert sorted(a["train"] + a["val"] + a["test"]) == list(range(20))
    assert a == export.split_scenes(list(range(20)), (0.7, 0.15, 0.15), seed=1)
    assert a != export.split_scenes(list(range(20)), (0.7, 0.15, 0.15), seed=2)


def test_split_small_dataset_keeps_every_split():
    a = export.split_scenes([0, 1, 2], (0.7, 0.15, 0.15), seed=0)
    assert [len(a[k]) for k in ("train", "val", "test")] == [1, 1, 1]


def test_scene_never_in_two_splits(exported):
    _, out = exported
    seen = {}
    for split in ("train", "val", "test"):
        coco = json.loads((out / "coco" / f"{split}.json").read_text())
        for im in coco["images"]:
            assert seen.setdefault(im["scene"], split) == split


# --- YOLO ---
def test_yolo_layout_and_labels(exported):
    src, out = exported
    data = (out / "data.yaml").read_text()
    for line in ("train: images/train", "val: images/val", "test: images/test", "0: door", "1: handle"):
        assert line in data
    imgs = sorted((out / "images").glob("*/*.png"))
    assert len(imgs) == 30
    assert all(p.is_symlink() and p.resolve().parent.parent.parent == src.resolve() for p in imgs)
    label = out / "labels" / imgs[0].parent.name / (imgs[0].stem + ".txt")
    rows = [l.split() for l in label.read_text().splitlines()]
    assert len(rows) == 3  # glass_a (class null) and objects without a box are left out
    door_a = next(r for r in rows if r[0] == "0" and float(r[1]) < 0.5)
    assert [float(x) for x in door_a[1:]] == pytest.approx([200 / W, 250 / H, 200 / W, 400 / H])
    assert sum(r[0] == "1" for r in rows) == 1


# --- COCO ---
def test_coco_boxes_and_attributes(exported):
    _, out = exported
    coco = json.loads((out / "coco" / "train.json").read_text())
    assert {c["id"]: c["name"] for c in coco["categories"]} == {1: "door", 2: "handle"}
    im = coco["images"][0]
    assert (im["width"], im["height"]) == (W, H)
    assert (out / im["file_name"]).is_symlink()
    anns = [a for a in coco["annotations"] if a["image_id"] == im["id"]]
    by_obj = {a["object"]: a for a in anns}
    assert set(by_obj) == {"door_a", "handle_a", "door_b"}
    a = by_obj["door_a"]
    assert a["category_id"] == 1 and a["bbox"] == [100, 50, 200, 400] and a["area"] == 200 * 400 and a["iscrowd"] == 0
    assert a["attributes"]["opening"] == "push" and a["attributes"]["glass"] is True
    assert a["attributes"]["state"] == "open" and a["attributes"]["joint_pos"] == -0.5
    assert by_obj["handle_a"]["attributes"]["handle_type"] == "lever"
    assert "box" not in a["attributes"] and "id" not in a["attributes"]
    assert len({x["id"] for x in coco["annotations"]}) == len(coco["annotations"])


def test_size_and_truncated_flags(exported):
    _, out = exported
    coco = json.loads((out / "coco" / "train.json").read_text())
    by_obj = {a["object"]: a for a in coco["annotations"] if a["image_id"] == coco["images"][0]["id"]}
    assert by_obj["handle_a"]["size_px"] == [16, 12]
    assert by_obj["handle_a"]["truncated"] is False
    assert by_obj["door_b"]["truncated"] is True  # touches the right image edge


# --- export.json + guard ---
def test_export_info(exported):
    src, out = exported
    info = json.loads((out / "export.json").read_text())
    assert info["source"]["path"] == str(src.resolve()) and info["source"]["git"]["commit"] == "abc"
    assert info["seed"] == 0 and info["ratios"] == [0.7, 0.15, 0.15]
    assert sum(info["counts"][s]["images"] for s in ("train", "val", "test")) == 30
    assert info["counts"]["train"]["door"] == 2 * info["counts"]["train"]["images"]


def test_refuses_existing_export(exported):
    src, out = exported
    with pytest.raises(FileExistsError):
        export.export(src, out, seed=0)
