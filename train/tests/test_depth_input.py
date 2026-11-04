# Unit tests for depth input to the detector (Madelief 2026-11-04: depth-only detector, input = depth + height above floor
# + surface normal angle to gravity, plus a plain-depth comparison): train/.venv/bin/python -m pytest train/tests
import json, pathlib, sys
import numpy as np
import pytest
from PIL import Image

TRAIN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))
from doordet import config, data

W, H = 64, 48
F = 50.0
K = [F, 0.0, W / 2, 0.0, F, H / 2, 0.0, 0.0, 1.0]
CAM = {"world_position": [0.0, 0.0, 1.0], "world_rpy": [0.0, 0.0, 0.0]}  # 1 m above the floor, looking +x, level


def wall_and_floor():
    """Planar z-depth: a wall 3 m ahead, the floor (1 m below the camera) in the rows where it is closer."""
    v = np.arange(H) + 0.5
    floor = np.where(v > H / 2, 1.0 * F / np.maximum(v - H / 2, 1e-6), np.inf)
    return np.minimum(np.full((H, W), 3.0), floor[:, None]).astype(np.float32)


def make_export(root, depth):
    """Dataset folder (frames/NNNNNN/{rgb,depth,meta}) + an export whose images are symlinks to rgb.png, like export.py."""
    ds = root / "ds"
    for i in range(2):
        f = ds / "frames" / f"{i:06d}"
        f.mkdir(parents=True)
        Image.fromarray(np.full((H, W, 3), 99, np.uint8)).save(f / "rgb.png")
        Image.fromarray(np.where(np.isfinite(depth), np.round(depth * 1000), 0).astype(np.uint16)).save(f / "depth.png")
        (f / "meta.json").write_text(json.dumps({"camera": CAM, "objects": {}, "scene": i, "view": 0}))
    (ds / "dataset.json").write_text(json.dumps({"camera_intrinsics": {"width": W, "height": H, "K": K}}))
    ex = root / "export"
    for split in ("train", "val", "test"):
        (ex / "images" / split).mkdir(parents=True)
        coco = {"images": [], "categories": [{"id": 1, "name": "door"}], "annotations": []}
        for i in range(2):
            fn = f"images/{split}/{i:06d}.png"
            (ex / fn).symlink_to(ds / "frames" / f"{i:06d}" / "rgb.png")
            coco["images"].append({"id": i, "file_name": fn, "width": W, "height": H, "frame": f"{i:06d}"})
        (ex / "coco").mkdir(exist_ok=True)
        (ex / "coco" / f"{split}.json").write_text(json.dumps(coco))
    (ex / "export.json").write_text(json.dumps({"source": {"path": str(ds)}}))
    return ex


@pytest.fixture
def export_dir(tmp_path):
    return make_export(tmp_path, wall_and_floor())


def test_rgb_is_still_the_default(export_dir):
    img, _ = data.CocoDetection(export_dir, "train")[0]
    assert np.asarray(img)[0, 0].tolist() == [99, 99, 99]


def test_plain_depth_input(export_dir):
    img, _ = data.CocoDetection(export_dir, "train", input="depth")[0]
    a = np.asarray(img)
    assert img.mode == "RGB" and img.size == (W, H)
    assert (a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all()
    assert a[5, 5, 0] == pytest.approx(3.0 / data.DEPTH_MAX * 255, abs=1)  # wall at 3 m


def test_no_depth_is_zero(tmp_path):
    ex = make_export(tmp_path, np.zeros((H, W), np.float32))
    for kind in ("depth", "dha"):
        assert np.asarray(data.CocoDetection(ex, "train", input=kind)[0][0]).max() == 0


def test_depth_height_angle_input(export_dir):
    a = np.asarray(data.CocoDetection(export_dir, "train", input="dha")[0][0]).astype(float)
    wall, floor = (5, 30), (H - 3, 30)  # a pixel high on the wall, one on the floor near the camera
    assert a[wall][0] == pytest.approx(3.0 / data.DEPTH_MAX * 255, abs=1)
    # height above the floor: wall pixel at row 5 -> 1 + (24 - 5.5) / 50 * 3 = 2.11 m; floor 0 m
    assert a[wall][1] == pytest.approx((1 + (H / 2 - 5.5) / F * 3) / data.HEIGHT_MAX * 255, abs=3)
    assert a[floor][1] == pytest.approx(0, abs=3)
    # surface normal vs gravity: wall 90 deg, floor 0 deg (scaled 0..180 -> 0..255)
    assert a[wall][2] == pytest.approx(255 / 2, abs=6)
    assert a[floor][2] == pytest.approx(0, abs=6)


def test_config_input_key(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("export: /x\n")
    assert config.load(p)["input"] == "rgb"
    assert config.load(p, ["input=dha"])["input"] == "dha"
    with pytest.raises(ValueError):
        config.load(p, ["input=bogus"])
