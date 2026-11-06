# Unit tests for the depth-aware door property classifier (Madelief 2026-11-06, option A: the dha crop next to the RGB crop, more
# margin so frame and wall are in view): train/.venv/bin/python -m pytest train/tests/test_props_depth.py
import json, pathlib, sys
import numpy as np
import pytest
import torch
from PIL import Image

TRAIN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))
sys.path.insert(0, str(TRAIN / "tests"))
from doordet import data, props
import test_depth_input as tdi  # dataset + export with frames (rgb.png, depth.png, meta.json) and symlinked images

DOOR = {"opening": "push", "hinge_side": "right", "door_type": "hinged", "glass": False}
BOX = [10, 5, 30, 40]  # xywh, inside the 64 x 48 test frame


@pytest.fixture
def export_dir(tmp_path):
    """tdi's export (wall + floor depth, flat rgb) with one door box per image; rgb gets a pattern so crops can be told apart."""
    ex = tdi.make_export(tmp_path, tdi.wall_and_floor())
    for f in sorted((tmp_path / "ds" / "frames").iterdir()):
        a = np.zeros((tdi.H, tdi.W, 3), np.uint8)
        a[:, :, 0] = np.arange(tdi.W)[None, :] * 3
        a[:, :, 1] = np.arange(tdi.H)[:, None] * 5
        Image.fromarray(a).save(f / "rgb.png")
    for split in ("train", "val", "test"):
        p = ex / "coco" / f"{split}.json"
        coco = json.loads(p.read_text())
        coco["categories"] = [{"id": 1, "name": "door"}, {"id": 2, "name": "handle"}]
        coco["annotations"] = [{"id": im["id"] + 1, "image_id": im["id"], "category_id": 1, "bbox": BOX, "area": BOX[2] * BOX[3],
                                "iscrowd": 0, "object": "door_a", "attributes": DOOR} for im in coco["images"]]
        p.write_text(json.dumps(coco))
    return ex


def ds(export_dir, input, jitter=None, seed=0, split="train"):
    return props.CropDataset(export_dir, split, "door", imgsz=32, margin=0.3, min_px=16, jitter=jitter, seed=seed, input=input)


# --- input images ---

@pytest.mark.parametrize("kind", ["rgb", "dha", "depth"])
def test_load_input_matches_the_detector_loader(export_dir, kind):
    coco = data.CocoDetection(export_dir, "train", input=kind)
    img = data.load_input(export_dir / coco.images[0]["file_name"], kind)
    assert np.array_equal(np.asarray(img), np.asarray(coco[0][0]))


# --- config ---

def test_input_defaults_to_rgb_and_is_checked(export_dir):
    assert props.resolve({"export": str(export_dir), "crop": "door"})["input"] == "rgb"
    for kind in ("dha", "rgb_dha"):
        assert props.resolve({"export": str(export_dir), "crop": "door", "input": kind})["input"] == kind
    with pytest.raises(ValueError):
        props.resolve({"export": str(export_dir), "crop": "door", "input": "lidar"})


def test_channels_per_input():
    assert props.CHANNELS == {"rgb": 3, "dha": 3, "rgb_dha": 6}


# --- crops ---

def test_crop_channels(export_dir):
    assert ds(export_dir, "rgb")[0][0].shape == (3, 32, 32)
    assert ds(export_dir, "dha")[0][0].shape == (3, 32, 32)
    assert ds(export_dir, "rgb_dha")[0][0].shape == (6, 32, 32)


def test_rgb_dha_is_the_rgb_crop_and_the_dha_crop_stacked(export_dir):
    both, rgb, dha = ds(export_dir, "rgb_dha")[0][0], ds(export_dir, "rgb")[0][0], ds(export_dir, "dha")[0][0]
    assert torch.allclose(both[:3], rgb) and torch.allclose(both[3:], dha)
    assert not torch.allclose(rgb, dha)


def test_rgb_and_dha_crops_stay_aligned_under_jitter(export_dir):
    # one jittered box per item, used for both images: same seed -> the rgb part equals a jittered rgb-only crop
    j = {"shift": 0.2, "scale": 0.3}
    both, rgb, dha = ds(export_dir, "rgb_dha", j, seed=3), ds(export_dir, "rgb", j, seed=3), ds(export_dir, "dha", j, seed=3)
    for i in range(len(both)):
        x = both[i][0]
        assert torch.allclose(x[:3], rgb[i][0]) and torch.allclose(x[3:], dha[i][0])


def test_dha_crop_normalised_to_about_unit_range(export_dir):
    # dha channels are depth / height / angle in 0..255, not photos: normalised with their own mean / std, not ImageNet's
    x = ds(export_dir, "dha")[0][0]
    raw = props.crop_box(BOX, 0.3, None, None)
    img = data.load_input(export_dir / "images/train/000000.png", "dha")
    s = (raw[2] - raw[0]) / 32
    ref = np.asarray(img.transform((32, 32), Image.AFFINE, (s, 0, raw[0], 0, s, raw[1]), Image.BILINEAR, fillcolor=(0, 0, 0)), np.float32) / 255
    assert torch.allclose(x, torch.from_numpy((ref - props.DHA_MEAN) / props.DHA_STD).permute(2, 0, 1), atol=1e-5)


# --- model ---

def test_widen_stem_keeps_rgb_weights_and_inits_extra_channels_from_their_mean():
    conv = torch.nn.Conv2d(3, 8, 3, bias=False)
    wide = props.widen_stem(conv, 6)
    assert wide.in_channels == 6 and wide.out_channels == 8 and wide.weight.shape == (8, 6, 3, 3)
    assert torch.equal(wide.weight[:, :3], conv.weight)
    assert torch.allclose(wide.weight[:, 3:], conv.weight.mean(1, keepdim=True).expand(-1, 3, -1, -1))
    assert props.widen_stem(conv, 3) is conv


@pytest.mark.parametrize("backbone", ["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
def test_model_takes_six_channels(backbone):
    m = props.Model(backbone, {"opening": ["push", "pull", "slide"]}, pretrained=False, in_ch=6).eval()
    with torch.no_grad():
        assert m(torch.zeros(2, 6, 64, 64))["opening"].shape == (2, 3)


# --- train / load / score on detector boxes ---

def cfg(tmp_path, export_dir, input):
    return props.resolve({"export": str(export_dir), "crop": "door", "input": input, "margin": 0.3, "backbone": "resnet18",
                          "pretrained": False, "imgsz": 32, "epochs": 1, "batch": 2, "device": "cpu", "workers": 0,
                          "out": str(tmp_path / "runs"), "mlflow": {"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}", "experiment": "test"}})


def test_train_load_and_score_rgb_dha(tmp_path, export_dir):
    r = props.train(cfg(tmp_path, export_dir, "rgb_dha"), run_name="unit")
    meta = json.loads((pathlib.Path(r["out"]) / "best" / "heads.json").read_text())
    assert meta["input"] == "rgb_dha"
    model, meta = props.load(r["out"])
    assert meta["input"] == "rgb_dha"
    with torch.no_grad():
        assert model(torch.zeros(1, 6, 32, 32))["opening"].shape == (1, 3)
    x, y, w, h = BOX
    preds = {i: {"boxes": np.array([[x, y, x + w, y + h]], float), "scores": np.array([0.9]), "labels": np.array([0])} for i in range(2)}
    s = props.score_on_boxes(r["out"], export_dir, "test", preds)
    assert s["matched"] == 2 and 0.0 <= s["acc_opening"] <= 1.0


def test_old_runs_without_input_load_as_rgb(tmp_path, export_dir):
    r = props.train(cfg(tmp_path, export_dir, "rgb"), run_name="unit")
    p = pathlib.Path(r["out"]) / "best" / "heads.json"
    meta = json.loads(p.read_text()); meta.pop("input", None); p.write_text(json.dumps(meta))
    model, meta = props.load(r["out"])
    assert meta["input"] == "rgb"
