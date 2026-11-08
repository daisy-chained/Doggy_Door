# Unit tests for a thermal input on the door property heads (Madelief 2026-11-08: real-world fix for closed glass doors; glass is
# opaque in long-wave infrared, so the thermal crop shows a closed pane as a surface, an open one as the room behind):
# train/.venv/bin/python -m pytest train/tests/test_props_thermal.py
import pathlib, sys
import numpy as np
import pytest
import torch
from PIL import Image

TRAIN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))
sys.path.insert(0, str(TRAIN / "tests"))
from doordet import data, props
import test_depth_input as tdi
import test_props_depth as tpd


@pytest.fixture
def export_dir(tmp_path):
    """tpd's export plus a thermal.png per frame (uint16, 0.01 K, 4 x smaller like the 320 x 240 camera, with a gradient)."""
    ex = tpd.export_dir.__wrapped__(tmp_path)
    for f in sorted((tmp_path / "ds" / "frames").iterdir()):
        k = 29315 + np.arange(tdi.W // 4)[None, :] * 50 + np.arange(tdi.H // 4)[:, None] * 20
        Image.fromarray(k.astype(np.uint16)).save(f / "thermal.png")
    return ex


def test_thermal_inputs_are_known(export_dir):
    for kind in ("thermal", "rgb_thermal"):
        assert props.resolve({"export": str(export_dir), "crop": "door", "input": kind})["input"] == kind
    assert props.CHANNELS["thermal"] == 3 and props.CHANNELS["rgb_thermal"] == 6
    assert {"rgb", "dha", "rgb_dha"} <= set(props.CHANNELS)  # old inputs stay


def test_thermal_crop_is_the_crop_of_the_thermal_image(export_dir):
    d = tpd.ds(export_dir, "thermal")
    x = d[0][0]
    assert x.shape == (3, 32, 32)
    img = data.load_input(export_dir / d.items[0][0], "thermal")
    box = props.crop_box(tpd.BOX, 0.3, None, None)
    ref = props.crop_tensor(img, box, 32, props.DHA_MEAN, props.DHA_STD)
    assert torch.allclose(x, ref)
    assert not torch.allclose(x, tpd.ds(export_dir, "rgb")[0][0])


def test_rgb_thermal_is_rgb_and_thermal_stacked(export_dir):
    both, rgb, th = (tpd.ds(export_dir, k)[0][0] for k in ("rgb_thermal", "rgb", "thermal"))
    assert both.shape == (6, 32, 32)
    assert torch.allclose(both[:3], rgb) and torch.allclose(both[3:], th)


def test_train_with_thermal_input(export_dir, tmp_path):
    cfg = props.resolve({"export": str(export_dir), "crop": "door", "input": "rgb_thermal", "backbone": "resnet18", "pretrained": False,
                         "imgsz": 32, "epochs": 1, "batch": 4, "device": "cpu", "workers": 0, "out": str(tmp_path / "runs"),
                         "mlflow": {"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}", "experiment": "test"}})
    r = props.train(cfg, run_name="unit")
    model, meta = props.load(r["out"])[:2]
    assert meta["input"] == "rgb_thermal"


def test_dha_thermal_is_dha_and_thermal_stacked(export_dir):
    # depth shows where the leaf is (in the frame or swung), thermal shows a pane as a surface (Madelief 2026-11-08 follow-up:
    # thermal alone lost on closed solid doors, so the geometry has to come from depth)
    assert props.CHANNELS["dha_thermal"] == 6
    assert props.resolve({"export": str(export_dir), "crop": "door", "input": "dha_thermal"})["input"] == "dha_thermal"
    both, dha, th = (tpd.ds(export_dir, k)[0][0] for k in ("dha_thermal", "dha", "thermal"))
    assert both.shape == (6, 32, 32)
    assert torch.allclose(both[:3], dha) and torch.allclose(both[3:], th)
