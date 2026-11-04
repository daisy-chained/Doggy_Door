# Unit tests for thermal input to the detector (Madelief 2026-11-04: thermal-only detector + late fusion with RGB and
# dha): train/.venv/bin/python -m pytest train/tests
import json, pathlib, sys
import numpy as np
import pytest
from PIL import Image

TRAIN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))
import score_fusion
from doordet import config, data, fusion

W, H = 64, 48  # RGB-D frame; thermal is half of it (W / 2 x H / 2), like 640 x 480 vs 320 x 240


def make_export(root):
    """Base frame (rgb) -> thermal_fx frame (rgb.png a symlink to the base, own thermal.png) -> export image (a symlink
    to the thermal_fx rgb.png), like export.py on a thermal_fx dataset."""
    base = root / "base" / "frames" / "000000"
    base.mkdir(parents=True)
    Image.fromarray(np.full((H, W, 3), 99, np.uint8)).save(base / "rgb.png")
    th = root / "th" / "frames" / "000000"
    th.mkdir(parents=True)
    (th / "rgb.png").symlink_to(base / "rgb.png")
    k = np.full((H // 2, W // 2), 29315, np.uint16)  # 293.15 K in 0.01 K
    k[:, W // 4:] = 29515  # right half 2 K warmer
    Image.fromarray(k).save(th / "thermal.png")
    (th / "meta.json").write_text(json.dumps({"camera": {}, "objects": {}, "scene": 0, "view": 0}))
    ex = root / "export"
    (ex / "images" / "train").mkdir(parents=True)
    (ex / "images" / "train" / "000000.png").symlink_to(th / "rgb.png")
    (ex / "coco").mkdir()
    (ex / "coco" / "train.json").write_text(json.dumps({"images": [{"id": 0, "file_name": "images/train/000000.png", "width": W, "height": H,
                                                                   "frame": "000000"}], "categories": [{"id": 1, "name": "door"}], "annotations": []}))
    (ex / "export.json").write_text(json.dumps({"source": {"path": str(root / "th")}}))
    return ex


def test_thermal_is_a_valid_input():
    assert "thermal" in config.INPUTS


def test_thermal_input_reads_the_thermal_frame_not_the_base(tmp_path):
    img, _ = data.CocoDetection(make_export(tmp_path), "train", input="thermal")[0]
    a = np.asarray(img)
    assert a.shape == (H, W, 3)  # scaled up to the RGB size, so the boxes match
    assert (a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all()
    assert a[:, : W // 2 - 2].mean() < 30 and a[:, W // 2 + 2:].mean() > 225  # contrast stretched: cold dark, warm bright


def test_encode_thermal_stretch_ignores_outliers():
    k = np.full((100, 100), 293.15)
    k[0, 0] = 1000.0  # one hot pixel must not flatten the rest
    k[50:, :] = 295.15
    out = data.encode_thermal(k)
    assert out.dtype == np.uint8
    assert out[60, 60] - out[10, 10] > 200


def test_fusion_takes_three_models():
    def p(scores):
        return {"boxes": np.array([[0, 0, 10, 10]] * len(scores), float), "scores": np.array(scores, float), "labels": np.zeros(len(scores), int)}
    out = fusion.wbf([p([0.9]), p([0.6]), p([])])
    assert out["scores"][0] == pytest.approx(0.5)


def test_score_fusion_cli_takes_n_runs():
    a = score_fusion.parse(["runs/rgb", "runs/dha", "runs/thermal", "exports/x", "--weights", "1", "1", "2"])
    assert a.runs == ["runs/rgb", "runs/dha", "runs/thermal"] and a.export == "exports/x" and a.weights == [1, 1, 2]
    b = score_fusion.parse(["runs/rgb", "runs/dha", "exports/x"])
    assert b.weights is None  # equal weights
    with pytest.raises(SystemExit):
        score_fusion.parse(["runs/rgb", "runs/dha", "exports/x", "--weights", "1"])  # one weight per run
