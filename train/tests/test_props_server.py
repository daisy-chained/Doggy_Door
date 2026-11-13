# Property-model helper for the robot tour + the unfamiliar-door check (Madelief 2026-11-13: confidence option C, "A+b+c": the robot
# does not trust a guess on an unfamiliar-looking door and checks by looking / trying instead).
#   props_server.classify(model, meta, heads, T, dev, req, checker=None): unchanged output when checker is None; with a
#   familiar.Checker (train/doordet/familiar.py) the output also has "familiar": {"score": float, "flag": bool} for the crop, from
#   the same backbone features the heads use (one forward pass). "skip" answers stay as they are.
#   CLI: train/props_server.py RUN [--device D] [--familiar FAMILIAR_DIR]
import json, pathlib, sys
import numpy as np
import pytest
import torch
from PIL import Image

TRAIN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))
from doordet import props, familiar
import props_server
from test_familiar import run  # noqa: F401  (fixture: a tiny trained props run with held-out test doors)


@pytest.fixture
def req(tmp_path):
    img = np.zeros((120, 160, 3), np.uint8); img[10:110, 10:60] = (200, 0, 0)
    Image.fromarray(img).save(tmp_path / "rgb.png")
    Image.fromarray(np.full((120, 160), 2000, np.uint16)).save(tmp_path / "depth.png")
    return {"rgb": str(tmp_path / "rgb.png"), "depth": str(tmp_path / "depth.png"), "box": [10, 10, 60, 110],
            "K": [100, 0, 80, 0, 100, 60, 0, 0, 1], "camera": {"world_position": [0, 0, 0.4], "world_rpy": [0, 0, 0]}}


def test_classify_with_checker(tmp_path, run, req):
    model, meta = props.load(run)
    heads, T = list(meta["heads"]), meta.get("temperature") or {}
    familiar.build(run, tmp_path / "fam")
    ck = familiar.load(tmp_path / "fam")
    plain = props_server.classify(model, meta, heads, T, "cpu", req)
    out = props_server.classify(model, meta, heads, T, "cpu", req, checker=ck)
    assert "familiar" not in plain and set(out["familiar"]) == {"score", "flag"}
    assert isinstance(out["familiar"]["flag"], bool) and np.isfinite(out["familiar"]["score"])
    for h in heads:
        for c, p in plain[h].items():
            assert out[h][c] == pytest.approx(p, abs=1e-5)
    with torch.no_grad():
        x = props.crop_tensor(Image.open(req["rgb"]).convert("RGB"), props.crop_box((10, 10, 50, 100), meta["margin"], None, None), meta["imgsz"])
        f = model.backbone(x[None]).numpy()
    assert out["familiar"]["score"] == pytest.approx(float(ck.score(f)[0]), rel=1e-4, abs=1e-4)
    assert out["familiar"]["flag"] == bool(ck.flag(f)[0])


def test_classify_skip_small_unchanged(tmp_path, run, req):
    model, meta = props.load(run)
    familiar.build(run, tmp_path / "fam")
    small = dict(req, box=[10, 10, 12, 12])
    assert props_server.classify(model, meta, list(meta["heads"]), {}, "cpu", small, checker=familiar.load(tmp_path / "fam")) == {"skip": "small"}
