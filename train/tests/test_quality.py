# Unit tests for adaptive late fusion (Madelief 2026-11-06, option A "quality rule"): each model's weight in the fusion follows a
# per-image "can this sensor see" score made from the model's own input image, no training.
# train/.venv/bin/python -m pytest train/tests/test_quality.py
import pathlib, sys
import numpy as np
import pytest

TRAIN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))
from doordet import fusion, quality


def scene(h=240, w=320, seed=0):
    """Indoor-looking test image: big flat blocks (walls / doors) with edges, mid grey, uint8 RGB."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 120.0)
    for _ in range(12):
        y, x = rng.integers(0, h - 40), rng.integers(0, w - 40)
        img[y:y + rng.integers(20, 120), x:x + rng.integers(20, 120)] = rng.uniform(30, 230)
    return np.repeat(img[..., None], 3, axis=2).clip(0, 255).astype(np.uint8)


def haze(img, t, air=190):
    return (img * t + air * (1 - t)).round().astype(np.uint8)


def dark(img, f, seed=0):
    noise = np.random.default_rng(seed).normal(0, 2, img.shape)
    return (img * f + noise).clip(0, 255).round().astype(np.uint8)


def depth_img(h=240, w=320):
    """Encoded depth input (channel 0 = depth, 0 = invalid): a smooth ramp, like a wall seen at an angle."""
    ch = np.tile(np.linspace(40, 200, w), (h, 1))
    return np.repeat(ch[..., None], 3, axis=2).round().astype(np.uint8)


# --- raw scores -------------------------------------------------------------------------------------------------------------

def test_rgb_scores_have_brightness_and_detail():
    s = quality.scores(scene(), "rgb")
    assert set(s) == {"bright", "detail"}
    assert 0 < s["bright"] < 1 and s["detail"] > 0


def test_rgb_detail_drops_in_smoke_brightness_does_not():
    clean, smoky = quality.scores(scene(), "rgb"), quality.scores(haze(scene(), 0.1), "rgb")
    assert smoky["detail"] < 0.3 * clean["detail"]
    assert smoky["bright"] >= 0.8 * clean["bright"]  # smoke is bright: brightness alone can't see it


def test_rgb_brightness_drops_in_the_dark():
    assert quality.scores(dark(scene(), 0.1), "rgb")["bright"] < 0.2 * quality.scores(scene(), "rgb")["bright"]


def test_rgb_detail_ignores_pixel_noise_and_smooth_gradients():
    # band-pass (blur 2 px minus blur 8 px): grain and the smoke layer's slow ceiling-to-floor gradient give little detail
    rng = np.random.default_rng(1)
    noise = np.repeat(rng.normal(128, 10, (240, 320))[..., None], 3, axis=2).clip(0, 255).astype(np.uint8)
    ramp = np.repeat(np.tile(np.linspace(60, 200, 240)[:, None], (1, 320))[..., None], 3, axis=2).astype(np.uint8)
    d = quality.scores(scene(), "rgb")["detail"]
    assert quality.scores(noise, "rgb")["detail"] < 0.3 * d
    assert quality.scores(ramp, "rgb")["detail"] < 0.1 * d


def test_thermal_uses_detail_only():
    s = quality.scores(scene(), "thermal")
    assert set(s) == {"detail"}
    flat = (scene().astype(float) * 0.2 + 100).round().astype(np.uint8)  # less contrast left
    assert quality.scores(flat, "thermal")["detail"] < 0.3 * s["detail"]


@pytest.mark.parametrize("kind", ["depth", "dha"])
def test_depth_clean_fraction(kind):
    img = depth_img()
    assert quality.scores(img, kind) == {"clean": pytest.approx(1.0, abs=0.02)}
    rng = np.random.default_rng(2)
    lost = img.copy()
    lost[rng.random(img.shape[:2]) < 0.7] = 0  # smoke: 70 % of pixels give no return
    assert quality.scores(lost, kind)["clean"] < 0.35
    speckle = img.copy()
    m = rng.random(img.shape[:2]) < 0.5  # smoke: false close returns scattered over the image (still "valid")
    speckle[m] = (speckle[m] * rng.uniform(0.1, 0.6, (m.sum(), 1))).astype(np.uint8)
    assert quality.scores(speckle, kind)["clean"] < 0.65


def test_scores_accept_pil_and_grey():
    from PIL import Image
    a = quality.scores(Image.fromarray(scene()), "rgb")
    b = quality.scores(scene()[..., 0], "rgb")
    ref = quality.scores(scene(), "rgb")
    assert a == pytest.approx(ref) and b == pytest.approx(ref)


def test_unknown_kind():
    with pytest.raises(ValueError):
        quality.scores(scene(), "lidar")


# --- reference + weight -----------------------------------------------------------------------------------------------------

def test_reference_is_90th_percentile_per_key():
    rows = [{"bright": b, "detail": 10 * b} for b in np.linspace(0, 1, 101)]
    ref = quality.reference(rows)
    assert ref == pytest.approx({"bright": 0.9, "detail": 9.0})


def test_weight_is_ratio_to_reference_clipped():
    ref = {"clean": 0.8}
    assert quality.weight({"clean": 0.4}, ref, "depth") == pytest.approx(0.5)
    assert quality.weight({"clean": 0.95}, ref, "dha") == 1.0  # better than the reference: never above 1
    assert quality.weight({"clean": 0.0}, ref, "depth") == pytest.approx(0.05)  # floor: a model never fully drops out
    assert quality.weight({"clean": 0.0}, ref, "depth", floor=0.0) == 0.0


def test_rgb_weight_is_the_worse_of_brightness_and_detail():
    ref = {"bright": 0.5, "detail": 0.04}
    assert quality.weight({"bright": 0.5, "detail": 0.01}, ref, "rgb") == pytest.approx(0.25)  # smoke
    assert quality.weight({"bright": 0.1, "detail": 0.04}, ref, "rgb") == pytest.approx(0.2)  # dark


def test_weight_power_sharpens():
    assert quality.weight({"clean": 0.4}, {"clean": 0.8}, "depth", power=2) == pytest.approx(0.25)


def test_blind_camera_gets_low_weight_end_to_end():
    ref = quality.reference([quality.scores(scene(seed=s), "rgb") for s in range(5)])
    assert quality.weight(quality.scores(scene(seed=9), "rgb"), ref, "rgb") > 0.6
    assert quality.weight(quality.scores(haze(scene(seed=9), 0.05), "rgb"), ref, "rgb") < 0.2
    assert quality.weight(quality.scores(dark(scene(seed=9), 0.05), "rgb"), ref, "rgb") < 0.2


# --- fusion with per-image weights ------------------------------------------------------------------------------------------

def p(boxes, scores, labels):
    return {"boxes": np.array(boxes, float).reshape(-1, 4), "scores": np.array(scores, float), "labels": np.array(labels, int)}


def test_fuse_runs_per_image_weights():
    # image 0: model A is blind (weight 0.05) and sees a false box; model B (weight 1) sees the true one
    a = {0: p([[50, 50, 60, 60]], [0.9], [0]), 1: p([[0, 0, 10, 10]], [0.8], [0])}
    b = {0: p([[0, 0, 10, 10]], [0.8], [0]), 1: p([[0, 0, 10, 10]], [0.6], [0])}
    out = fusion.fuse_runs([a, b], image_weights={0: [0.05, 1.0], 1: [1.0, 1.0]})
    s0 = dict(zip(map(tuple, out[0]["boxes"].tolist()), out[0]["scores"]))
    assert s0[(0, 0, 10, 10)] == pytest.approx(0.8 / 1.05)
    assert s0[(50, 50, 60, 60)] == pytest.approx(0.05 * 0.9 / 1.05)
    assert out[1]["scores"][0] == pytest.approx(0.7)  # equal weights as before


def test_fuse_runs_image_weights_combine_with_model_weights_and_default_to_them():
    a, b = {0: p([[0, 0, 10, 10]], [0.8], [0]), 1: p([[0, 0, 10, 10]], [0.8], [0])}, {0: p([], [], []), 1: p([], [], [])}
    out = fusion.fuse_runs([a, b], weights=[2, 1], image_weights={0: [0.5, 1.0]})
    assert out[0]["scores"][0] == pytest.approx(2 * 0.5 * 0.8 / (2 * 0.5 + 1))
    assert out[1]["scores"][0] == pytest.approx(2 * 0.8 / 3)  # image without own weights: model weights only


# --- CLI ----------------------------------------------------------------------------------------------------------------------

def test_cli_adaptive_flag():
    import score_fusion
    args = score_fusion.parse(["runs/a", "runs/b", "exports/x", "--adaptive"])
    assert args.adaptive and args.power == 1.0 and args.floor == 0.05
    args = score_fusion.parse(["runs/a", "runs/b", "exports/x", "--adaptive", "--power", "2", "--floor", "0"])
    assert args.power == 2.0 and args.floor == 0.0
    assert not score_fusion.parse(["runs/a", "runs/b", "exports/x"]).adaptive
