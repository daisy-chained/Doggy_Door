# Unit tests for sim/doorsim/effects.py (smoke + depth effects + camera noise on saved frames, no Gazebo): python3 -m pytest sim/tests
import json, math, pathlib, sys
import numpy as np
import pytest
import yaml
from PIL import Image

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import effects, export

W, H = 64, 48
K = [40.0, 0.0, 32.0, 0.0, 40.0, 24.0, 0.0, 0.0, 1.0]
CAM = {"world_position": [0.0, 0.0, 1.0], "world_rpy": [0.0, 0.0, 0.0]}  # looking along +x, z up
CFG = yaml.safe_load((SIM / "configs" / "effects_train.yaml").read_text())


def smoke(v, layer=None, patch=0.0):
    return {"visibility_m": v, "layer": layer, "patch": patch, "patch_scale": 1.5, "seed": 1}


# --- geometry ---
def test_world_points_centre_pixel_is_straight_ahead():
    depth = np.full((H, W), 2.0, np.float32)
    p = effects.world_points(depth, K, CAM)
    assert p.shape == (H, W, 3)
    assert p[24, 32] == pytest.approx([2.0, 0.0, 1.0], abs=1e-5)
    assert p[24, 0, 1] > 0 and p[0, 32, 2] > 1.0  # left in the image = +y (left), top = up


def test_world_points_no_value_is_nan():
    depth = np.zeros((H, W), np.float32)
    assert np.isnan(effects.world_points(depth, K, CAM)).all()


# --- smoke ---
def test_beta_visibility_definition_and_layer():
    assert effects.smoke_beta(np.array([1.0]), 3.0, None)[0] == pytest.approx(1.0)  # beta = 3 / V
    layer = {"height": 2.0, "factor": 5.0, "width": 0.05}
    b = effects.smoke_beta(np.array([0.5, 2.0, 3.0]), 3.0, layer)
    assert b[0] == pytest.approx(1.0, rel=1e-3) and b[2] == pytest.approx(5.0, rel=1e-3) and b[1] == pytest.approx(3.0, rel=1e-3)


def test_transmission_homogeneous_matches_exp():
    depth = np.full((H, W), 2.0, np.float32)
    t = effects.transmission(depth, K, CAM, smoke(4.0))
    dist = np.linalg.norm(effects.world_points(depth, K, CAM) - np.array(CAM["world_position"]), axis=-1)
    assert t == pytest.approx(np.exp(-3.0 / 4.0 * dist), rel=1e-3)
    assert t[24, 32] == pytest.approx(math.exp(-1.5), rel=1e-3)


def test_transmission_at_visibility_distance_is_five_percent():
    depth = np.full((H, W), 6.0, np.float32)
    assert effects.transmission(depth, K, CAM, smoke(6.0))[24, 32] == pytest.approx(0.05, abs=0.002)


def test_transmission_no_value_counts_as_far():
    depth = np.zeros((H, W), np.float32)
    t = effects.transmission(depth, K, CAM, smoke(4.0), far=12.0)
    assert t.max() <= math.exp(-3.0 / 4.0 * 12.0) + 1e-6


def test_transmission_layer_makes_upper_part_worse():
    depth = np.full((H, W), 3.0, np.float32)
    flat = effects.transmission(depth, K, CAM, smoke(8.0))
    layered = effects.transmission(depth, K, CAM, smoke(8.0, layer={"height": 1.2, "factor": 6.0, "width": 0.1}))
    assert layered[0, 32] < 0.5 * flat[0, 32]               # looking up into the smoke layer
    assert layered[H - 1, 32] == pytest.approx(flat[H - 1, 32], rel=0.05)  # looking down, below the layer


def test_transmission_patch_is_seeded_and_varies():
    depth = np.full((H, W), 3.0, np.float32)
    a = effects.transmission(depth, K, CAM, smoke(4.0, patch=0.4))
    b = effects.transmission(depth, K, CAM, smoke(4.0, patch=0.4))
    assert np.array_equal(a, b)
    assert a.std() > effects.transmission(depth, K, CAM, smoke(4.0)).std()


def test_add_smoke_limits():
    rgb = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    assert np.array_equal(effects.add_smoke(rgb, np.ones((H, W)), [0.75] * 3), rgb)
    full = effects.add_smoke(np.full((H, W, 3), 180, np.uint8), np.zeros((H, W)), [0.75] * 3)
    assert full.std() < 3 and full.mean() > 150  # all smoke, lit by the scene's own light


def test_add_smoke_is_dark_in_a_dark_scene():
    dark = effects.add_smoke(np.full((H, W, 3), 5, np.uint8), np.zeros((H, W)), [0.75] * 3)
    lit = effects.add_smoke(np.full((H, W, 3), 200, np.uint8), np.zeros((H, W)), [0.75] * 3)
    assert dark.mean() < 0.2 * lit.mean()  # smoke isn't self-lit: no light, no glow


def test_object_transmission_merges_parts():
    attrs = {"objects": {"door_a": {"id": 11, "class": 1}, "glass_a": {"id": 13, "class": None, "part_of": "door_a"},
                         "handle_b": {"id": 22, "class": 2}, "box_1": {"id": 100, "class": None}}}
    seg = np.zeros((H, W), np.uint8); seg[:, :10] = 11; seg[:, 10:20] = 13; seg[:, 30:] = 100
    t = np.zeros((H, W)); t[:, :10] = 0.2; t[:, 10:20] = 0.6
    out = effects.object_transmission(t, seg, attrs)
    assert out["door_a"] == pytest.approx(0.4) and out["handle_b"] is None and "box_1" not in out


# --- depth ---
def test_smoke_depth_clear_air_unchanged():
    d = np.full((H, W), 2.5, np.float32)
    assert np.array_equal(effects.smoke_depth(d, np.ones((H, W)), np.random.default_rng(0), CFG["depth"]), d)


def test_smoke_depth_thick_smoke_loses_or_shortens():
    d = np.full((200, 200), 3.0, np.float32)
    out = effects.smoke_depth(d, np.full((200, 200), 0.01), np.random.default_rng(0), {"t_lost": 0.3, "p_false": 0.3})
    lost, false = (out == 0).mean(), ((out > 0) & (out < 3.0)).mean()
    assert 0.6 < lost < 0.8 and 0.2 < false < 0.4  # ~all pixels hit; 30 % of them give a false close return
    assert out.max() <= 3.0


# --- camera noise ---
def test_noise_keeps_a_normal_image():
    rgb = np.full((H, W, 3), 128, np.uint8)
    out, gain = effects.sensor_noise(rgb, np.random.default_rng(0), CFG["noise"])
    assert abs(out.mean() - 128) < 25 and out.std() < 12 and gain >= 1.0


def test_noise_grows_as_light_drops_and_gain_is_capped():
    rng = np.random.default_rng(0)
    bright, _ = effects.sensor_noise(np.full((H, W, 3), 128, np.uint8), rng, CFG["noise"])
    dim, g_dim = effects.sensor_noise(np.full((H, W, 3), 25, np.uint8), rng, CFG["noise"])
    black, g_black = effects.sensor_noise(np.full((H, W, 3), 1, np.uint8), rng, CFG["noise"])
    assert dim.std() > 2 * bright.std()
    assert g_black == CFG["noise"]["max_gain"] and black.mean() < 100  # max gain reached: stays dark


# --- sampling ---
def test_sample_smoke_random_mode_share_and_range():
    s = [effects.sample_smoke(np.random.default_rng(i), CFG["smoke"]) for i in range(400)]
    on = [x for x in s if x is not None]
    assert abs(len(on) / 400 - CFG["smoke"]["p"]) < 0.08
    lo, hi = CFG["smoke"]["visibility_m"]
    assert all(lo <= x["visibility_m"] <= hi for x in on)
    assert any(x["layer"] for x in on) and any(x["layer"] is None for x in on)


def test_sample_smoke_fixed_visibility_keeps_the_rest_the_same():
    a = effects.sample_smoke(np.random.default_rng(5), CFG["smoke"], visibility=2.0)
    b = effects.sample_smoke(np.random.default_rng(5), CFG["smoke"], visibility=8.0)
    assert a["visibility_m"] == 2.0 and b["visibility_m"] == 8.0
    assert a["layer"] == b["layer"] and a["patch"] == b["patch"] and a["seed"] == b["seed"]
    assert effects.sample_smoke(np.random.default_rng(5), CFG["smoke"], visibility=math.inf) is None  # noise only


# --- dataset tool ---
ATTRS = {"classes": {"1": "door", "2": "handle"},
         "objects": {"door_a": {"id": 11, "class": 1}, "handle_a": {"id": 12, "class": 2}}}


def make_dataset(root, n_scenes=10, views=2):
    i = 0
    for s in range(n_scenes):
        for v in range(views):
            d = root / "frames" / f"{i:06d}"; d.mkdir(parents=True)
            Image.fromarray(np.full((H, W, 3), 150, np.uint8)).save(d / "rgb.png")
            Image.fromarray(np.full((H, W), 2000, np.uint16)).save(d / "depth.png")
            Image.fromarray(np.full((H, W), 2000, np.uint16)).save(d / "depth_gt.png")
            seg = np.zeros((H, W), np.uint8); seg[10:40, 10:30] = 11; seg[20:24, 26:29] = 12
            Image.fromarray(seg).save(d / "seg.png")
            objs = {"door_a": {"id": 11, "class": 1, "box": {"x_min": 10, "y_min": 10, "x_max": 29, "y_max": 39}},
                    "handle_a": {"id": 12, "class": 2, "box": {"x_min": 26, "y_min": 20, "x_max": 28, "y_max": 23}}}
            (d / "meta.json").write_text(json.dumps({"classes": ATTRS["classes"], "camera": CAM, "objects": objs, "scene": s, "view": v}))
            i += 1
    (root / "dataset.json").write_text(json.dumps({"world": "fake", "seed": 0, "frames": i, "attributes": ATTRS,
                                                   "camera_intrinsics": {"width": W, "height": H, "K": K}}))
    return root


def test_apply_dataset_random_mode(tmp_path):
    src = make_dataset(tmp_path / "src")
    out = effects.apply_dataset(src, tmp_path / "out", CFG)
    frames = sorted((tmp_path / "out" / "frames").iterdir())
    assert len(frames) == 20
    for f in frames:
        assert (f / "rgb.png").exists() and (f / "depth.png").exists()
        assert (f / "seg.png").is_symlink() and (f / "depth_gt.png").is_symlink()
        m = json.loads((f / "meta.json").read_text())
        assert "noise_gain" in m["effects"] and "smoke" in m["effects"]
        if m["effects"]["smoke"] is not None:
            assert 0.0 <= m["objects"]["door_a"]["smoke_transmission"] <= 1.0
    ds = json.loads((tmp_path / "out" / "dataset.json").read_text())
    assert ds["effects"]["source"] == str(src.resolve()) and ds["frames"] == 20
    assert out["frames"] == 20
    again = effects.apply_dataset(src, tmp_path / "out2", CFG)  # same seed = same images
    assert np.array_equal(np.asarray(Image.open(tmp_path / "out" / "frames" / "000003" / "rgb.png")),
                          np.asarray(Image.open(tmp_path / "out2" / "frames" / "000003" / "rgb.png")))


def test_apply_dataset_fixed_level_on_test_split_exports_as_test(tmp_path):
    src = make_dataset(tmp_path / "src")
    effects.apply_dataset(src, tmp_path / "v2m", CFG, visibility=2.0, split="test", export_seed=0)
    test_scenes = export.split_scenes(range(10), (0.7, 0.15, 0.15), 0)["test"]
    metas = [json.loads((f / "meta.json").read_text()) for f in sorted((tmp_path / "v2m" / "frames").iterdir())]
    assert {m["scene"] for m in metas} == set(test_scenes)
    assert all(m["effects"]["smoke"]["visibility_m"] == 2.0 for m in metas)
    ds = json.loads((tmp_path / "v2m" / "dataset.json").read_text())
    assert ds["split_scenes"]["test"] == test_scenes
    info = export.export(tmp_path / "v2m", tmp_path / "exp")  # exporter keeps the given split
    assert info["counts"]["test"]["images"] == len(metas) and info["counts"]["train"]["images"] == 0
    assert info["source"]["effects"]["visibility_m"] == 2.0  # the level travels into export.json (for scoring)
    ann = json.loads((tmp_path / "exp" / "coco" / "test.json").read_text())["annotations"][0]
    assert "smoke_transmission" in ann["attributes"]


def test_apply_dataset_refuses_existing(tmp_path):
    src = make_dataset(tmp_path / "src", n_scenes=2, views=1)
    effects.apply_dataset(src, tmp_path / "out", CFG)
    with pytest.raises(FileExistsError):
        effects.apply_dataset(src, tmp_path / "out", CFG)
