# Unit tests for sim/doorsim/thermal_fx.py (no Gazebo): smoke in the thermal image (LWIR extinction = k x visible,
# Madelief 2026-11-04 / Claude k = 0.1 with a 0.3 check), smoke gas radiating at its own temperature (room: the air;
# fire: a hot ceiling layer), sensor noise, LWIR transmission per object, and the dataset step.
import json, math, pathlib, sys
import numpy as np
import pytest
from PIL import Image

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import glass, thermal, thermal_fx

W, H = 64, 48  # thermal resolution in these tests (the RGB-D frame is twice as big)
F = 50.0
K_THERMAL = [F, 0.0, W / 2, 0.0, F, H / 2, 0.0, 0.0, 1.0]
CAM = {"world_position": [0.0, 0.0, 1.0], "world_rpy": [0.0, 0.0, 0.0]}  # looking +x, level
AIR = 293.15


def smoke(v, layer=None):
    return {"visibility_m": v, "layer": layer, "patch": 0.0, "patch_scale": 1.5, "seed": 1}


def test_no_smoke_only_noise():
    scene = np.full((H, W), AIR + 2.0)
    out = thermal_fx.apply(scene, np.full((H, W), 3.0), K_THERMAL, CAM, None, k=0.1, gas=None, rng=np.random.default_rng(0), netd=0.05)
    assert np.abs(out - scene).mean() == pytest.approx(0.05 * math.sqrt(2 / math.pi), rel=0.2)  # mean |N(0, NETD)|


def test_room_smoke_pulls_contrast_toward_the_air():
    scene = np.full((H, W), AIR + 2.0)
    depth = np.full((H, W), 3.0)  # planar z-depth: a wall 3 m ahead
    out = thermal_fx.apply(scene, depth, K_THERMAL, CAM, smoke(2.0), k=0.1, gas={"air": AIR, "layer": None},
                           rng=np.random.default_rng(0), netd=0.0)
    c = out[H // 2, W // 2]  # centre ray: path 3 m
    t = math.exp(-0.1 * 3 / 2.0 * 3.0)  # LWIR transmission: k x (3 / V) x distance
    want = (t * (AIR + 2) ** 4 + (1 - t) * AIR ** 4) ** 0.25  # grey-body radiance ~ T^4, smoke at air temperature
    assert c == pytest.approx(want, abs=0.02)
    assert AIR < c < AIR + 2.0


def test_k_scales_the_attenuation():
    scene, depth = np.full((H, W), AIR + 2.0), np.full((H, W), 3.0)
    gas = {"air": AIR, "layer": None}
    a = thermal_fx.apply(scene, depth, K_THERMAL, CAM, smoke(2.0), k=0.1, gas=gas, rng=np.random.default_rng(0), netd=0.0)
    b = thermal_fx.apply(scene, depth, K_THERMAL, CAM, smoke(2.0), k=0.3, gas=gas, rng=np.random.default_rng(0), netd=0.0)
    assert (b[H // 2, W // 2] - AIR) < (a[H // 2, W // 2] - AIR)


def test_hot_layer_glows():
    scene, depth = np.full((H, W), AIR), np.full((H, W), 3.0)
    layer = {"height": 1.2, "factor": 4.0, "width": 0.15}
    cold = thermal_fx.apply(scene, depth, K_THERMAL, CAM, smoke(2.0, layer), k=0.1, gas={"air": AIR, "layer": None},
                            rng=np.random.default_rng(0), netd=0.0)
    hot = thermal_fx.apply(scene, depth, K_THERMAL, CAM, smoke(2.0, layer), k=0.1, gas={"air": AIR, "layer": AIR + 200},
                           rng=np.random.default_rng(0), netd=0.0)
    assert cold[2, W // 2] == pytest.approx(AIR, abs=0.01)  # smoke as warm as the wall: nothing to see
    assert hot[2, W // 2] > AIR + 1  # upper rows look through the hot layer
    assert hot[2, W // 2] > hot[H - 3, W // 2]  # more than the rows looking down


def test_object_transmission_lwir():
    t_vis = np.full((H, W), 0.01)
    seg = np.zeros((H, W), np.uint16)
    seg[10:20, 10:20] = 11
    attrs = {"objects": {"door_a": {"id": 11, "class": 1}, "box_1": {"id": 100}}}
    got = thermal_fx.object_transmission(t_vis ** 0.1, seg, attrs)
    assert got["door_a"] == pytest.approx(0.01 ** 0.1)


# --- dataset step ---
def make_base(root, n=12):
    """Base dataset: frames with rgb/depth/depth_gt/seg/meta + thermal_room.png and thermal_fire.png (640 x 480 RGB-D,
    320 x 240 thermal scaled down here to 2W x 2H / W x H), per-world temperatures in thermal/<world>_<scenario>.json."""
    (root / "thermal").mkdir(parents=True)
    for sc, air, layer in (("room", AIR, None), ("fire", 330.0, 500.0)):
        (root / "thermal" / f"w0_{sc}.json").write_text(json.dumps({"air": air, "layer": layer, "models": {"door_a": air + 1}}))
    for i in range(n):
        f = root / "frames" / f"{i:06d}"
        f.mkdir(parents=True)
        Image.fromarray(np.zeros((2 * H, 2 * W, 3), np.uint8)).save(f / "rgb.png")
        d = glass.to_png16(np.full((2 * H, 2 * W), 3.0))
        Image.fromarray(d).save(f / "depth.png"); Image.fromarray(d).save(f / "depth_gt.png")
        seg = np.zeros((2 * H, 2 * W), np.uint16); seg[20:40, 20:40] = 11
        Image.fromarray(seg).save(f / "seg.png")
        for sc, base in (("room", AIR + 2), ("fire", 340.0)):
            Image.fromarray(thermal.to_png16(np.full((H, W), base))).save(f / f"thermal_{sc}.png")
        m = {"world": "w0", "scene": i, "view": 0, "camera": CAM, "objects": {"door_a": {"id": 11, "class": 1, "smoke_transmission": 0.02}},
             "effects": {"smoke": smoke(2.0) if i % 2 else None, "noise_gain": 1.0}}
        (f / "meta.json").write_text(json.dumps(m))
    (root / "dataset.json").write_text(json.dumps({"camera_intrinsics": {"width": 2 * W, "height": 2 * H, "K": [2 * F, 0, W, 0, 2 * F, H, 0, 0, 1]},
                                                   "split_scenes": {"train": list(range(n)), "val": [], "test": []}}))
    return root


def test_apply_dataset_mix(tmp_path):
    src = make_base(tmp_path / "base")
    out = tmp_path / "th"
    thermal_fx.apply_dataset(src, out, scenario="mix", k=0.1, seed=0)
    ds = json.loads((out / "dataset.json").read_text())
    assert ds["thermal"]["scenario"] == "mix" and ds["thermal"]["k"] == 0.1
    assert ds["split_scenes"] == json.loads((src / "dataset.json").read_text())["split_scenes"]
    seen = set()
    for f in sorted((out / "frames").iterdir()):
        assert (f / "rgb.png").is_symlink() and (f / "depth.png").is_symlink()
        th = thermal.from_png16(np.asarray(Image.open(f / "thermal.png")))
        assert th.shape == (H, W)
        m = json.loads((f / "meta.json").read_text())
        sc = m["effects"]["thermal"]["scenario"]
        seen.add(sc)
        assert m["effects"]["thermal"]["k"] == 0.1
        o = m["objects"]["door_a"]
        if m["effects"]["smoke"]:
            assert o["smoke_transmission_vis"] == 0.02  # the visible value is kept
            assert o["smoke_transmission"] > 0.02  # LWIR sees much further
    assert seen == {"room", "fire"}


def test_apply_dataset_fixed_scenario(tmp_path):
    src = make_base(tmp_path / "base")
    thermal_fx.apply_dataset(src, tmp_path / "th", scenario="fire", k=0.3, seed=0)
    for f in (tmp_path / "th" / "frames").iterdir():
        assert json.loads((f / "meta.json").read_text())["effects"]["thermal"]["scenario"] == "fire"


def test_refuses_existing_output(tmp_path):
    src = make_base(tmp_path / "base", n=2)
    thermal_fx.apply_dataset(src, tmp_path / "th", scenario="room", k=0.1, seed=0)
    with pytest.raises(FileExistsError):
        thermal_fx.apply_dataset(src, tmp_path / "th", scenario="room", k=0.1, seed=0)
