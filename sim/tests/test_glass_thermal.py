# Unit tests for sim/doorsim/glass_thermal.py and the --glass step of thermal_fx.py (no Gazebo).
#
# Why (Madelief 2026-11-08): the thermal state head must not lean on how cleanly the sim draws glass. Real glass in LWIR
# (8-14 um) is opaque, but what the camera sees on a pane is a mix (Claude, research-based, logged in the project note):
#   - its own emission:  emissivity e(theta) x T_glass^4      (uncoated glass e0 ~ 0.84-0.92 at normal incidence)
#   - a reflection:      R(theta) x T_refl^4                   (R0 = 1 - e0 - leak, rising to 1 at grazing: Schlick)
#                        T_refl = the surroundings (air + an offset + smooth noise) and the robot's own warm mirror image
#   - a leak:            leak x T_behind^4                     (real glass ~0; 0-10 % as a stress test)
# all mixed as radiance (T^4) and blended by the pane's area fraction per thermal pixel.
# Spec of the module (glass_thermal):
#   schlick(cos_t, r0) -> r0 + (1 - r0) (1 - cos_t)^5
#   pane_geometry(meta, pane) -> {"cos": |n . c| / |c|, "mirror": 2 (c . n) n} from the pane's door ("glass_a" -> "door_a")
#       box_3d (camera optical frame: x right, y down, z forward; normal = the box axis with the smallest size), None when
#       the door has no box_3d
#   draw(rng, cfg) -> {"emissivity", "leak", "env_K", "robot"} for one pane (uniform draws in the cfg ranges, robot = rng < p)
#   fractions(seg_full, meta, shape) -> {pane: (H, W) area fraction of that pane's seg id per thermal pixel}
#   apply(scene_K, behind_K, frac, geo, params, air_K, robot_K, K_thermal, rng, cfg) -> (H, W) K
# thermal_fx: apply_dataset(..., glass=cfg) / CLI --glass CONFIG.yaml; reads frames/<f>/thermal_<sc>_behind.png when a pane
# has leak > 0 (missing file -> the leak uses the air); records the per-pane draws in meta effects.thermal.glass.
import json, math, pathlib, sys
import numpy as np
import pytest
from PIL import Image

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import glass_thermal as gt

W, H = 64, 48
F = 50.0
K_TH = [F, 0.0, W / 2, 0.0, F, H / 2, 0.0, 0.0, 1.0]
AIR = 293.15
CFG = {"emissivity": [0.84, 0.92], "leak": [0.0, 0.02], "env_K": [-3.0, 3.0], "env_noise_K": 1.0, "env_cell": 8,
       "robot": {"p": 1.0, "radius_m": 0.25}}


def door_meta(center, q_xyzw=(0, 0, 0, 1), size=(0.8, 2.2, 0.04)):
    """meta with door_a (glass, box_3d in the camera frame; default: pane facing the camera, normal = z) and glass_a."""
    return {"objects": {"door_a": {"id": 11, "class": 1, "glass": True,
                                   "box_3d": {"center": list(center), "orientation_xyzw": list(q_xyzw), "size": list(size)}},
                        "glass_a": {"id": 13, "class": None}}}


def T4(t):
    return np.asarray(t, float) ** 4


# --- physics pieces ---

def test_schlick_ends():
    assert gt.schlick(1.0, 0.1) == pytest.approx(0.1)
    assert gt.schlick(0.0, 0.1) == pytest.approx(1.0)
    assert gt.schlick(0.5, 0.1) > 0.1


def test_pane_geometry_head_on():
    g = gt.pane_geometry(door_meta((0, 0, 3.0)), "glass_a")
    assert g["cos"] == pytest.approx(1.0)
    assert np.allclose(g["mirror"], [0, 0, 6.0])  # the camera's mirror image: straight ahead at twice the distance


def test_pane_geometry_slanted():
    # pane normal turned 60 deg about y: cos = 0.5 for a pane straight ahead
    a = math.radians(60)
    q = (0, math.sin(a / 2), 0, math.cos(a / 2))
    g = gt.pane_geometry(door_meta((0, 0, 3.0), q), "glass_a")
    assert g["cos"] == pytest.approx(0.5, abs=1e-6)


def test_pane_geometry_needs_a_box():
    m = door_meta((0, 0, 3.0)); del m["objects"]["door_a"]["box_3d"]
    assert gt.pane_geometry(m, "glass_a") is None


def test_draw_in_ranges():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = gt.draw(rng, CFG)
        assert 0.84 <= p["emissivity"] <= 0.92 and 0.0 <= p["leak"] <= 0.02 and -3 <= p["env_K"] <= 3 and p["robot"] is True


def test_fractions_area_average():
    seg = np.zeros((2 * H, 2 * W), np.uint16)
    seg[:, :W] = 13  # left half of the full-res frame is pane
    seg[:, W - 1] = 0  # one full-res column off: the thermal column W//2 - 1 is half pane
    f = gt.fractions(seg, door_meta((0, 0, 3)), (H, W))
    assert set(f) == {"glass_a"}
    assert f["glass_a"][:, : W // 2 - 1].min() == 1.0 and f["glass_a"][:, W // 2:].max() == 0.0
    assert f["glass_a"][0, W // 2 - 1] == pytest.approx(0.5)


# --- the mix ---

def _apply(scene, behind, frac, geo, params, robot_K=AIR + 10, cfg=None, seed=0):
    cfg = dict(CFG, env_noise_K=0.0, robot={"p": 1.0, "radius_m": 0.25}) if cfg is None else cfg
    return gt.apply(scene, behind, frac, geo, params, AIR, robot_K, K_TH, np.random.default_rng(seed), cfg)


def test_outside_the_pane_unchanged():
    scene = np.full((H, W), AIR + 5.0)
    frac = {"glass_a": np.zeros((H, W))}; frac["glass_a"][10:20, 10:20] = 1
    geo = {"glass_a": {"cos": 1.0, "mirror": np.array([0, 0, 1e6])}}  # robot image far away: tiny / off
    out = _apply(scene, scene, frac, geo, {"glass_a": {"emissivity": 0.9, "leak": 0.0, "env_K": 0.0, "robot": False}})
    assert np.allclose(out[frac["glass_a"] == 0], scene[frac["glass_a"] == 0])


def test_radiance_mix_head_on():
    tg, tb = AIR + 5.0, AIR + 40.0
    scene, behind = np.full((H, W), tg), np.full((H, W), tb)
    frac = {"glass_a": np.ones((H, W))}
    geo = {"glass_a": {"cos": 1.0, "mirror": np.array([0, 0, -1.0])}}  # mirror point behind the camera: no robot image
    p = {"emissivity": 0.9, "leak": 0.05, "env_K": 2.0, "robot": True}
    out = _apply(scene, behind, frac, geo, {"glass_a": p})
    r = 1 - 0.9 - 0.05  # R0 at normal incidence
    want = (0.9 * T4(tg) + r * T4(AIR + 2.0) + 0.05 * T4(tb)) ** 0.25
    assert np.allclose(out, want, atol=1e-6)


def test_grazing_view_is_mostly_reflection():
    scene = np.full((H, W), AIR + 20.0)
    frac = {"glass_a": np.ones((H, W))}
    p = {"glass_a": {"emissivity": 0.9, "leak": 0.0, "env_K": 0.0, "robot": False}}
    head_on = _apply(scene, scene, frac, {"glass_a": {"cos": 1.0, "mirror": np.array([0, 0, -1.0])}}, p)
    grazing = _apply(scene, scene, frac, {"glass_a": {"cos": 0.05, "mirror": np.array([0, 0, -1.0])}}, p)
    assert abs(grazing.mean() - AIR) < abs(head_on.mean() - AIR)  # closer to the reflected air than to the warm pane


def test_robot_mirror_image_is_a_warm_blob_where_the_mirror_point_projects():
    scene = np.full((H, W), AIR)
    frac = {"glass_a": np.ones((H, W))}
    geo = {"glass_a": {"cos": 1.0, "mirror": np.array([0.0, 0.0, 2.0])}}  # head on, 1 m away: image at 2 m, centre pixel
    p = {"glass_a": {"emissivity": 0.85, "leak": 0.0, "env_K": 0.0, "robot": True}}
    out = _apply(scene, scene, frac, geo, p, robot_K=AIR + 15)
    cx, cy = W // 2, H // 2
    r_px = F * 0.25 / 2.0  # ~6 px
    assert out[cy, cx] > AIR + 0.5
    assert out[cy, int(cx + 2 * r_px)] == pytest.approx(AIR, abs=1e-6)
    off = _apply(scene, scene, frac, geo, {"glass_a": dict(p["glass_a"], robot=False)}, robot_K=AIR + 15)
    assert np.allclose(off, AIR, atol=1e-6)


def test_coated_glass_mirrors_more():
    scene = np.full((H, W), AIR + 20.0)
    frac = {"glass_a": np.ones((H, W))}
    geo = {"glass_a": {"cos": 1.0, "mirror": np.array([0, 0, -1.0])}}
    plain = _apply(scene, scene, frac, geo, {"glass_a": {"emissivity": 0.9, "leak": 0.0, "env_K": 0.0, "robot": False}})
    coated = _apply(scene, scene, frac, geo, {"glass_a": {"emissivity": 0.15, "leak": 0.0, "env_K": 0.0, "robot": False}})
    assert coated.mean() < plain.mean() - 10  # mostly the reflected air


def test_env_noise_is_smooth_and_seeded():
    scene = np.full((H, W), AIR)
    frac = {"glass_a": np.ones((H, W))}
    geo = {"glass_a": {"cos": 0.05, "mirror": np.array([0, 0, -1.0])}}  # grazing: reflection dominates
    p = {"glass_a": {"emissivity": 0.9, "leak": 0.0, "env_K": 0.0, "robot": False}}
    cfg = dict(CFG, env_noise_K=2.0)
    a = _apply(scene, scene, frac, geo, p, cfg=cfg, seed=1)
    b = _apply(scene, scene, frac, geo, p, cfg=cfg, seed=1)
    assert np.array_equal(a, b) and a.std() > 0.1
    assert np.abs(np.diff(a, axis=1)).mean() < a.std()  # neighbours alike: smooth, not pixel noise


# --- thermal_fx --glass on a tiny dataset ---

def make_dataset(root):
    """One frame: 2W x 2H RGB-D, seg with a pane (id 13) in the middle, thermal_room.png (pane +5 K) and a behind file (+40 K)."""
    from doorsim import thermal
    ds = root / "src"
    fr = ds / "frames" / "000000"
    fr.mkdir(parents=True)
    (ds / "thermal").mkdir()
    Image.fromarray(np.zeros((2 * H, 2 * W, 3), np.uint8)).save(fr / "rgb.png")
    Image.fromarray(np.full((2 * H, 2 * W), 3000, np.uint16)).save(fr / "depth_gt.png")
    Image.fromarray(np.full((2 * H, 2 * W), 3000, np.uint16)).save(fr / "depth.png")
    seg = np.zeros((2 * H, 2 * W), np.uint16); seg[H // 2: 3 * H // 2, W // 2: 3 * W // 2] = 13
    Image.fromarray(seg).save(fr / "seg.png")
    k = np.full((H, W), AIR); k[H // 4: 3 * H // 4, W // 4: 3 * W // 4] = AIR + 5
    Image.fromarray(thermal.to_png16(k)).save(fr / "thermal_room.png")
    Image.fromarray(thermal.to_png16(np.full((H, W), AIR + 40))).save(fr / "thermal_room_behind.png")
    meta = door_meta((0, 0, 3.0)) | {"world": "w0", "scene": 0, "camera": {"world_position": [0, 0, 1], "world_rpy": [0, 0, 0]}}
    (fr / "meta.json").write_text(json.dumps(meta))
    (ds / "thermal" / "w0_room.json").write_text(json.dumps({"air": AIR, "layer": None, "models": {"rig": AIR + 12}}))
    (ds / "thermal" / "camera.json").write_text(json.dumps({"width": W, "height": H, "K": K_TH}))
    (ds / "dataset.json").write_text(json.dumps({"world": "w0", "camera_intrinsics": {"width": 2 * W, "height": 2 * H,
                                                 "K": [2 * F, 0, W, 0, 2 * F, H, 0, 0, 1]}, "split_scenes": {}}))
    return ds


def test_thermal_fx_glass_step(tmp_path):
    from doorsim import thermal, thermal_fx
    src = make_dataset(tmp_path)
    off = tmp_path / "off"; on = tmp_path / "on"
    thermal_fx.apply_dataset(src, off, "room", k=0.1, seed=0, workers=1, netd=0.0)
    thermal_fx.apply_dataset(src, on, "room", k=0.1, seed=0, workers=1, netd=0.0,
                             glass=dict(CFG, emissivity=[0.85, 0.85], leak=[0.1, 0.1], env_K=[0, 0], env_noise_K=0.0))
    a = thermal.from_png16(np.asarray(Image.open(off / "frames" / "000000" / "thermal.png")))
    b = thermal.from_png16(np.asarray(Image.open(on / "frames" / "000000" / "thermal.png")))
    pane = (slice(H // 4 + 1, 3 * H // 4 - 1), slice(W // 4 + 1, 3 * W // 4 - 1))
    assert np.allclose(a[0, :], b[0, :], atol=0.02)  # off the pane: unchanged
    assert not np.allclose(a[pane], b[pane], atol=0.05)  # on the pane: the mix (leak of the +40 K behind pulls it up)
    m = json.loads((on / "frames" / "000000" / "meta.json").read_text())
    g = m["effects"]["thermal"]["glass"]
    assert g["glass_a"]["emissivity"] == pytest.approx(0.85) and g["glass_a"]["leak"] == pytest.approx(0.1)
    m_off = json.loads((off / "frames" / "000000" / "meta.json").read_text())
    assert "glass" not in m_off["effects"]["thermal"]  # old behaviour without --glass


# --- thermal_render --behind: which frames need the render with the panes moved away ---

def test_panes_in_view():
    from doorsim import thermal_render
    m = door_meta((0, 0, 3.0))
    m["objects"]["door_a"]["visible_fraction"] = 0.4
    m["objects"]["door_b"] = {"id": 21, "class": 1, "glass": False, "visible_fraction": 1.0}
    m["objects"]["glass_c"] = {"id": 33, "class": None}
    m["objects"]["door_c"] = {"id": 31, "class": 1, "glass": True, "visible_fraction": 0.0}
    assert thermal_render.panes_in_view(m) == ["glass_a"]
    m["objects"]["door_a"]["visible_fraction"] = None  # unknown: render it
    assert thermal_render.panes_in_view(m) == ["glass_a"]
