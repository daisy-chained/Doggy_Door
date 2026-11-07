# Unit tests for sim/doorsim/radar.py (no Gazebo). Madelief 2026-11-04: radar proxy = a replayed wide depth + label render
# turned into radar returns (coarse angle cells, specular dropout at grazing angles, sparse points, range noise, no smoke)
# with a material layer (metal strong, wood / plaster / glass weak, glass partly passes); two radar types: single-chip
# (~15 deg azimuth) and cascade imaging (~1.5 deg).
import json, math, pathlib, sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import effects, radar, thermal

WORLD = SIM / "door_room" / "v4" / "door_room_v4_000.sdf"
ATTRS = json.loads((SIM / "door_room" / "v4" / "door_room_v4_000.attributes.json").read_text())
W, H = 240, 50                       # 120 x ~40 deg (square pixels: tan(20) / tan(60) x 240)
CAM = {"world_position": [0.0, 0.0, 1.0], "world_rpy": [0.0, 0.0, 0.0]}  # looking +x, level


def K():
    return radar.camera_K(W, H, math.radians(120))


def wall(x=3.0, gap=None):
    """Planar z-depth of a wall facing the radar at x; gap = (y_min, y_max) with no return (an open doorway into a big room)."""
    d = np.full((H, W), x)
    if gap:
        y = effects.world_points(d, K(), CAM)[..., 1]
        d[(y > gap[0]) & (y < gap[1])] = 0.0
    return d


def run(depth, seg=None, noglass=None, model="imaging", seed=0):
    seg = np.full((H, W), thermal.STRUCT_IDS["back_wall"], np.uint16) if seg is None else seg
    return radar.simulate(depth, seg, depth if noglass is None else noglass, K(), CAM, radar.MODELS[model],
                          radar.materials(ATTRS), np.random.default_rng(seed))


def test_models():
    assert radar.MODELS["single"]["az_res_deg"] == pytest.approx(15, abs=1)
    assert radar.MODELS["imaging"]["az_res_deg"] == pytest.approx(1.5, abs=0.5)
    for m in radar.MODELS.values():
        assert m["fov_deg"] == [120, 40] and m["max_points"] > 0


def test_materials():
    m = radar.materials(ATTRS)
    o = ATTRS["objects"]
    assert m[o["door_a"]["id"]] == "wood" and m[o["handle_a"]["id"]] == "metal" and m[o["glass_a"]["id"]] == "glass"
    assert m[thermal.STRUCT_IDS["back_wall"]] == "wall" and m[thermal.STRUCT_IDS["room_shell"]] == "wall"
    assert radar.REFLECT_DB["metal"] > radar.REFLECT_DB["wall"] and radar.REFLECT_DB["metal"] > radar.REFLECT_DB["wood"]


def test_specular_falloff():
    p = [radar.returned_power(radar.REFLECT_DB["wall"], a, 3.0) for a in (0, 30, 60, 85)]
    assert p[0] > p[1] > p[2] > p[3] > 0  # smooth walls reflect away at grazing angles; beyond the cutoff only the diffuse share is left
    assert radar.returned_power(0, 0, 2.0) > radar.returned_power(0, 0, 4.0)  # weaker with range


def test_wall_points_at_the_right_range_and_world_frame():
    out = run(wall(3.0))
    pts = out["points"]
    assert len(pts) > 0 and pts.shape[1] == 3 and len(out["power"]) == len(pts)
    assert np.abs(pts[:, 0] - 3.0).max() < 0.15  # world x of the wall, range noise only
    assert len(pts) <= radar.MODELS["imaging"]["max_points"]


def test_gap_is_empty_for_the_imaging_radar():
    pts = run(wall(3.0, gap=(-0.45, 0.45)))["points"]
    near = pts[np.abs(pts[:, 0] - 3.0) < 0.3]
    assert len(near) > 10
    assert not ((near[:, 1] > -0.3) & (near[:, 1] < 0.3)).any()  # no wall points inside the doorway


def test_single_chip_is_coarse():
    img = run(wall(3.0), model="imaging")["points"]
    one = run(wall(3.0), model="single")["points"]
    assert 0 < len(one) < len(img)
    az = np.degrees(np.arctan2(one[:, 1], one[:, 0]))
    assert len(np.unique(np.round(az / 15))) <= 9  # ~one point direction per 15 deg cell over 120 deg


def test_glass_passes_part_of_the_signal():
    seg = np.full((H, W), thermal.STRUCT_IDS["back_wall"], np.uint16)
    seg[:, W // 2 - 20: W // 2 + 20] = ATTRS["objects"]["glass_a"]["id"]
    behind = wall(3.0)
    behind[:, W // 2 - 20: W // 2 + 20] = 6.0  # the room behind the glass pane
    pts = run(wall(3.0), seg=seg, noglass=behind)["points"]
    assert (np.abs(pts[:, 0] - 6.0) < 0.2).any()  # returns from behind the glass
    assert (np.abs(pts[:, 0] - 3.0) < 0.2).any()


def test_metal_is_stronger_than_wall():
    seg = np.full((H, W), thermal.STRUCT_IDS["back_wall"], np.uint16)
    seg[23:27, 100:104] = ATTRS["objects"]["handle_a"]["id"]
    out = run(wall(3.0), seg=seg)
    hy = effects.world_points(wall(3.0), K(), CAM)[25, 102, 1]
    assert out["power"][np.abs(out["points"][:, 1] - hy) < 0.05].max() > np.median(out["power"])


def test_no_returns_without_depth_and_reproducible():
    assert len(run(np.zeros((H, W)))["points"]) == 0
    a, b = run(wall(3.0), seed=4), run(wall(3.0), seed=4)
    assert np.array_equal(a["points"], b["points"])


def test_inject_wide_cameras():
    root = ET.fromstring(radar.inject(WORLD.read_text()).encode())
    rig = {m.get("name"): m for m in root.find("world").findall("model")}["rig"]
    for typ, topic in (("depth_camera", "radar/depth"), ("segmentation", "radar/seg")):
        s = [x for x in rig.iter("sensor") if x.get("type") == typ and x.find("topic") is not None and x.find("topic").text == topic]
        assert len(s) == 1
        cam = s[0].find("camera")
        assert float(cam.find("horizontal_fov").text) == pytest.approx(math.radians(120))
        assert cam.find("image/width").text == "480" and cam.find("image/height").text == "100"
