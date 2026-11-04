# Unit tests for sim/doorsim/radar_heatmap.py (no Gazebo): the radar's raw range x azimuth power map (before the
# detection threshold throws most of it away; RadarHD 2206.09273 learns from this) and a top-down (BEV) image in the
# robot's frame for a learned doorway detector (Madelief 2026-11-04, "just do both").
import json, math, pathlib, sys
import numpy as np
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import radar, radar_heatmap as rh, thermal
from test_radar import ATTRS, CAM, H, W, K, wall


def ra(depth, seg=None, model="imaging", seed=0):
    seg = np.full((H, W), thermal.STRUCT_IDS["back_wall"], np.uint16) if seg is None else seg
    return rh.ra_map(depth, seg, depth, K(), CAM, radar.MODELS[model], radar.materials(ATTRS), np.random.default_rng(seed))


def test_ra_map_peak_at_the_wall():
    m = ra(wall(3.0))
    assert m["power_db"].shape == (len(m["ranges"]), len(m["az_deg"]))
    assert m["az_deg"][0] == pytest.approx(-60, abs=1) and m["az_deg"][-1] == pytest.approx(60, abs=1)
    centre = m["power_db"][:, len(m["az_deg"]) // 2]
    assert m["ranges"][centre.argmax()] == pytest.approx(3.0, abs=0.15)  # slant range straight ahead


def test_no_returns_is_noise_only():
    m = ra(np.zeros((H, W)))
    assert m["power_db"].std() < 6 and m["power_db"].max() < ra(wall(3.0))["power_db"].max() - 10


def test_single_chip_is_blurrier_in_azimuth():
    seg = np.full((H, W), thermal.STRUCT_IDS["back_wall"], np.uint16)
    seg[20:30, 118:122] = ATTRS["objects"]["handle_a"]["id"]  # a small metal target straight ahead
    d = np.where(seg > 10, 2.0, 0.0)  # only the target returns
    def spread(model):
        p = ra(d, seg, model)["power_db"]
        row = p[p.max(1).argmax()]
        return (row > row.max() - 6).sum()  # azimuth bins within 6 dB of the peak
    assert spread("single") > 3 * spread("imaging")


def test_bev_wall_ahead():
    m = ra(wall(3.0))
    img = rh.bev([m], [CAM], CAM, size_m=8.0, res=0.05)
    Hb, Wb = img.shape
    assert (Hb, Wb) == (160, 160)
    col = img[:, Wb // 2]
    assert Hb - 1 - col.argmax() == pytest.approx(3.0 / 0.05, abs=4)  # robot at the bottom centre, forward = up


def test_bev_box_of_a_doorway():
    door = {"center": [3.0, 0.0, 1.0], "width": 0.9, "normal": [-1, 0, 0]}
    x0, y0, x1, y1 = rh.bev_box(door, CAM, size_m=8.0, res=0.05, thick=0.3)
    assert (x0 + x1) / 2 == pytest.approx(80, abs=1) and (x1 - x0) == pytest.approx(0.9 / 0.05, abs=2)
    assert 160 - (y0 + y1) / 2 == pytest.approx(3.0 / 0.05, abs=2)
    assert rh.bev_box(dict(door, center=[-3.0, 0.0, 1.0]), CAM, size_m=8.0, res=0.05) is None  # behind the robot
