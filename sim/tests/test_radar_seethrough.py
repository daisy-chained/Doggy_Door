# Unit tests for the radar see-through + diffuse scatter (Madelief 2026-11-07, "get started on both"; numbers Claude, research-based):
# real 60-77 GHz radar sees partly through drywall and wooden doors (73 GHz: drywall ~4 dB, wooden door ~7.6 dB one-way,
# arXiv 1908.00166; NIST 60 GHz doors 25-40 dB, setup-dependent) and rough surfaces scatter some power back at any angle
# (effective roughness model, Degli-Esposti). In v4 the radar proxy made walls / doors opaque and purely specular.
#   - radar.inject adds a hidden-wall lidar "radar_behind" (gpu_lidar 480 x 100 rays over the wide camera's view,
#     <visibility_mask>1</visibility_mask>); every visual of back_wall / door_* / handle_* / glass_* gets <visibility_flags>2,
#     so the lidar sees what is behind the wall and the doors (Gazebo's depth camera ignores the mask, the gpu_lidar honours it).
#   - radar.behind_depth: lidar ranges (100 x 480, row 0 = lowest ray, col 0 = min_angle = rightmost) -> planar z-depth on the
#     wide camera's pixel grid (row 0 = top, col 0 = left), so it lines up with radar_depth.
#   - radar.pixel_returns(..., behind=, pass_db=): pixels whose surface is the back wall or a door panel also return from the
#     behind depth, reflecting like a wall, weakened by twice the one-way loss of what they passed (pass_db["wall"] / ["door"]);
#     glass keeps its own two-way loss (GLASS_PASS_DB) and uses the behind depth when given; metal (handles) passes nothing.
#   - returned_power = gain x ((1 - DIFFUSE_FRACTION) x specular lobe + DIFFUSE_FRACTION x cos^2(incidence)) / r^4
#     (diffuse: Lambertian there and back, no cutoff).
#   - radar_export draws the one-way losses per frame (seeded by the frame name) from PASS_DB_RANGE (domain randomisation:
#     the literature spread is wide), so one export mixes thin and thick walls / doors.
# python3 -m pytest sim/tests/test_radar_seethrough.py
import json, math, pathlib, sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import effects, radar, radar_export, radar_render, thermal

WORLD = SIM / "door_room" / "v4" / "door_room_v4_000.sdf"
ATTRS = json.loads((SIM / "door_room" / "v4" / "door_room_v4_000.attributes.json").read_text())
W, H = 240, 50
CAM = {"world_position": [0.0, 0.0, 1.0], "world_rpy": [0.0, 0.0, 0.0]}
WALL_ID = thermal.STRUCT_IDS["back_wall"]
SHELL_ID = thermal.STRUCT_IDS["room_shell"]
DOOR_ID = next(o["id"] for o in ATTRS["objects"].values() if o.get("class") == 1)
HANDLE_ID = next(o["id"] for o in ATTRS["objects"].values() if o.get("class") == 2)
PASS = {"wall": -5.0, "door": -8.0}


def K():
    return radar.camera_K(W, H, math.radians(120))


def plane(x):
    return np.full((H, W), float(x))


def returns(seg_id, behind=None, pass_db=PASS, depth=3.0):
    seg = np.full((H, W), seg_id, np.uint16)
    d = plane(depth)
    return radar.pixel_returns(d, seg, d, K(), CAM, radar.materials(ATTRS), behind=behind, pass_db=pass_db)


# --- diffuse scatter ---
def test_diffuse_keeps_some_power_at_grazing_angles():
    f = radar.DIFFUSE_FRACTION
    assert 0.03 <= f <= 0.2
    g = 10 ** (radar.REFLECT_DB["wall"] / 10)
    assert radar.returned_power(radar.REFLECT_DB["wall"], 0, 3.0) == pytest.approx(g / 3.0 ** 4)  # facing: specular + diffuse = gain
    p85 = radar.returned_power(radar.REFLECT_DB["wall"], 85, 3.0)
    assert p85 == pytest.approx(g * f * math.cos(math.radians(85)) ** 2 / 3.0 ** 4)            # beyond the lobe cutoff: diffuse only
    p60 = radar.returned_power(radar.REFLECT_DB["wall"], 60, 3.0)
    lobe = math.cos(math.radians(60)) ** radar.LOBE_EXPONENT
    assert p60 == pytest.approx(g * ((1 - f) * lobe + f * math.cos(math.radians(60)) ** 2) / 3.0 ** 4)
    p = [radar.returned_power(0, a, 3.0) for a in (0, 30, 60, 85)]
    assert p[0] > p[1] > p[2] > p[3] > 0


# --- behind layer ---
def test_wall_and_door_pass_part_of_the_signal_to_what_is_behind():
    for seg_id, key in ((WALL_ID, "wall"), (DOOR_ID, "door")):
        pts0, r0, p0 = returns(seg_id)                      # no behind render: surface only (v4 behaviour)
        pts, r, p = returns(seg_id, behind=plane(6.0))
        n = len(r0)
        assert len(r) == 2 * n
        np.testing.assert_allclose(p[:n], p0)               # the surface layer is unchanged
        far = r[n:]
        assert far.min() > 5.9                              # the second layer sits at the behind depth
        # same pixel, wall-like reflection at 6 m, two-way loss of what it passed
        i = np.argmin(r0)                                   # the centre pixel (facing, incidence ~0)
        expect = p0[i] * 10 ** ((radar.REFLECT_DB["wall"] - radar.REFLECT_DB[radar.materials(ATTRS)[seg_id]] + 2 * PASS[key]) / 10) \
            * (r0[i] / r[n + i]) ** 4
        assert p[n + i] == pytest.approx(expect, rel=0.05)


def test_door_loss_bigger_than_wall_loss_shows_in_power():
    _, _, pw = returns(WALL_ID, behind=plane(6.0), pass_db={"wall": -5.0, "door": -5.0})
    _, _, pw2 = returns(WALL_ID, behind=plane(6.0), pass_db={"wall": -15.0, "door": -5.0})
    n = len(pw) // 2
    np.testing.assert_allclose(pw2[n:], pw[n:] * 10 ** (-20 / 10), rtol=1e-6)  # 10 dB more one-way = 20 dB two-way


def test_nothing_behind_shell_or_metal():
    for seg_id in (SHELL_ID, HANDLE_ID):
        _, r0, _ = returns(seg_id)
        _, r, _ = returns(seg_id, behind=plane(6.0))
        assert len(r) == len(r0)


def test_behind_needs_losses_and_nan_behind_is_no_return():
    with pytest.raises(ValueError):
        returns(WALL_ID, behind=plane(6.0), pass_db=None)
    b = plane(6.0)
    b[:, : W // 2] = np.nan   # nothing behind on the left half (lidar max range)
    _, r0, _ = returns(WALL_ID)
    _, r, _ = returns(WALL_ID, behind=b)
    assert len(r0) < len(r) < 2 * len(r0)


def test_simulate_and_ra_map_take_the_behind_layer():
    seg = np.full((H, W), WALL_ID, np.uint16)
    d = plane(3.0)
    rng = lambda: np.random.default_rng(0)
    mats = radar.materials(ATTRS)
    a = radar.simulate(d, seg, d, K(), CAM, radar.MODELS["imaging"], mats, rng())
    b = radar.simulate(d, seg, d, K(), CAM, radar.MODELS["imaging"], mats, rng(), behind=plane(5.0), pass_db=PASS)
    far = lambda out: (np.linalg.norm(out["points"] - CAM["world_position"], axis=1) > 4.5).sum()
    assert far(a) == 0 and far(b) > 0                      # the room behind the wall shows up
    from doorsim import radar_heatmap as rh
    m0 = rh.ra_map(d, seg, d, K(), CAM, radar.MODELS["imaging"], mats, rng())
    m1 = rh.ra_map(d, seg, d, K(), CAM, radar.MODELS["imaging"], mats, rng(), behind=plane(5.0), pass_db=PASS)
    i5 = np.searchsorted(m1["ranges"], 5.0)
    assert m1["power_db"][i5].max() > m0["power_db"][i5].max() + 10


# --- lidar ranges -> pixel depth ---
def lidar_geometry():
    return radar.behind_lidar_geometry(W, H, math.radians(120))


def ranges_of(depth_fn, geom):
    """Lidar ranges (rows = vertical rays from the lowest up, cols = horizontal rays from min_angle = right to left) for a scene
    given as planar depth along x by a function of the ray's (azimuth, elevation)."""
    az = np.linspace(geom["h_min"], geom["h_max"], geom["h_samples"])
    el = np.linspace(geom["v_min"], geom["v_max"], geom["v_samples"])
    A, E = np.meshgrid(az, el)
    x = depth_fn(A, E)
    return x / (np.cos(A) * np.cos(E))


def test_lidar_geometry_covers_the_wide_view():
    g = lidar_geometry()
    assert (g["h_samples"], g["v_samples"]) == (W, H)  # one ray per pixel column / row
    assert g["h_min"] == pytest.approx(-math.radians(60)) and g["h_max"] == pytest.approx(math.radians(60))
    vfov = 2 * math.atan(math.tan(math.radians(60)) * H / W)
    assert g["v_max"] - g["v_min"] == pytest.approx(vfov, rel=0.02)


def test_behind_depth_flat_wall():
    g = lidar_geometry()
    r = ranges_of(lambda A, E: np.full_like(A, 5.0), g)
    d = radar.behind_depth(r, g, K(), (H, W))
    assert d.shape == (H, W)
    np.testing.assert_allclose(d[2:-2, 2:-2], 5.0, rtol=0.01)


def test_behind_depth_left_right_and_up_down_orientation():
    g = lidar_geometry()
    # left of the radar (azimuth > 0) at 2 m, right at 4 m; above the horizon + 1 m
    r = ranges_of(lambda A, E: np.where(A > 0, 2.0, 4.0) + np.where(E > 0, 1.0, 0.0), g)
    d = radar.behind_depth(r, g, K(), (H, W))
    assert d[H - 5, 10] == pytest.approx(2.0, rel=0.02)       # image left, lower half
    assert d[H - 5, W - 10] == pytest.approx(4.0, rel=0.02)   # image right
    assert d[5, 10] == pytest.approx(3.0, rel=0.02)           # image top = up
    r[...] = np.inf
    assert np.isnan(radar.behind_depth(r, g, K(), (H, W))).all()


# --- Gazebo side ---
def test_inject_hidden_wall_lidar_and_flags():
    root = ET.fromstring(radar.inject(WORLD.read_text()).encode())
    ms = {m.get("name"): m for m in root.find("world").findall("model")}
    lid = [s for s in ms["rig"].iter("sensor") if s.get("name") == "radar_behind"]
    assert len(lid) == 1 and lid[0].get("type") == "gpu_lidar"
    s = lid[0]
    assert s.find("topic").text == "radar/behind"
    assert s.find("lidar/visibility_mask").text == "1"
    g = radar.behind_lidar_geometry(*radar.WIDE_SIZE, radar.WIDE_HFOV)
    assert int(s.find("lidar/scan/horizontal/samples").text) == g["h_samples"]
    assert float(s.find("lidar/scan/horizontal/min_angle").text) == pytest.approx(g["h_min"])
    assert float(s.find("lidar/scan/vertical/max_angle").text) == pytest.approx(g["v_max"])
    hidden = lambda m: all(v.find("visibility_flags") is not None and v.find("visibility_flags").text == "2" for v in m.iter("visual"))
    for n, m in ms.items():
        if n == "back_wall" or n.startswith(("door_", "handle_", "glass_")):
            assert hidden(m), n
        elif n in ("room_shell",) or n.startswith(("clutter", "box_", "floor", "ground")):
            assert not any(v.find("visibility_flags") is not None for v in m.iter("visual")), n
    assert sum(1 for v in ms["back_wall"].iter("visual")) > 0


def test_render_writes_behind_last():
    assert radar_render.FILES[-1] == "radar_behind.png"  # the done marker: frames rendered before see-through get re-rendered
    assert "radar_depth_noglass.png" in radar_render.FILES


def test_export_draws_losses_per_frame():
    lo_w, hi_w = radar.PASS_DB_RANGE["wall"]
    lo_d, hi_d = radar.PASS_DB_RANGE["door"]
    assert lo_w < hi_w < 0 and lo_d < hi_d < 0
    assert hi_w >= -5 and lo_w <= -10 and hi_d >= -8 and lo_d <= -20    # spans the 73 GHz and the NIST 60 GHz figures
    a = radar_export.pass_db_for("000040", seed=0)
    assert a == radar_export.pass_db_for("000040", seed=0)              # reproducible
    assert lo_w <= a["wall"] <= hi_w and lo_d <= a["door"] <= hi_d
    draws = [radar_export.pass_db_for(f"{i:06d}", seed=0)["wall"] for i in range(50)]
    assert len(set(round(x, 3) for x in draws)) > 40                    # varies per frame
