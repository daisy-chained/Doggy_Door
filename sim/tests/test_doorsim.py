# Unit tests for sim/doorsim (no Gazebo needed): python3 -m pytest sim/tests
import json, math, pathlib, sys
import numpy as np
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import frames, glass, sampling

ATTRS_V1 = json.loads((SIM / "door_room" / "door_room_v1.attributes.json").read_text())
VIEW_CFG = {"distance": [0.5, 4.0], "angle_deg": 60, "height": [0.3, 1.5], "target_height": [0.6, 1.6],
            "target_jitter": 0.3, "roll_deg": 0}


# --- frames ---
def test_gl_box_to_optical_flips_y_and_z():
    c, q = frames.gl_box_to_optical([0.64, -0.2, -2.95], [0, 0, 0, 1])
    assert c == [0.64, 0.2, 2.95]
    assert q == [1, 0, 0, 0]  # identity in the graphics frame = 180 deg about x in optical


def test_world_to_cam_fixed_rig():
    cam = {"world_position": [0, 0, 1.2], "world_rpy": [0, 0, 0]}
    # door A hinge point: 3 m ahead, 1.44 m to the right, 0.15 m below the camera
    p = frames.world_to_cam([3.02, -1.44, 1.05], cam, True)
    assert p == pytest.approx([1.44, 0.15, 3.02])
    assert frames.world_to_cam([0, 0, 1], cam, False) == pytest.approx([0, -1, 0])  # up = -y in optical


def test_parse_text_boxes_missing_zero_fields():
    text = "annotated_box {\n box {\n min_corner {\n y: 5\n }\n max_corner {\n x: 30\n y: 40\n }\n }\n label: 12\n}\n"
    assert frames.parse_text_boxes(text, "2d") == [{"label": 12, "x_min": 0, "y_min": 5, "x_max": 30, "y_max": 40}]


def test_build_frame_links_boxes_and_rejects_unknown_ids():
    f = frames.build_frame(ATTRS_V1, [{"label": 22, "x_min": 1, "y_min": 2, "x_max": 3, "y_max": 4}], [],
                           {"door_b": 0.004}, extra={"glass_depth_mode": "holes"})
    assert f["objects"]["handle_b"]["box"] == {"x_min": 1, "y_min": 2, "x_max": 3, "y_max": 4}
    assert f["objects"]["door_a"]["box"] is None
    assert f["objects"]["door_b"]["state"] == "closed"
    assert f["glass_depth_mode"] == "holes"
    assert "slide_cam" in f["objects"]["door_b"] and "front_cam" in f["objects"]["door_a"]
    with pytest.raises(ValueError):
        frames.build_frame(ATTRS_V1, [{"label": 99, "x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1}], [])


# --- sampling ---
def test_door_targets_within_range():
    rng = np.random.default_rng(0)
    for _ in range(200):
        t = sampling.door_targets(rng, ATTRS_V1, {"p_closed": 0.3, "min_open_fraction": 0.1})
        assert set(t) == {"door_a", "door_b"}
        assert -1.6 <= t["door_a"] <= 0 and 0 <= t["door_b"] <= 0.9


def test_view_in_zone_and_aimed_at_target():
    rng = np.random.default_rng(1)
    for _ in range(200):
        v = sampling.view(rng, ATTRS_V1, VIEW_CFG)
        f = ATTRS_V1["objects"][v["target_door"]]["front_world"]["point"]
        x, y, z = v["world_position"]
        d = math.hypot(x - f[0], y - f[1])
        assert 0.5 - 1e-9 <= d <= 4.0 + 1e-9 and 0.3 <= z <= 1.5
        assert x < f[0]  # on the camera side of the wall
        t = frames.world_to_cam(v["target_world"], v, True)
        assert t[0] == pytest.approx(0, abs=1e-9) and t[1] == pytest.approx(0, abs=1e-9) and t[2] > 0  # dead centre


def test_same_seed_same_views():
    a = [sampling.view(np.random.default_rng(7), ATTRS_V1, VIEW_CFG) for _ in range(2)]
    assert a[0] == a[1]


def test_lights_scale_base_colour():
    l = sampling.lights(np.random.default_rng(0), {"ceiling_light": {"base": [0.8, 0.8, 0.75], "intensity": [0.5, 0.5], "tint": 0.0}})
    assert l["ceiling_light"]["diffuse"] == pytest.approx([0.4, 0.4, 0.375])


# --- glass ---
def _scene():
    gt = np.full((40, 60), 3.0, np.float32)
    behind = np.full((40, 60), 7.0, np.float32)
    mask = np.zeros((40, 60), bool); mask[10:30, 20:40] = True
    return gt, behind, mask


def test_glass_holes_and_see_through():
    gt, behind, mask = _scene()
    rng = np.random.default_rng(0)
    h = glass.apply(gt, mask, "holes", rng)
    assert np.isnan(h[mask]).all() and (h[~mask] == 3.0).all()
    s = glass.apply(gt, mask, "see_through", rng, behind)
    assert (s[mask] == 7.0).all() and (s[~mask] == 3.0).all()
    with pytest.raises(ValueError):
        glass.apply(gt, mask, "see_through", rng)


def test_glass_partial_mixes_all_three():
    gt, behind, mask = _scene()
    p = glass.apply(gt, mask, "partial", np.random.default_rng(3), behind, {"holes": 0.33, "see_through": 0.33, "cell": 4})
    vals = p[mask]
    assert np.isnan(vals).any() and (vals == 7.0).any() and (vals == 3.0).any()
    assert (p[~mask] == 3.0).all()


def test_pick_mode_follows_mix():
    rng = np.random.default_rng(0)
    assert {glass.pick_mode(rng, {"holes": 1}) for _ in range(20)} == {"holes"}


def test_png16_invalid_is_zero():
    d = np.array([[1.234, np.nan], [np.inf, -1.0]], np.float32)
    assert glass.to_png16(d).tolist() == [[1234, 0], [0, 0]]
