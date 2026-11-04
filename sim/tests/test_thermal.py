# Unit tests for sim/doorsim/thermal.py (no Gazebo): temperatures per model for the two scenarios, thermal camera +
# temperatures injected into a v4 world file, the replay plan for a thermal render pass over an existing dataset, PNG
# encoding. Madelief 2026-11-04: thermal camera, room-temperature + fire-heated scenarios, 320 x 240, same FOV as RGB.
import json, pathlib, sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import clutter, thermal

WORLD = SIM / "door_room" / "v4" / "door_room_v4_000.sdf"
MODELS = ["floor", "back_wall", "room_shell", "door_a", "glass_a", "handle_a", "handle_a_back", "door_b", "handle_b",
          "handle_b_back", "box_1", "table", "plant", "rig"]


def test_room_scenario_small_contrast_around_room_air():
    t = thermal.temperatures(np.random.default_rng(0), "room", MODELS)
    assert 291.15 <= t["air"] <= 297.15  # 18-24 degC
    assert "rig" not in t["models"]  # the camera rig has no temperature
    assert set(t["models"]) == set(MODELS) - {"rig"}
    assert all(abs(v - t["air"]) <= 3.5 for v in t["models"].values())


def test_fire_scenario_hot_room():
    t = thermal.temperatures(np.random.default_rng(0), "fire", MODELS)
    assert t["models"]["room_shell"] >= 313.15  # ceiling / walls heated: >= 40 degC
    assert t["layer"] > t["air"]  # hot smoke layer under the ceiling (used by the thermal smoke step)
    assert all(v >= 293.15 for v in t["models"].values())


def test_temperatures_are_reproducible_and_differ_by_scenario():
    a = thermal.temperatures(np.random.default_rng(3), "room", MODELS)
    b = thermal.temperatures(np.random.default_rng(3), "room", MODELS)
    c = thermal.temperatures(np.random.default_rng(3), "fire", MODELS)
    assert a == b and a != c
    with pytest.raises(ValueError):
        thermal.temperatures(np.random.default_rng(0), "arctic", MODELS)


def test_inject_thermal_camera_and_temperatures():
    sdf = WORLD.read_text()
    temps = {"door_a": 330.0, "room_shell": 295.0}
    root = ET.fromstring(thermal.inject(sdf, temps).encode())
    world = root.find("world")
    models = {m.get("name"): m for m in world.findall("model")}
    for name, k in temps.items():
        vis = models[name].findall(".//visual")
        assert vis
        for v in vis:
            p = v.find("plugin[@name='gz::sim::systems::Thermal']")
            assert p is not None and float(p.find("temperature").text) == pytest.approx(k)
    assert models["door_b"].find(".//visual/plugin") is None  # models without a temperature are left alone
    sensor = models["rig"].find(".//sensor[@type='thermal']")
    assert sensor is not None and sensor.find("topic").text == "thermal"
    cam = sensor.find("camera")
    rgbd = models["rig"].find(".//sensor[@type='rgbd_camera']/camera")
    assert cam.find("horizontal_fov").text == rgbd.find("horizontal_fov").text  # same field of view as the RGB camera
    assert cam.find("image/width").text == "320" and cam.find("image/height").text == "240"
    assert sensor.find("plugin[@name='gz::sim::systems::ThermalSensor']/resolution").text == "0.01"


def test_png_roundtrip_in_centikelvin():
    k = np.array([[293.15, 330.0], [253.15, 400.0]])
    png = thermal.to_png16(k)
    assert png.dtype == np.uint16 and png[0, 0] == 29315
    assert thermal.from_png16(png) == pytest.approx(k)


def meta(world, scene, view, joint_a, cam_x, clutter_items=None):
    return {"world": world, "scene": scene, "view": view,
            "camera": {"world_position": [cam_x, 0.0, 1.0], "world_rpy": [0.0, 0.0, 0.0]},
            "objects": {"door_a": {"joint": {"name": "hinge"}, "joint_pos": joint_a}, "door_b": {"joint": {"name": "hinge"}, "joint_pos": 0.0},
                        "handle_a": {"id": 12}},
            "clutter": clutter_items or {}}


def test_replay_plan_groups_frames_by_world_and_scene():
    item = {"position": [1.0, 2.0, 0.0], "yaw": 0.5, "colour": None, "in_scene": True}
    metas = {"000000": meta("w0", 0, 0, -0.5, 0.1, {"box_1": item}), "000001": meta("w0", 0, 1, -0.5, 0.2, {"box_1": item}),
             "000002": meta("w0", 1, 0, 0.0, 0.3), "000003": meta("w1", 2, 0, -1.0, 0.4)}
    plan = thermal.replay_plan(metas)
    assert [(p["world"], p["scene"]) for p in plan] == [("w0", 0), ("w0", 1), ("w1", 2)]
    first = plan[0]
    assert first["doors"] == {"door_a": -0.5, "door_b": 0.0}
    assert [f for f, _ in first["views"]] == ["000000", "000001"]
    assert first["views"][1][1]["world_position"] == [0.2, 0.0, 1.0]
    # every pool item is placed: the scene's own items where they were, all others parked
    assert first["clutter"]["box_1"] == item
    assert set(first["clutter"]) == {p["name"] for p in clutter.POOL}
    assert all(not v["in_scene"] for n, v in first["clutter"].items() if n != "box_1")
    assert plan[1]["clutter"]["box_1"]["in_scene"] is False
