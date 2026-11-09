# Unit tests for saving what the robot sees at each door stop (phase 2e, Madelief 2026-11-09: "save frames at stops").
# No Gazebo needed: python3 -m pytest sim/tests/test_stop_frames.py
#
#   stop_frames.TOPICS(name="robot") -> {"rgb": "/<name>/rgbd/image", "depth": "/<name>/rgbd/depth_image",
#     "seg": "/<name>/seg/labels_map", "boxes": "/<name>/boxes", "boxes3d": "/<name>/boxes3d"}
#   stop_frames.door_of(label) -> door id for a building label id (10 * door + part, doors < 200), None for clutter (>= 200) / 0.
#   stop_frames.save_stop(d, rgb, depth, seg, boxes2d, boxes3d, meta): writes into folder d (made if missing) rgb.png (uint8 HxWx3),
#     depth.png (uint16 mm, 0 = no value / not finite / <= 0, like generate.py), seg.png (uint8 label ids), meta.json = meta +
#     "boxes" (2D boxes, each with "door" = door_of(label)) + "boxes_3d" (same) + "doors_in_view" (sorted door ids with a 2D box).
#     boxes2d / boxes3d are lists as doorsim.frames.boxes_from_msg returns them (dicts with "label").
#   stop_frames.Grabber(name="robot"): subscribes the TOPICS with gz-transport; grab(after_ns, timeout=10) -> dict with "rgb"
#     (uint8 HxWx3), "depth" (float32 m), "seg" (uint8), "boxes" (list), "boxes3d" (list), "stamp_ns": the newest set of messages
#     whose stamps are all >= after_ns, or None on timeout. (Gazebo part: not unit tested.)
import json, pathlib, sys
import numpy as np
import pytest
from PIL import Image

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import stop_frames


def test_topics():
    t = stop_frames.TOPICS("bot")
    assert t == {"rgb": "/bot/rgbd/image", "depth": "/bot/rgbd/depth_image", "seg": "/bot/seg/labels_map",
                 "boxes": "/bot/boxes", "boxes3d": "/bot/boxes3d"}
    assert stop_frames.TOPICS()["rgb"] == "/robot/rgbd/image"


@pytest.mark.parametrize("label,door", [(11, 1), (14, 1), (17, 1), (25, 2), (191, 19), (200, None), (247, None), (0, None)])
def test_door_of(label, door):
    assert stop_frames.door_of(label) == door


def test_save_stop(tmp_path):
    rgb = np.zeros((4, 6, 3), np.uint8); rgb[0, 0] = [255, 1, 2]
    depth = np.array([[1.234, np.nan, np.inf, -1.0, 0.0, 12.0]] * 4, np.float32)
    seg = np.full((4, 6), 21, np.uint8)
    b2 = [{"label": 21, "x_min": 0, "y_min": 0, "x_max": 5, "y_max": 3}, {"label": 203, "x_min": 1, "y_min": 1, "x_max": 2, "y_max": 2},
          {"label": 12, "x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1}]
    b3 = [{"label": 21, "center": [0, 0, 1], "orientation_xyzw": [0, 0, 0, 1], "size": [1, 1, 1]}]
    d = tmp_path / "stops" / "000"
    stop_frames.save_stop(d, rgb, depth, seg, b2, b3, {"door": 2, "dist": 2.5})
    assert (np.array(Image.open(d / "rgb.png")) == rgb).all()
    dep = np.array(Image.open(d / "depth.png"))
    assert dep.dtype == np.uint16 and dep[0].tolist() == [1234, 0, 0, 0, 0, 12000]
    assert (np.array(Image.open(d / "seg.png")) == seg).all()
    m = json.loads((d / "meta.json").read_text())
    assert m["door"] == 2 and m["dist"] == 2.5
    assert [b["door"] for b in m["boxes"]] == [2, None, 1]
    assert m["boxes_3d"][0]["door"] == 2
    assert m["doors_in_view"] == [1, 2]
