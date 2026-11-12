"""Save what the robot sees at each door stop: gz-transport grabber for the robot's camera topics + PNG/JSON writer.

Same file layout as generate.py frames (rgb.png, depth.png uint16 mm, seg.png, meta.json). Gazebo is only needed for Grabber."""
import json, threading, time
import numpy as np
from PIL import Image

from doorsim import glass
from doorsim.frames import boxes_from_msg


def TOPICS(name="robot", cam=0):
    """gz topics of the robot model's camera sensors; cam k >= 1 = ring camera k (/<name>/cam<k>/...)."""
    b = f"/{name}" if cam == 0 else f"/{name}/cam{cam}"
    return {"rgb": f"{b}/rgbd/image", "depth": f"{b}/rgbd/depth_image", "seg": f"{b}/seg/labels_map",
            "boxes": f"{b}/boxes", "boxes3d": f"{b}/boxes3d"}


def door_of(label):
    """Door id for a building label id (10 * door + part, doors < 200); None for clutter (>= 200) and 0."""
    return label // 10 if 0 < label < 200 else None


def save_stop(d, rgb, depth, seg, boxes2d, boxes3d, meta):
    """Write rgb/depth/seg PNGs and meta.json (meta + boxes + boxes_3d with door ids + doors_in_view) into folder d."""
    from pathlib import Path
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(d / "rgb.png")
    Image.fromarray(glass.to_png16(depth)).save(d / "depth.png")
    Image.fromarray(seg).save(d / "seg.png")
    b2 = [{**b, "door": door_of(b["label"])} for b in boxes2d]
    b3 = [{**b, "door": door_of(b["label"])} for b in boxes3d]
    out = {**meta, "boxes": b2, "boxes_3d": b3,
           "doors_in_view": sorted({b["door"] for b in b2 if b["door"] is not None})}
    (d / "meta.json").write_text(json.dumps(out, indent=2) + "\n")


class Grabber:
    """Subscribes the robot's camera topics (cam = ring camera index, 0 = front); grab() returns the newest set of messages stamped at or after a time."""

    def __init__(self, name="robot", cam=0):
        # gz imports here so the module (door_of, save_stop) works without gz-transport
        from doorsim.gzsim import Node
        from gz.msgs10.annotated_axis_aligned_2d_box_v_pb2 import AnnotatedAxisAligned2DBox_V
        from gz.msgs10.annotated_oriented_3d_box_v_pb2 import AnnotatedOriented3DBox_V
        from gz.msgs10.image_pb2 import Image as GzImage
        types = {"rgb": GzImage, "depth": GzImage, "seg": GzImage,
                 "boxes": AnnotatedAxisAligned2DBox_V, "boxes3d": AnnotatedOriented3DBox_V}
        self.node, self.latest, self.lock = Node(), {}, threading.Lock()
        for key, topic in TOPICS(name, cam).items():
            self.node.subscribe(types[key], topic, self._cb(key))

    def _cb(self, key):
        def cb(m):
            with self.lock:
                self.latest[key] = m
        return cb

    def grab(self, after_ns, timeout=10):
        """Dict rgb (uint8 HxWx3), depth (float32 m), seg (uint8), boxes, boxes3d, stamp_ns of the newest messages all stamped
        >= after_ns; None after timeout seconds."""
        from doorsim.gzsim import image_array, ns
        end = time.time() + timeout
        while time.time() < end:
            with self.lock:
                m = dict(self.latest)
            if len(m) == 5 and min(ns(x.header.stamp) for x in m.values()) >= after_ns:
                return {"rgb": image_array(m["rgb"], "rgb"), "depth": image_array(m["depth"], "depth"),
                        "seg": image_array(m["seg"], "seg"), "boxes": boxes_from_msg(m["boxes"], "2d"),
                        "boxes3d": boxes_from_msg(m["boxes3d"], "3d"), "stamp_ns": ns(m["rgb"].header.stamp)}
            time.sleep(0.05)
        return None
