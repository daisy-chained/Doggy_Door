"""COCO export folder -> torch Dataset of (PIL RGB image, target) with 0-based category ids.

Ignore boxes (iscrowd 1, from the exporter's --ignore-below) are ignore in scoring (pycocotools reads them from the gt json)
and left out of the training targets, like DETR's COCO loader. A true ignore in D-FINE's loss (matched but unpunished)
would need a custom loss: its denoising queries are built from every target box."""
import json, math, os, pathlib
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

DEPTH_MAX, HEIGHT_MAX = 8.0, 3.0  # m: depth / height above floor that map to 255


def rot(roll, pitch, yaw):
    """Rotation matrix (body -> world), Gazebo's convention; copy of sim/doorsim/frames.py (train must not import sim)."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp, cp*sr, cp*cr]])


def world_points(depth, K, camera):
    """(H, W, 3) world points of the depth pixels (planar z-depth in m, nan where invalid); copy of sim/doorsim/effects.py.
    Optical frame -> camera body (z, -x, -y) -> world."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    body = np.stack([depth, -(u - K[2]) / K[0] * depth, -(v - K[5]) / K[4] * depth], axis=-1)
    return body @ rot(*camera["world_rpy"]).T + np.array(camera["world_position"])


def encode_depth(depth_mm, K, camera, kind):
    """uint16 mm depth (0 = invalid) -> uint8 (H, W, 3). depth: clip(depth / DEPTH_MAX) in all channels. dha: that, height above
    the floor (world z / HEIGHT_MAX), angle of the surface normal to the vertical (0 floor .. 90 wall, as deg / 180).
    Everything is 0 where the depth, or for the normal a neighbour, is invalid; the unoriented normal makes the angle 0..90."""
    d = depth_mm.astype(np.float64) / 1000
    ok = d > 0
    ch0 = np.where(ok, np.clip(d / DEPTH_MAX, 0, 1), 0) * 255
    if kind == "depth":
        return np.repeat(ch0[..., None], 3, axis=-1).round().astype(np.uint8)
    P = world_points(np.where(ok, d, np.nan), K, camera)
    height = np.clip(P[..., 2] / HEIGHT_MAX, 0, 1) * 255
    with np.errstate(invalid="ignore", divide="ignore"):
        n = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0))  # nan if the pixel or a neighbour is invalid
        ang = np.degrees(np.arccos(np.clip(np.abs(n[..., 2]) / np.linalg.norm(n, axis=-1), 0, 1))) / 180 * 255
    out = np.stack([ch0, height, ang], axis=-1)
    return np.nan_to_num(np.where(ok[..., None], out, 0)).round().astype(np.uint8)


def encode_thermal(kelvin):
    """Thermal frame (K) -> uint8 like a camera's automatic gain control: clip to the 1st-99th percentile of the frame, stretch to 0-255."""
    k = np.asarray(kelvin, float)
    lo, hi = np.percentile(k, [1, 99])
    if not hi > lo:
        return np.full(k.shape, 128, np.uint8)
    return ((np.clip(k, lo, hi) - lo) / (hi - lo) * 255).round().astype(np.uint8)


def frame_dir(path):
    """Folder of the dataset that made an export image: the first symlink target's folder (thermal datasets symlink rgb.png to the base
    frame, so path.resolve().parent would land in the base dataset)."""
    path = pathlib.Path(path)
    if not path.is_symlink():
        return path.parent
    target = pathlib.Path(os.readlink(path))
    return (target if target.is_absolute() else path.parent / target).parent


def load_input(path, kind, _K_cache=None):
    """Export image path -> PIL RGB image of the input kind (rgb, depth, dha, thermal); intrinsics K from <dataset>/dataset.json, cached in _K_cache."""
    path = pathlib.Path(path)
    if kind == "rgb":
        return Image.open(path).convert("RGB")
    if kind == "thermal":
        rgb = Image.open(path)
        k = np.asarray(Image.open(frame_dir(path) / "thermal.png"), float) / 100  # uint16, 0.01 K
        t = Image.fromarray(encode_thermal(k)).resize(rgb.size, Image.BILINEAR)
        return Image.fromarray(np.asarray(t)[..., None].repeat(3, axis=2))
    frame = frame_dir(path)  # the export's image is a symlink to <dataset>/frames/<frame>/rgb.png
    ds = frame.parent.parent
    cache = {} if _K_cache is None else _K_cache
    if ds not in cache:
        cache[ds] = json.loads((ds / "dataset.json").read_text())["camera_intrinsics"]["K"]
    return Image.fromarray(encode_depth(np.asarray(Image.open(frame / "depth.png")), cache[ds],
                                        json.loads((frame / "meta.json").read_text())["camera"], kind))


class CocoDetection(Dataset):
    def __init__(self, export_dir, split, input="rgb"):
        self.input, self._K = input, {}
        self.root = pathlib.Path(export_dir)
        self.gt_path = self.root / "coco" / f"{split}.json"
        coco = json.loads(self.gt_path.read_text())
        self.images = sorted(coco["images"], key=lambda im: im["id"])
        self.id2label = {c["id"] - 1: c["name"] for c in coco["categories"]}
        self.anns = {im["id"]: [] for im in self.images}
        for a in coco["annotations"]:
            if a.get("iscrowd", 0):
                continue
            self.anns[a["image_id"]].append({"bbox": list(a["bbox"]), "category_id": a["category_id"] - 1,
                                             "area": a["area"], "iscrowd": 0})

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = self.images[i]
        path = self.root / im["file_name"]
        img = load_input(path, self.input, self._K)
        return img, {"image_id": im["id"], "annotations": self.anns[im["id"]]}
