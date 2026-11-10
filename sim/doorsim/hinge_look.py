"""Hinge look (option A, Madelief 2026-11-10): guess the hinge edge from the handle (hinge is opposite the handle), then
drive to extra spots aimed at that edge and look again. Pure helpers: map, geometry, detection checks, offline scoring."""
import math
import numpy as np
import yaml
from PIL import Image
from pathlib import Path


def load_map(yaml_path):
    """Read a map_server yaml + PGM -> {"occ": bool HxW (occupied OR unknown), "res", "origin"}."""
    yaml_path = Path(yaml_path)
    cfg = yaml.safe_load(yaml_path.read_text())
    p = np.asarray(Image.open(yaml_path.parent / cfg["image"]).convert("L"), dtype=float)
    occ_p = (255.0 - p) / 255.0
    occupied = occ_p > cfg.get("occupied_thresh", 0.65)
    is_free = occ_p < cfg.get("free_thresh", 0.25)
    return {"occ": occupied | ~is_free, "res": float(cfg["resolution"]), "origin": tuple(cfg["origin"][:2])}


def free(m, x, y, radius):
    """True when the square of half-size ceil(radius / res) cells around (x, y) is inside the map and all free."""
    h, w = m["occ"].shape
    half = int(math.ceil(radius / m["res"]))
    col = int((x - m["origin"][0]) / m["res"])
    row = h - 1 - int((y - m["origin"][1]) / m["res"])
    if col - half < 0 or row - half < 0 or col + half >= w or row + half >= h:
        return False
    return not m["occ"][row - half:row + half + 1, col - half:col + half + 1].any()


def pick_door(dets, img_w, score_min=0.5):
    """The door box (score >= score_min) whose centre column is closest to the image centre; None if none."""
    best, best_d = None, None
    for b, s, l in zip(dets["boxes"], dets["scores"], dets["labels"]):
        if l != "door" or s < score_min:
            continue
        d = abs((b[0] + b[2]) / 2 - img_w / 2)
        if best_d is None or d < best_d:
            best, best_d = list(b), d
    return best


def pick_door_span(dets, cols, score_min=0.5):
    """The door box (score >= score_min) overlapping the pixel span of cols the most; None without any overlap."""
    lo, hi = min(cols), max(cols)
    best, best_o = None, 0.0
    for b, s, l in zip(dets["boxes"], dets["scores"], dets["labels"]):
        if l != "door" or s < score_min:
            continue
        o = min(b[2], hi) - max(b[0], lo)
        if o > best_o:
            best, best_o = list(b), o
    return best


def hinge_side_image(door_xyxy, dets, score_min=0.3, depth_img=None, expected=None, depth_tol=1.0):
    """Hinge side in the image ("left" / "right") from the largest handle inside the door box (at depth `expected` +- depth_tol with depth_img); None without one."""
    cx = (door_xyxy[0] + door_xyxy[2]) / 2
    best, best_a = None, -1.0
    for b, s, l in zip(dets["boxes"], dets["scores"], dets["labels"]):
        if l != "handle" or s < score_min:
            continue
        if depth_img is not None and expected is not None:
            d = _box_depth(depth_img, b)
            if d is None or abs(d - expected) > depth_tol:
                continue
        hx, hy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if not (door_xyxy[0] <= hx <= door_xyxy[2] and door_xyxy[1] <= hy <= door_xyxy[3]):
            continue
        a = (b[2] - b[0]) * (b[3] - b[1])
        if a > best_a:
            best, best_a = hx, a
    if best is None:
        return None
    return "right" if best < cx else "left"


def edge_point(door, image_side, robot_yaw):
    """World (x, y) of the door edge on the given image side for a robot facing the door with heading robot_yaw."""
    tx, ty = -math.sin(door["yaw"]), math.cos(door["yaw"])          # along the wall
    if tx * -math.sin(robot_yaw) + ty * math.cos(robot_yaw) < 0:      # point it to the robot's left
        tx, ty = -tx, -ty
    s = (1.0 if image_side == "left" else -1.0) * door["width"] / 2
    return (door["pos"][0] + s * tx, door["pos"][1] + s * ty)


def look_poses(door, edge, stop_xy, free_fn, dist=1.5, angles=(20, 40), fallback=(1.2, 1.0)):
    """One pose per angle (deg off head-on) aimed at the edge: out before in, dist then fallbacks; angles without a free spot dropped."""
    px, py = door["pos"]
    nx, ny = math.cos(door["yaw"]), math.sin(door["yaw"])
    sgn = 1.0 if (stop_xy[0] - px) * nx + (stop_xy[1] - py) * ny >= 0 else -1.0
    nx, ny = sgn * nx, sgn * ny
    ux, uy = edge[0] - px, edge[1] - py
    ul = math.hypot(ux, uy) or 1.0
    ux, uy = ux / ul, uy / ul
    poses = []
    for ang in angles:
        a = math.radians(ang)
        found = None
        for d in (dist, *fallback):
            for s, name in ((1.0, "out"), (-1.0, "in")):
                x = edge[0] + d * (math.cos(a) * nx + s * math.sin(a) * ux)
                y = edge[1] + d * (math.cos(a) * ny + s * math.sin(a) * uy)
                if free_fn(x, y):
                    found = {"x": x, "y": y, "yaw": math.atan2(edge[1] - y, edge[0] - x), "angle": ang, "dist": d, "toward": name}
                    break
            if found:
                break
        if found:
            poses.append(found)
    return poses


def edge_column(cam_x, cam_y, cam_yaw, edge, fx, cx):
    """Image column of the edge for a camera at (cam_x, cam_y) looking along cam_yaw; None if not in front (< 0.05 m)."""
    dx, dy = edge[0] - cam_x, edge[1] - cam_y
    fwd = math.cos(cam_yaw) * dx + math.sin(cam_yaw) * dy
    if fwd <= 0.05:
        return None
    left = -math.sin(cam_yaw) * dx + math.cos(cam_yaw) * dy
    return cx - fx * left / fwd


def edge_forward(cam_x, cam_y, cam_yaw, edge):
    """How far the edge is in front of the camera along cam_yaw (the depth z); None when <= 0.05 m."""
    fwd = math.cos(cam_yaw) * (edge[0] - cam_x) + math.sin(cam_yaw) * (edge[1] - cam_y)
    return fwd if fwd > 0.05 else None


def _box_depth(depth_img, box):
    """Median valid depth in the box (widened to >= 3 px around its centre column, clipped); None without valid pixels."""
    h, w = depth_img.shape
    c = int((box[0] + box[2]) / 2)
    x1, x2 = max(0, min(int(math.floor(box[0])), c - 1)), min(w, max(int(math.ceil(box[2])), c + 2))
    y1, y2 = max(0, int(math.floor(box[1]))), min(h, int(math.ceil(box[3])))
    v = np.asarray(depth_img[y1:y2, x1:x2], dtype=float).ravel()
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.median(v)) if v.size else None


def hinge_seen(dets, column, depth, fx, tol=0.15, score_min=0.5, depth_img=None, depth_tol=0.3):
    """True when a hinge box (score >= score_min) is centred within fx * tol / depth px of column (and, with depth_img, at that depth)."""
    if column is None:
        return False
    for b, s, l in zip(dets["boxes"], dets["scores"], dets["labels"]):
        if l != "hinge" or s < score_min or abs((b[0] + b[2]) / 2 - column) > fx * tol / depth:
            continue
        if depth_img is not None:
            d = _box_depth(depth_img, b)
            if d is None or abs(d - depth) > depth_tol:
                continue
        return True
    return False


def score_visits(visits):
    """Offline score of a tour: found_first/any over truth visits, false_yes_first/any over the others."""
    pos = [v for v in visits if v["truth"]]
    neg = [v for v in visits if not v["truth"]]
    rate = lambda g, f: sum(1 for v in g if f(v)) / len(g) if g else 0.0
    return {"n_pos": len(pos), "n_neg": len(neg),
            "found_first": rate(pos, lambda v: v["first"]), "found_any": rate(pos, lambda v: any(v["looks"])),
            "false_yes_first": rate(neg, lambda v: v["first"]), "false_yes_any": rate(neg, lambda v: any(v["looks"]))}
