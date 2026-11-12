"""A ROS nav_msgs/OccupancyGrid as a hinge_look map dict (numpy only). Contract: sim/tests/test_slam_patrol.py."""
import numpy as np


def from_grid(width, height, res, origin_xy, data):
    """{"occ", "unknown", "res", "origin"}: data is row-major int8 with row 0 at the bottom (origin y); the map has row 0 at the top.
    occ = value >= 50 or unknown; unknown = value < 0."""
    a = np.asarray(data, np.int16).reshape(height, width)[::-1]
    unknown = a < 0
    return {"occ": (a >= 50) | unknown, "unknown": unknown, "res": float(res), "origin": (float(origin_xy[0]), float(origin_xy[1]))}


def _window_any(a, k):
    """bool HxW: any True in the (2k+1) square window around each cell (cells outside the map are False)."""
    h, w = a.shape
    p = np.pad(a.astype(np.int32), k)
    ii = np.zeros((h + 2 * k + 1, w + 2 * k + 1), np.int32)
    ii[1:, 1:] = p.cumsum(0).cumsum(1)
    n = 2 * k + 1
    return (ii[n:n + h, n:n + w] - ii[:h, n:n + w] - ii[n:n + h, :w] + ii[:h, :w]) > 0


def fill_gaps(m, cells=2):
    """New map dict: unknown cells inside the closing (dilation then erosion, (2*cells+1) square, outside the map = not free in
    both steps) of the known-free mask become known free. Known occupied cells never change; input untouched."""
    free = ~m["occ"] & ~m["unknown"]
    dil = _window_any(free, cells)
    pad = np.pad(dil, cells, constant_values=False)
    h, w = free.shape
    n = 2 * cells + 1
    ero = np.ones((h, w), bool)
    for dr in range(n):
        for dc in range(n):
            ero &= pad[dr:dr + h, dc:dc + w]
    fill = ero & m["unknown"]
    return {**m, "occ": m["occ"] & ~fill, "unknown": m["unknown"] & ~fill}
