"""A ROS nav_msgs/OccupancyGrid as a hinge_look map dict (numpy only). Contract: sim/tests/test_slam_patrol.py."""
import numpy as np


def from_grid(width, height, res, origin_xy, data):
    """{"occ", "unknown", "res", "origin"}: data is row-major int8 with row 0 at the bottom (origin y); the map has row 0 at the top.
    occ = value >= 50 or unknown; unknown = value < 0."""
    a = np.asarray(data, np.int16).reshape(height, width)[::-1]
    unknown = a < 0
    return {"occ": (a >= 50) | unknown, "unknown": unknown, "res": float(res), "origin": (float(origin_xy[0]), float(origin_xy[1]))}
