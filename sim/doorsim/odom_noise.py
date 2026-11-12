"""Drifting wheel odometry from the true pose (odometry motion model, Thrun et al., Probabilistic Robotics ch. 5, 30 Hz stream).
Per step the true body-frame increment gets a systematic scale (trans_scale, rot_scale) and random noise whose variance grows
linearly with the step (so the drift does not depend on the publish rate). Spec: sim/tests/test_odom_noise.py. numpy only."""
import math

import numpy as np

DEFAULTS = {"trans_scale": 1.01, "rot_scale": 0.98, "k_tt": 4e-4, "k_rr": 1.2e-3, "k_rt": 3e-4, "seed": 0}


def _wrap(a):
    """Angle wrapped to (-pi, pi]."""
    return math.pi - (math.pi - a) % (2 * math.pi)


def increment(prev, cur):
    """(dtrans, dlat, drot): the move prev -> cur in prev's body frame (forward, left) and the wrapped yaw change."""
    dx, dy = cur[0] - prev[0], cur[1] - prev[1]
    c, s = math.cos(prev[2]), math.sin(prev[2])
    return dx * c + dy * s, -dx * s + dy * c, _wrap(cur[2] - prev[2])


def step(pose, inc, params, rng):
    """New (x, y, yaw): pose moved by the scaled + noisy increment, in pose's own body frame."""
    dtrans, dlat, drot = inc
    ts, rs = params["trans_scale"], params["rot_scale"]
    nt = rng.normal(0.0, math.sqrt(params["k_tt"] * abs(dtrans))) if params["k_tt"] > 0 else 0.0
    var_r = params["k_rr"] * abs(drot) + params["k_rt"] * abs(dtrans)
    nr = rng.normal(0.0, math.sqrt(var_r)) if var_r > 0 else 0.0
    dt, dl, dr = dtrans * ts + nt, dlat * ts, drot * rs + nr
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return x + dt * c - dl * s, y + dt * s + dl * c, _wrap(yaw + dr)


class Drift:
    """Feed true poses (x, y, yaw); get the drifting odometry estimate. The first pose is returned unchanged."""

    def __init__(self, params=None):
        self.params = dict(DEFAULTS if params is None else params)
        self.rng = np.random.default_rng(self.params["seed"])
        self.truth = self.est = None

    def update(self, truth):
        if self.est is None:
            self.est = tuple(truth)
        else:
            self.est = step(self.est, increment(self.truth, truth), self.params, self.rng)
        self.truth = tuple(truth)
        return self.est
