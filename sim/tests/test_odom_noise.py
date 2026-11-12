# Drifting wheel odometry (Madelief 2026-11-12: "do a sim run where we remove its perfect position and require it to figure it out
# themselves"). Until now /odom + odom->base_link were Gazebo's TRUE pose. Noise model = Claude decisions (research-based, dev log
# 2026-11-12): the odometry motion model (Thrun et al., Probabilistic Robotics ch. 5) adapted to a 30 Hz stream: per step the true
# body-frame increment (dtrans along the old heading, drot) gets
#   (a) a systematic scale (wheel radius / wheel base calibration error, UMBmark): dtrans * trans_scale, drot * rot_scale;
#   (b) random noise whose VARIANCE grows linearly with the step (a random walk, so the drift does not depend on the publish rate):
#       dtrans += N(0, sqrt(k_tt * |dtrans|)), drot += N(0, sqrt(k_rr * |drot| + k_rt * |dtrans|)).
# No ROS needed: python3 -m pytest sim/tests/test_odom_noise.py
#
# Module sim/doorsim/odom_noise.py (numpy only):
#   DEFAULTS = {"trans_scale": 1.01, "rot_scale": 0.98, "k_tt": 4e-4, "k_rr": 1.2e-3, "k_rt": 3e-4, "seed": 0}
#   increment(prev, cur) -> (dtrans, dlat, drot): prev / cur = (x, y, yaw) true poses; the move expressed in prev's body frame
#     (dtrans forward, dlat to the left), drot = cur yaw - prev yaw wrapped to (-pi, pi].
#   step(pose, inc, params, rng) -> new (x, y, yaw): apply (a) + (b) to inc (dlat gets trans_scale and no noise), then move pose by
#     the noisy increment in pose's own body frame; yaw wrapped to (-pi, pi]. rng = numpy Generator (unused when all k are 0).
#   class Drift(params=None): first update(truth) returns truth unchanged (odometry starts at the true start pose); every later
#     update(truth) -> step(last estimate, increment(last truth, truth), ...) and returns the estimate. Seeded with params["seed"].
#
# Module sim/doorsim/robot.py additions:
#   robot_sdf(p, name="robot", odom_truth=False): odom_truth=True -> the ground-truth OdometryPublisher publishes on /odom_truth with
#     tf_topic /tf_truth (so neither /odom nor /tf carry the true pose); the DiffDrive plugin is unchanged.
#   bridge_args(p, name="robot", odom_truth=False): odom_truth=True -> "/odom_truth@nav_msgs/msg/Odometry[gz.msgs.Odometry"
#     instead of the /odom and /tf entries (the noisy /odom + /tf come from sim/building/odom_noise_node.py).
import math, pathlib, sys, xml.etree.ElementTree as ET
import numpy as np
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import odom_noise, robot

RING = SIM / "configs" / "robots" / "wheeled_ring.yaml"
CLEAN = {"trans_scale": 1.0, "rot_scale": 1.0, "k_tt": 0.0, "k_rr": 0.0, "k_rt": 0.0, "seed": 0}


def path(n=300, rate=30.0, v=0.4, w=0.3, start=(1.0, 2.0, 0.5)):
    """True poses of a robot driving an arc at v m/s, w rad/s, sampled at `rate` Hz."""
    x, y, a = start
    out = [(x, y, a)]
    dt = 1.0 / rate
    for _ in range(n):
        x += v * dt * math.cos(a + w * dt / 2); y += v * dt * math.sin(a + w * dt / 2); a = math.atan2(math.sin(a + w * dt), math.cos(a + w * dt))
        out.append((x, y, a))
    return out


def run(poses, params):
    d = odom_noise.Drift(params)
    return [d.update(p) for p in poses]


def test_defaults():
    assert set(odom_noise.DEFAULTS) == {"trans_scale", "rot_scale", "k_tt", "k_rr", "k_rt", "seed"}


def test_increment_body_frame():
    dt, dl, dr = odom_noise.increment((1.0, 1.0, math.pi / 2), (1.0, 1.5, math.pi / 2 + 0.1))
    assert dt == pytest.approx(0.5) and dl == pytest.approx(0.0, abs=1e-9) and dr == pytest.approx(0.1)
    dt, dl, dr = odom_noise.increment((0.0, 0.0, 0.0), (0.0, 0.2, 0.0))
    assert dt == pytest.approx(0.0, abs=1e-9) and dl == pytest.approx(0.2)
    assert odom_noise.increment((0, 0, 3.1), (0, 0, -3.1))[2] == pytest.approx(2 * math.pi - 6.2)


def test_no_noise_reproduces_truth():
    poses = path()
    est = run(poses, CLEAN)
    assert est[0] == poses[0]
    for e, t in zip(est, poses):
        assert e[0] == pytest.approx(t[0], abs=1e-6) and e[1] == pytest.approx(t[1], abs=1e-6)
        assert math.cos(e[2] - t[2]) == pytest.approx(1.0, abs=1e-9)


def test_systematic_trans_scale():
    poses = [(0.01 * i, 0.0, 0.0) for i in range(1001)]           # 10 m straight
    est = run(poses, dict(CLEAN, trans_scale=1.01))
    assert est[-1][0] == pytest.approx(10.1, abs=1e-6) and est[-1][1] == pytest.approx(0.0, abs=1e-9)


def test_systematic_rot_scale_bends_the_path():
    poses = path(n=600, w=0.3)                                     # 20 s, 6 rad of turning
    est = run(poses, dict(CLEAN, rot_scale=0.98))
    err = math.atan2(math.sin(est[-1][2] - poses[-1][2]), math.cos(est[-1][2] - poses[-1][2]))
    assert err == pytest.approx(-0.02 * 6.0, abs=0.01)


def test_seeded_and_drift_grows():
    poses = path(n=900, w=0.1)                                     # 30 s
    p = dict(odom_noise.DEFAULTS, trans_scale=1.0, rot_scale=1.0)
    assert run(poses, p) == run(poses, p)
    errs = []
    for s in range(40):
        est = run(poses, dict(p, seed=s))
        errs.append([math.hypot(e[0] - t[0], e[1] - t[1]) for e, t in zip(est, poses)])
    errs = np.array(errs)
    assert errs[:, 300].mean() < errs[:, 900].mean()
    assert errs[:, 900].mean() > 0.01


def test_drift_does_not_depend_on_rate():
    p = dict(odom_noise.DEFAULTS, trans_scale=1.0, rot_scale=1.0)
    slow, fast = path(n=300, rate=10.0), path(n=3000, rate=100.0)  # the same 30 s arc
    e_slow = [math.hypot(*np.subtract(run(slow, dict(p, seed=s))[-1][:2], slow[-1][:2])) for s in range(60)]
    e_fast = [math.hypot(*np.subtract(run(fast, dict(p, seed=s))[-1][:2], fast[-1][:2])) for s in range(60)]
    r = np.sqrt(np.mean(np.square(e_fast))) / np.sqrt(np.mean(np.square(e_slow)))
    assert 0.7 < r < 1.4


def _plugin(m, fname):
    return next(pl for pl in m.iter("plugin") if pl.get("filename") == fname)


def test_robot_sdf_odom_truth():
    p = robot.load_platform(RING)
    m = ET.fromstring(robot.robot_sdf(p, odom_truth=True)).find("model")
    truth = _plugin(m, "gz-sim-odometry-publisher-system")
    assert truth.findtext("odom_topic") == "/odom_truth" and truth.findtext("tf_topic") == "/tf_truth"
    m0 = ET.fromstring(robot.robot_sdf(p)).find("model")
    t0 = _plugin(m0, "gz-sim-odometry-publisher-system")
    assert t0.findtext("odom_topic") == "/odom" and t0.findtext("tf_topic") == "/tf"
    assert ET.tostring(_plugin(m, "gz-sim-diff-drive-system")) == ET.tostring(_plugin(m0, "gz-sim-diff-drive-system"))


def test_bridge_args_odom_truth():
    p = robot.load_platform(RING)
    b = robot.bridge_args(p, odom_truth=True)
    assert "/odom_truth@nav_msgs/msg/Odometry[gz.msgs.Odometry" in b
    assert not any(a.startswith("/odom@") or a.startswith("/tf@") for a in b)
    assert robot.bridge_args(p) == robot.bridge_args(p, odom_truth=False)
