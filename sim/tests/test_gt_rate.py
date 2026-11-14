# Ground-truth camera rate (Madelief 2026-11-14: speed up the runs). python3 -m pytest sim/tests/test_gt_rate.py
import pathlib, sys, xml.etree.ElementTree as ET
SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import robot

PLAT = robot.load_platform(SIM / "configs" / "robots" / "wheeled_ring.yaml")


def rates(m):
    return {s.get("name"): s.findtext("update_rate") for s in m.iter("sensor")}


def test_default_unchanged(monkeypatch):
    monkeypatch.delenv("DOORSIM_GT_HZ", raising=False)
    r = rates(ET.fromstring(robot.robot_sdf(PLAT, "robot")).find("model"))
    assert r["rgbd"] == r["boxes"] == r["boxes3d"] == r["seg"] == "10"
    assert r["rgbd_1"] == r["seg_1"] == r["boxes_1"] == "2"


def test_env_lowers_only_ground_truth(monkeypatch):
    monkeypatch.setenv("DOORSIM_GT_HZ", "2")
    r = rates(ET.fromstring(robot.robot_sdf(PLAT, "robot")).find("model"))
    assert r["rgbd"] == "10" and r["boxes"] == r["boxes3d"] == r["seg"] == "2"
    assert r["rgbd_1"] == r["seg_1"] == "2"        # ring already 2 Hz: never raised
    monkeypatch.setenv("DOORSIM_GT_HZ", "2.5")
    assert rates(ET.fromstring(robot.robot_sdf(PLAT, "robot")).find("model"))["seg"] == "2.5"
