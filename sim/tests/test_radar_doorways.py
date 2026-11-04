# Unit tests for sim/doorsim/radar_doorways.py (no Gazebo). Madelief 2026-11-04: first a geometric doorway finder on the
# radar points (top-down map: wall line + gaps of door width), scored in metres against the doorway truth per distance and
# radar type. Centre error is horizontal only (radar elevation is too coarse for height).
import pathlib, sys
import numpy as np
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import radar_doorways as rd

CAM = [0.0, 0.0, 1.0]


def wall_points(gap=(-0.45, 0.45), x=3.0, span=(-3.0, 3.0), step=0.05, behind=False, noise=0.0, drop=0.0, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(span[0], span[1], step)
    if gap:
        y = y[(y < gap[0]) | (y > gap[1])]
    z = rng.uniform(0.3, 2.0, len(y))
    pts = np.stack([np.full(len(y), x), y, z], 1)
    if behind:  # the far wall of the next room, seen through the doorway
        yb = np.arange(-0.4, 0.4, step)
        pts = np.vstack([pts, np.stack([np.full(len(yb), x + 2.5), yb, np.full(len(yb), 1.0)], 1)])
    pts = pts + rng.normal(0, noise, pts.shape)
    return pts[rng.random(len(pts)) >= drop]


def test_finds_the_doorway():
    got = rd.find(wall_points(), CAM)
    assert len(got) == 1
    d = got[0]
    assert np.linalg.norm(np.array(d["center"][:2]) - [3.0, 0.0]) < 0.1
    assert d["width"] == pytest.approx(0.9, abs=0.15)
    assert d["normal"][0] < -0.9  # wall normal points back to the radar side


def test_points_behind_the_gap_are_fine():
    assert len(rd.find(wall_points(behind=True), CAM)) == 1


def test_too_narrow_or_too_wide_is_no_doorway():
    assert rd.find(wall_points(gap=(-0.15, 0.15)), CAM) == []
    assert rd.find(wall_points(gap=(-1.5, 1.5)), CAM) == []


def test_wall_end_is_no_doorway():
    assert rd.find(wall_points(gap=None, span=(-3.0, 0.0)), CAM) == []


def test_noise_and_dropout():
    got = rd.find(wall_points(noise=0.03, drop=0.3, seed=2), CAM)
    assert len(got) == 1 and np.linalg.norm(np.array(got[0]["center"][:2]) - [3.0, 0.0]) < 0.2


def test_no_points():
    assert rd.find(np.zeros((0, 3)), CAM) == []


def test_ground_truth_and_score():
    meta = {"side": "front", "camera": {"world_position": CAM, "world_rpy": [0, 0, 0]},
            "objects": {"door_a": {"class": 1, "width": 0.9, "height": 2.0,
                                   "approach": {"front": {"point": [3.0, 0.0, 0.0], "direction": [-1.0, 0.0, 0.0]}}},
                        "handle_a": {"class": 2}}}
    gt = rd.ground_truth(meta)
    assert list(gt) == ["door_a"] and gt["door_a"]["center"] == pytest.approx([3.0, 0.0, 1.0])
    rows = rd.score(rd.find(wall_points(), CAM), gt, CAM)
    assert len(rows) == 1 and rows[0]["door"] == "door_a" and rows[0]["found"]
    assert rows[0]["center_m"] < 0.1 and rows[0]["distance_m"] == pytest.approx(3.0, abs=0.05)
    miss = rd.score([], gt, CAM)
    assert miss[0]["found"] is False and miss[0]["center_m"] is None
    far = rd.score([{"center": [3.0, 2.0, 1.0], "width": 0.9, "normal": [-1, 0, 0]}], gt, CAM)
    assert far[0]["found"] is False  # a doorway 2 m off is not this door (match radius rd.MATCH_M)


# --- stacked views (Madelief 2026-11-04): the finder on the radar points of the last n views of the same scene ---
def test_view_windows_same_targets_for_every_n():
    metas = {f"{i:06d}": {"world": "w0", "scene": i // 12, "view": i % 12} for i in range(24)}
    metas["000030"] = {"world": "w0", "scene": 9, "view": 0}  # a scene with one view only
    win = rd.view_windows(metas, [1, 3, 10])
    assert set(win) == {1, 3, 10}
    targets = [t for t, _ in win[10]]
    assert [t for t, _ in win[1]] == targets == [t for t, _ in win[3]]  # all n scored on the same frames
    assert targets == [f"{i:06d}" for i in list(range(9, 12)) + list(range(21, 24))]  # need 9 earlier views of the scene
    t, frames = win[3][0]
    assert t == "000009" and frames == ["000007", "000008", "000009"]  # the n views ending at the target, in view order
    assert win[1][0][1] == ["000009"]
    sided = {f: dict(m, side="front" if int(f) % 2 else "back") for f, m in metas.items()}  # alternate wall sides
    for t, frames in rd.view_windows(sided, [1, 3])[3]:
        assert len({sided[f]["side"] for f in frames}) == 1 and sided[t]["side"] == sided[frames[0]]["side"]  # own room only


def test_stacked_points_fill_the_wall():
    one = wall_points(drop=0.8, seed=1)
    stacked = np.vstack([wall_points(drop=0.8, seed=s) for s in range(1, 11)])
    assert rd.find(one, CAM) == [] or len(rd.find(stacked, CAM)) == 1
    assert len(rd.find(stacked, CAM)) == 1
