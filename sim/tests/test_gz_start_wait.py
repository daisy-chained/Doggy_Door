# Fewer failed Gazebo starts (2026-11-09, Claude). In the v7 render most failed starts were "did not subscribe to
# /window_glass/attach (/glass_b/attach ...) within 60 s": a glass pane's DetachableJoint only subscribes after the simulation has
# run, and the start waited without stepping. Fixes:
#   - gzsim.pane_topics(attrs) -> {pane: (detach, attach)} only for panes that have topics (door glass); panes without (the fixed
#     window glass) are moved aside by set_pose alone in depth_without_panes / seg_without_panes.
#   - Sim._wait_subscribed(pubs, timeout, poll, step_every): waits for every publisher to have a subscriber and steps the sim by 1
#     every step_every polls meanwhile; raises RuntimeError naming the missing topics after timeout. Start uses timeout 120 s.
#   - generate.START_TRIES >= 5, and start_sim waits longer after each failed try (wait * attempt).
# python3 -m pytest sim/tests/test_gz_start_wait.py   (gzsim parts need ROS sourced; skipped otherwise)
import pathlib, sys, threading, types
import numpy as np
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import generate

ATTRS = {"objects": {"glass_a": {"part": "glass_pane", "detach_topic": "/glass_a/detach", "attach_topic": "/glass_a/attach"},
                     "window_glass": {"part": "glass_pane"}, "door_a": {"id": 11}}}


def test_pane_topics_only_for_panes_with_topics():
    gzsim = pytest.importorskip("doorsim.gzsim")
    assert gzsim.pane_topics(ATTRS) == {"glass_a": ("/glass_a/detach", "/glass_a/attach")}


def bare(gzsim, connect_after=None):
    s = object.__new__(gzsim.Sim)
    s.lock, s.latest, s.steps, s.published, s.poses = threading.Lock(), {}, 0, [], []
    def step(n):
        s.steps += n
    s.step = step
    s.connect_after = connect_after
    return s


class Pub:
    def __init__(self, sim, name, after):
        self.sim, self.name, self.after = sim, name, after

    def has_connections(self):
        return self.after is not None and self.sim.steps >= self.after

    def publish(self, m):
        self.sim.published.append(self.name)


def test_wait_subscribed_steps_while_waiting(monkeypatch):
    gzsim = pytest.importorskip("doorsim.gzsim")
    monkeypatch.setattr(gzsim.time, "sleep", lambda s: None)
    s = bare(gzsim)
    pubs = {"/glass_a/attach": Pub(s, "a", 3), "light_config": Pub(s, "l", 0)}
    s._wait_subscribed(pubs, timeout=10, poll=0.25, step_every=2)
    assert s.steps >= 3


def test_wait_subscribed_names_what_is_missing(monkeypatch):
    gzsim = pytest.importorskip("doorsim.gzsim")
    monkeypatch.setattr(gzsim.time, "sleep", lambda s: None)
    s = bare(gzsim)
    pubs = {"/glass_a/attach": Pub(s, "a", None), "light_config": Pub(s, "l", 0)}
    with pytest.raises(RuntimeError, match="/glass_a/attach") as e:
        s._wait_subscribed(pubs, timeout=2, poll=0.25, step_every=2)
    assert "light_config" not in str(e.value) and s.steps > 0


def test_depth_without_panes_moves_a_pane_without_topics(monkeypatch):
    gzsim = pytest.importorskip("doorsim.gzsim")
    s = bare(gzsim)
    s.pane_pub = {"glass_a": (Pub(s, "glass_a/detach", 0), Pub(s, "glass_a/attach", 0))}
    s.model_pose = lambda n: ([1.0, 2.0, 1.0], {"x": 0, "y": 0, "z": 0, "w": 1})
    s.set_pose = lambda n, pos, rpy=None, orientation=None: s.poses.append((n, list(pos)))
    s.render = lambda keys: {"depth": "img", "seg": "img"}
    monkeypatch.setattr(gzsim, "image_array", lambda m, kind: np.zeros((2, 2)))
    s.depth_without_panes(["window_glass", "glass_a"])
    assert s.published == ["glass_a/detach", "glass_a/attach"]  # only the door glass has a joint to undo
    assert ("window_glass", [0, 0, -50]) in s.poses and ("window_glass", [1.0, 2.0, 1.0]) in s.poses
    s.published.clear(); s.poses.clear()
    s.seg_without_panes(["window_glass"])
    assert s.published == [] and ("window_glass", [0, 0, -50]) in s.poses


def test_more_start_tries_with_growing_waits(monkeypatch):
    assert generate.START_TRIES >= 5
    waits = []
    monkeypatch.setattr(generate.time, "sleep", lambda s: waits.append(s))
    n = {"i": 0}
    def make():
        n["i"] += 1
        if n["i"] < 4:
            raise RuntimeError("did not subscribe")
        return "sim"
    assert generate.start_sim(make, "w", wait=5) == "sim"
    assert waits == [5, 10, 15]
