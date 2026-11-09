# Render survives Gazebo hiccups mid-world (2026-11-09, Claude): the v7 6k render died three times overnight on a message that
# hadn't arrived yet (a door's joint_state -> "'NoneType' object has no attribute 'joint'", or pose/info -> KeyError 'poses'),
# and each try started again from frame 0. Two fixes:
#   1. generate.render_world: an error while rendering a world's scenes (not a start: that is start_sim's job) stops that Gazebo,
#      starts it again and renders the world's scenes again from the same frame number (run.count, run.scenes_done and
#      run.rejected put back as they were before the world), up to WORLD_TRIES (>= 3) tries; then the error is raised.
#      Scenes use per-scene seeds (per_scene_rng), so a redo gives the same frames. KeyboardInterrupt / SystemExit are not retried.
#   2. gzsim.Sim: set_doors keeps stepping while a door has no joint_state yet (a missing message = not settled yet) and raises a
#      RuntimeError naming the door if one never comes; model_pose steps a little while pose/info hasn't arrived yet and raises
#      a RuntimeError if it never does. Both errors are then retried by (1).
# python3 -m pytest sim/tests/test_render_world_retry.py   (part 2 needs ROS sourced for gz; skipped otherwise)
import pathlib, sys, threading, types
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import generate


def fake_gzsim():
    calls = {"starts": 0, "stops": 0}

    class Sim:
        def __init__(self, *a, **k):
            calls["starts"] += 1

        def camera_info(self):
            return {"K": [1]}

        def stop(self):
            calls["stops"] += 1
    return types.SimpleNamespace(Sim=Sim), calls


def setup(monkeypatch, render):
    mod, calls = fake_gzsim()
    monkeypatch.setitem(sys.modules, "doorsim.gzsim", mod)
    monkeypatch.setattr(sys.modules["doorsim"], "gzsim", mod, raising=False)
    monkeypatch.setattr(generate, "render_scenes", render)
    monkeypatch.setattr(generate.time, "sleep", lambda s: None, raising=False)
    return calls


def run_state():
    return types.SimpleNamespace(cfg={}, count=100, scenes_done=3, rejected=7)


@pytest.mark.parametrize("err", [AttributeError("'NoneType' object has no attribute 'joint'"), KeyError("poses"),
                                 TimeoutError("sim clock did not advance"), RuntimeError("no joint_state for door_a")])
def test_world_redone_after_a_hiccup(monkeypatch, tmp_path, err):
    seen = []

    def render(sim, attrs, scenes, run, w):
        seen.append((run.count, run.scenes_done, run.rejected))
        run.count += 20; run.scenes_done += 1; run.rejected += 2  # part of the world got rendered
        if len(seen) == 1:
            raise err
        run.count += 20; run.scenes_done += 1
    calls = setup(monkeypatch, render)
    run = run_state()
    cam = generate.render_world({"name": "w"}, tmp_path / "w.sdf", {}, [0, 1], run, tmp_path)
    assert cam == {"K": [1]}
    assert seen == [(100, 3, 7), (100, 3, 7)]  # second try starts from the same frame number and counters
    assert (run.count, run.scenes_done, run.rejected) == (140, 5, 9)
    assert calls["starts"] == 2 and calls["stops"] == 2


def test_world_gives_up_after_the_tries(monkeypatch, tmp_path):
    n = []

    def render(*a):
        n.append(1)
        raise AttributeError("'NoneType' object has no attribute 'joint'")
    calls = setup(monkeypatch, render)
    with pytest.raises(AttributeError):
        generate.render_world({"name": "w"}, tmp_path / "w.sdf", {}, [0], run_state(), tmp_path)
    assert len(n) == generate.WORLD_TRIES >= 3 and calls["starts"] == calls["stops"] == generate.WORLD_TRIES


@pytest.mark.parametrize("err", [KeyboardInterrupt, SystemExit])
def test_stop_signals_are_not_retried(monkeypatch, tmp_path, err):
    n = []

    def render(*a):
        n.append(1)
        raise err()
    calls = setup(monkeypatch, render)
    with pytest.raises(err):
        generate.render_world({"name": "w"}, tmp_path / "w.sdf", {}, [0], run_state(), tmp_path)
    assert len(n) == 1 and calls["stops"] == 1


# --- gzsim: a missing message is waited for, not a crash ---
def bare_sim(gzsim, latest, on_step=None):
    s = object.__new__(gzsim.Sim)
    s.lock, s.latest = threading.Lock(), latest
    s.doors = {"door_a": {"joint": {"name": "hinge"}}}
    s.door_pub = {"door_a": types.SimpleNamespace(publish=lambda m: None)}
    s.steps = 0

    def step(n):
        s.steps += n
        if on_step:
            on_step(s)
    s.step = step
    return s


def js(pos, vel=0.0):
    return types.SimpleNamespace(joint=[types.SimpleNamespace(name="hinge", axis1=types.SimpleNamespace(position=pos, velocity=vel))])


def test_set_doors_waits_for_a_late_joint_state():
    gzsim = pytest.importorskip("doorsim.gzsim")
    def arrive(s):
        if s.steps >= 1500:
            s.latest["js_door_a"] = js(0.5)
    s = bare_sim(gzsim, {}, arrive)
    assert s.set_doors({"door_a": 0.5}, max_steps=6000, chunk=500) == {"door_a": 0.5}


def test_set_doors_names_the_door_when_no_joint_state_comes():
    gzsim = pytest.importorskip("doorsim.gzsim")
    s = bare_sim(gzsim, {})
    with pytest.raises(RuntimeError, match="door_a"):
        s.set_doors({"door_a": 0.5}, max_steps=2000, chunk=500)


def test_model_pose_waits_for_pose_info():
    gzsim = pytest.importorskip("doorsim.gzsim")
    pose = types.SimpleNamespace(name="cam", position=types.SimpleNamespace(x=1, y=2, z=3),
                                 orientation=types.SimpleNamespace(x=0, y=0, z=0, w=1))
    def arrive(s):
        if s.steps >= 30:
            s.latest["poses"] = types.SimpleNamespace(pose=[pose])
    s = bare_sim(gzsim, {}, arrive)
    assert s.model_pose("cam")[0] == [1, 2, 3]
    s2 = bare_sim(gzsim, {})
    with pytest.raises(RuntimeError, match="pose"):
        s2.model_pose("cam")
