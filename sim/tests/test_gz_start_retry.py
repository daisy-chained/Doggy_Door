# Unit test for the Gazebo start retry (2026-11-07, Claude): with several Gazebo servers starting at once a start can time out
# ("did not subscribe ... within 60 s"), and one slow start used to kill a whole 2000-frame render. render_world now starts a
# world's Gazebo again (up to START_TRIES) when the start fails; nothing was rendered yet, so the retry is exact.
# python3 -m pytest sim/tests/test_gz_start_retry.py
import pathlib, sys, types
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import generate


def fake_gzsim(fail_times):
    calls = {"starts": 0, "stops": 0}

    class Sim:
        def __init__(self, *a, **k):
            calls["starts"] += 1
            if calls["starts"] <= fail_times:
                raise (TimeoutError("sim clock did not advance") if calls["starts"] % 2 else RuntimeError("did not subscribe within 60 s"))

        def camera_info(self):
            return {"K": [1]}

        def stop(self):
            calls["stops"] += 1
    return types.SimpleNamespace(Sim=Sim), calls


@pytest.mark.parametrize("fails", [0, 1, 2])
def test_render_world_retries_a_failed_start(monkeypatch, tmp_path, fails):
    mod, calls = fake_gzsim(fails)
    monkeypatch.setitem(sys.modules, "doorsim.gzsim", mod)
    monkeypatch.setattr(sys.modules["doorsim"], "gzsim", mod, raising=False)
    monkeypatch.setattr(generate, "render_scenes", lambda *a, **k: None)
    monkeypatch.setattr(generate.time, "sleep", lambda s: None, raising=False)
    cam = generate.render_world({"name": "w"}, tmp_path / "w.sdf", {}, [], types.SimpleNamespace(cfg={}), tmp_path)
    assert cam == {"K": [1]} and calls["starts"] == fails + 1


def test_render_world_gives_up_after_the_tries(monkeypatch, tmp_path):
    mod, calls = fake_gzsim(99)
    monkeypatch.setitem(sys.modules, "doorsim.gzsim", mod)
    monkeypatch.setattr(sys.modules["doorsim"], "gzsim", mod, raising=False)
    monkeypatch.setattr(generate, "render_scenes", lambda *a, **k: None)
    monkeypatch.setattr(generate.time, "sleep", lambda s: None, raising=False)
    with pytest.raises((RuntimeError, TimeoutError)):
        generate.render_world({"name": "w"}, tmp_path / "w.sdf", {}, [], types.SimpleNamespace(cfg={}), tmp_path)
    assert calls["starts"] == generate.START_TRIES >= 3
