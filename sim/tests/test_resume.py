# Resume a render (2026-11-09, Claude): the laptop crashed twice in a day (likely out of memory; earlyoom now prefers killing
# Gazebo), the second time at 5634 / 6000 frames, and a render could only start again from frame 0. With per_scene_rng every
# world has fixed frame numbers (frame_offsets) and fixed seeds, so finished worlds can be kept:
#   generate.frame_done(d): the frame folder has rgb / depth / depth_gt / seg .png and a meta.json that parses (meta is written last).
#   generate.resume_plan(plan, offsets, views_per_scene, out) -> (plan, offsets, kept_frames): drops the worlds whose frames are all
#     done; a world with any missing / broken frame stays in (rendered again at the same numbers).
#   generate.py --resume: an out folder without dataset.json is continued instead of refused (needs per_scene_rng); dataset.json
#     then counts all frames (kept + new) and records "resumed_worlds".
# python3 -m pytest sim/tests/test_resume.py
import json, pathlib, sys
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import generate


def make_frame(out, i, meta=True, broken=False, skip=None):
    d = out / "frames" / f"{i:06d}"
    d.mkdir(parents=True, exist_ok=True)
    for n in ("rgb.png", "depth.png", "depth_gt.png", "seg.png"):
        if n != skip:
            (d / n).write_bytes(b"png")
    if meta:
        (d / "meta.json").write_text('{"a": 1' if broken else '{"a": 1}\n')


def test_frame_done(tmp_path):
    make_frame(tmp_path, 0)
    make_frame(tmp_path, 1, meta=False)
    make_frame(tmp_path, 2, broken=True)
    make_frame(tmp_path, 3, skip="seg.png")
    f = lambda i: generate.frame_done(tmp_path / "frames" / f"{i:06d}")
    assert f(0) and not f(1) and not f(2) and not f(3) and not f(4)


def test_resume_plan_keeps_finished_worlds(tmp_path):
    plan = [({"name": f"w{k}"}, None, None, [2 * k, 2 * k + 1]) for k in range(4)]  # 2 scenes x 3 views = 6 frames each
    offsets = generate.frame_offsets(plan, 3)
    for i in range(0, 6):
        make_frame(tmp_path, i)  # w0 done
    for i in range(6, 11):
        make_frame(tmp_path, i)  # w1 misses its last frame
    for i in range(12, 18):
        make_frame(tmp_path, i, broken=(i == 14))  # w2 has a broken meta
    for i in range(18, 24):
        make_frame(tmp_path, i)  # w3 done
    p, o, kept = generate.resume_plan(plan, offsets, 3, tmp_path)
    assert [w["name"] for w, *_ in p] == ["w1", "w2"] and o == [6, 12] and kept == 12


def test_resume_plan_nothing_rendered(tmp_path):
    plan = [({"name": "w0"}, None, None, [0])]
    assert generate.resume_plan(plan, [0], 4, tmp_path) == (plan, [0], 0)


def test_main_has_resume_flag():
    import inspect
    src = inspect.getsource(generate.main)
    assert '"--resume"' in src and "resume_plan(" in src and "resumed_worlds" in src
