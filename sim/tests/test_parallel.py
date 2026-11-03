# Unit tests for parallel rendering (generator: several Gazebo runs at once, one world each) and the parallel effects
# step (frames over a process pool); no Gazebo needed: python3 -m pytest sim/tests
import json, pathlib, sys
import numpy as np
import pytest
from PIL import Image

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(SIM / "tests"))
from doorsim import effects, generate
from test_effects import CFG, make_dataset

WORLDS = [{"name": f"w{i}", "split": "train", "heldout": False} for i in range(7)]


# --- generator: which worlds go to which Gazebo run, and where their frames go ---
def test_frame_offsets_match_the_sequential_numbering():
    plan = generate.world_scenes(WORLDS, 2)
    # sequential rendering numbers frames scene by scene, views_per_scene each: world i starts at i * 2 * views
    assert generate.frame_offsets(plan, 5) == [0, 10, 20, 30, 40, 50, 60]
    plan = generate.world_scenes(WORLDS[:3], 3)
    assert generate.frame_offsets(plan, 4) == [0, 12, 24]


def test_assign_workers_every_world_once_balanced_and_fixed():
    plan = generate.world_scenes(WORLDS, 2)
    jobs = generate.assign_workers(plan, 3)
    assert len(jobs) == 3
    flat = sorted(i for job in jobs for i in job)
    assert flat == list(range(len(plan)))  # plan indices, each once
    sizes = [len(j) for j in jobs]
    assert max(sizes) - min(sizes) <= 1
    assert generate.assign_workers(plan, 3) == jobs
    assert generate.assign_workers(plan, 1) == [list(range(len(plan)))]
    assert len(generate.assign_workers(plan, 20)) == len(plan)  # never more runs than worlds


def test_worker_partitions_unique_and_set_for_gazebo():
    names = {generate.worker_partition(i) for i in range(8)}
    assert len(names) == 8 and all(n for n in names)
    env = generate.worker_env(3, {"PATH": "/usr/bin", "GZ_PARTITION": "old"})
    assert env["GZ_PARTITION"] == generate.worker_partition(3) and env["PATH"] == "/usr/bin"


def test_workers_setting():
    assert generate.n_workers({"workers": 4}, None) == 4
    assert generate.n_workers({"workers": 4}, 2) == 2  # --workers wins
    assert generate.n_workers({}, None) == 1
    with pytest.raises(ValueError):
        generate.n_workers({}, 0)


def test_v4_configs_have_workers():
    import yaml
    for name in ("door_room_v4_small", "door_room_v4_2k"):
        c = yaml.safe_load((SIM / "configs" / f"{name}.yaml").read_text())
        assert c["workers"] >= 2


# --- effects: the same frames whatever the number of workers ---
def files(root):
    out = {}
    for f in sorted((root / "frames").iterdir()):
        out[f.name] = (np.asarray(Image.open(f / "rgb.png")).tobytes(), np.asarray(Image.open(f / "depth.png")).tobytes(),
                       json.loads((f / "meta.json").read_text()), (f / "seg.png").is_symlink())
    return out


def test_effects_workers_give_identical_output(tmp_path):
    src = make_dataset(tmp_path / "src", n_scenes=6, views=2)
    one = effects.apply_dataset(src, tmp_path / "one", CFG, workers=1)
    three = effects.apply_dataset(src, tmp_path / "three", CFG, workers=3)
    assert files(tmp_path / "one") == files(tmp_path / "three")
    assert one["frames"] == three["frames"] == 12
    assert {k: v for k, v in one.items() if k != "effects"} == {k: v for k, v in three.items() if k != "effects"}


def test_effects_workers_on_a_split(tmp_path):
    src = make_dataset(tmp_path / "src", n_scenes=10, views=2)
    effects.apply_dataset(src, tmp_path / "a", CFG, visibility=2.0, split="test", workers=1)
    effects.apply_dataset(src, tmp_path / "b", CFG, visibility=2.0, split="test", workers=2)
    assert files(tmp_path / "a") == files(tmp_path / "b")


def test_effects_default_workers():
    n = effects.default_workers()
    assert 1 <= n <= 8
