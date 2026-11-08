# Train on part of the train split (frame-count test, Madelief 2026-11-08: "test to see how many of these frames we actually need").
#   props.DEFAULTS["train_fraction"] = 1.0 (all, as before). A value in (0, 1) keeps that share of the train split's scenes
#   (whole scenes, so views of one scene never split; images without "scene" count as their own scene), picked by a seeded shuffle of
#   the sorted scene ids: smaller fractions are subsets of bigger ones (nested), same seed = same scenes. val / test stay whole.
#   CropDataset(..., fraction=1.0, fraction_seed=0) does the picking; train() passes cfg["train_fraction"] and cfg["seed"] to the train
#   set only and logs param "train_fraction" plus metric "train_crops".
import json, pathlib, sys
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from doordet import props
from test_props import make_export


def with_scenes(root, per_scene=2):
    for split in ("train", "val", "test"):
        p = root / "coco" / f"{split}.json"
        c = json.loads(p.read_text())
        for im in c["images"]:
            im["scene"] = im["id"] // per_scene
        p.write_text(json.dumps(c))
    return root


def scenes_of(ds, root, split="train"):
    c = json.loads((root / "coco" / f"{split}.json").read_text())
    sc = {im["id"]: im["scene"] for im in c["images"]}
    return {sc[i] for i in ds.image_ids}


def test_default_all():
    assert props.DEFAULTS["train_fraction"] == 1.0
    assert props.resolve({"export": "x", "crop": "door"})["train_fraction"] == 1.0


def test_fraction_keeps_whole_scenes_nested_and_seeded(tmp_path):
    root = with_scenes(make_export(tmp_path / "e", n=16))  # 8 scenes of 2 images
    full = props.CropDataset(root, "train", "door", 64, 0.1, 4)
    all_scenes = scenes_of(full, root)
    assert len(all_scenes) >= 4
    half = props.CropDataset(root, "train", "door", 64, 0.1, 4, fraction=0.5, fraction_seed=0)
    quarter = props.CropDataset(root, "train", "door", 64, 0.1, 4, fraction=0.25, fraction_seed=0)
    s_half, s_q = scenes_of(half, root), scenes_of(quarter, root)
    assert len(s_half) == round(0.5 * len(all_scenes)) and 1 <= len(s_q) <= len(s_half)
    assert s_q <= s_half <= all_scenes  # nested
    for s in s_half:  # whole scenes: every crop of a kept scene stays
        assert sum(1 for i in half.image_ids if i // 2 == s) == sum(1 for i in full.image_ids if i // 2 == s)
    again = props.CropDataset(root, "train", "door", 64, 0.1, 4, fraction=0.5, fraction_seed=0)
    assert again.items == half.items
    other = props.CropDataset(root, "train", "door", 64, 0.1, 4, fraction=0.5, fraction_seed=1)
    assert len(scenes_of(other, root)) == len(s_half)


def test_fraction_one_unchanged(tmp_path):
    root = with_scenes(make_export(tmp_path / "e"))
    a = props.CropDataset(root, "train", "door", 64, 0.1, 4)
    b = props.CropDataset(root, "train", "door", 64, 0.1, 4, fraction=1.0, fraction_seed=5)
    assert a.items == b.items


def test_no_scene_field_counts_each_image(tmp_path):
    root = make_export(tmp_path / "e")  # images without "scene"
    full = props.CropDataset(root, "train", "door", 64, 0.1, 4)
    half = props.CropDataset(root, "train", "door", 64, 0.1, 4, fraction=0.5)
    assert 0 < len(set(half.image_ids)) < len(set(full.image_ids))


def test_train_uses_fraction_on_train_only(tmp_path):
    import mlflow
    root = with_scenes(make_export(tmp_path / "e"))
    cfg = props.resolve({"export": str(root), "crop": "door", "backbone": "resnet18", "pretrained": False, "imgsz": 64, "epochs": 1,
                         "batch": 4, "device": "cpu", "workers": 0, "out": str(tmp_path / "runs"), "train_fraction": 0.5,
                         "mlflow": {"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}", "experiment": "test"}})
    r = props.train(cfg, run_name="unit")
    run = mlflow.get_run(r["run_id"])
    assert float(run.data.params["train_fraction"]) == 0.5
    full = props.CropDataset(root, "train", "door", 64, 0.1, 4)
    assert 0 < run.data.metrics["train_crops"] < len(full)
