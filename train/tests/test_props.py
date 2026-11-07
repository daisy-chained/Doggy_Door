# Unit tests for train/doordet/props (door property + handle type classifier on crops; no GPU, no downloads):
# train/.venv/bin/python -m pytest train/tests
import json, math, pathlib, sys
import numpy as np
import pytest
import torch
from PIL import Image

TRAIN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))
from doordet import props

W, H = 160, 120
DOOR_A = {"opening": "push", "hinge_side": "right", "door_type": "hinged", "glass": True}
DOOR_B = {"opening": "slide", "slide_direction": "left", "door_type": "sliding", "glass": False}


def make_export(root, n=4):
    """Exporter-like folder: images/<split>/*.png + coco/<split>.json with attributes on each box."""
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        coco = {"images": [], "categories": [{"id": 1, "name": "door"}, {"id": 2, "name": "handle"}], "annotations": []}
        for i in range(n):
            fn = f"images/{split}/{i:06d}.png"
            img = np.zeros((H, W, 3), np.uint8); img[10:110, 10:60] = (200, 0, 0); img[10:110, 90:140] = (0, 0, 200)
            Image.fromarray(img).save(root / fn)
            coco["images"].append({"id": i, "file_name": fn, "width": W, "height": H})
            for obj, cat, bbox, attrs in (("door_a", 1, [10, 10, 50, 100], DOOR_A), ("handle_a", 2, [50, 60, 6, 3], {"handle_type": "lever", "door": "door_a"}),
                                          ("door_b", 1, [90, 10, 50, 100], DOOR_B), ("handle_b", 2, [92, 60, 3, 8], {"handle_type": "flush_pull", "door": "door_b"}),
                                          ("door_c", 1, [0, 0, 4, 5], DOOR_A)):  # too small: skipped
                coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": i, "category_id": cat, "bbox": bbox,
                                            "area": bbox[2] * bbox[3], "iscrowd": 0, "object": obj, "truncated": False,
                                            "size_px": bbox[2:], "attributes": attrs})
        (root / "coco").mkdir(exist_ok=True)
        (root / "coco" / f"{split}.json").write_text(json.dumps(coco))
    (root / "export.json").write_text(json.dumps({"source": {"path": "/x", "git": {"commit": "abc"}}, "seed": 0}))
    return root


@pytest.fixture
def export_dir(tmp_path):
    return make_export(tmp_path / "export")


# --- labels ---
def test_labels_from_attributes():
    assert props.labels(DOOR_A, "door") == {"opening": "push", "hinge_side": "right", "glass": "yes", "door_type": "hinged"}
    assert props.labels(DOOR_B, "door") == {"opening": "slide", "hinge_side": "none", "glass": "no", "door_type": "sliding"}  # sliding: no hinge
    assert props.labels({"handle_type": "knob"}, "handle") == {"handle_type": "knob"}
    assert set(props.HEADS["door"]) == {"opening", "hinge_side", "glass", "door_type"} and set(props.HEADS["handle"]) == {"handle_type"}
    assert props.HEADS["handle"]["handle_type"] == ["lever", "knob", "flush_pull"]


# --- crops ---
def test_crop_box_margin_and_square():
    x0, y0, x1, y1 = props.crop_box([40, 30, 20, 40], margin=0.25, rng=None, jitter=None)
    assert (x1 - x0) == pytest.approx(y1 - y0)                    # square around the box
    assert (x1 - x0) == pytest.approx(40 * 1.5)                   # long side + margin on both sides
    assert (x0 + x1) / 2 == pytest.approx(50) and (y0 + y1) / 2 == pytest.approx(50)


def test_crop_box_jitter_seeded_and_bounded():
    jit = {"shift": 0.1, "scale": 0.15}
    a = props.crop_box([40, 30, 20, 40], 0.25, np.random.default_rng(1), jit)
    b = props.crop_box([40, 30, 20, 40], 0.25, np.random.default_rng(1), jit)
    assert a == b
    for s in range(50):
        x0, y0, x1, y1 = props.crop_box([40, 30, 20, 40], 0.25, np.random.default_rng(s), jit)
        side = x1 - x0
        assert 60 * 0.85 - 1e-6 <= side <= 60 * 1.15 + 1e-6
        assert abs((x0 + x1) / 2 - 50) <= 0.1 * 40 + 1e-6


def test_dataset_items(export_dir):
    ds = props.CropDataset(export_dir, "train", "door", imgsz=64, margin=0.1, min_px=8)
    assert len(ds) == 8                                           # 2 doors x 4 images, tiny door_c skipped
    x, y = ds[0]
    assert x.shape == (3, 64, 64) and x.dtype == torch.float32
    assert set(y) == set(props.HEADS["door"]) and all(isinstance(v, int) for v in y.values())
    h = props.CropDataset(export_dir, "train", "handle", imgsz=32, margin=0.5, min_px=3)
    assert len(h) == 8 and h[0][0].shape == (3, 32, 32)


def test_dataset_jitter_only_when_training(export_dir):
    ev = props.CropDataset(export_dir, "val", "door", imgsz=64, margin=0.1, min_px=8)
    assert torch.equal(ev[0][0], ev[0][0])                         # no jitter in eval
    tr = props.CropDataset(export_dir, "train", "door", imgsz=64, margin=0.1, min_px=8, jitter={"shift": 0.2, "scale": 0.2}, seed=0)
    assert not torch.equal(tr[0][0], tr[0][0])                     # a new jitter every time an item is read


# --- model ---
def test_model_one_head_per_property_and_subset():
    m = props.Model("resnet18", props.HEADS["door"], pretrained=False)
    out = m(torch.zeros(2, 3, 64, 64))
    assert {k: v.shape for k, v in out.items()} == {"opening": (2, 3), "hinge_side": (2, 3), "glass": (2, 2), "door_type": (2, 2)}
    one = props.Model("resnet18", {"glass": props.HEADS["door"]["glass"]}, pretrained=False)
    assert set(one(torch.zeros(1, 3, 64, 64))) == {"glass"}        # a single head = a separate model


# --- confidence ---
def test_ece_perfect_and_overconfident():
    conf, correct = np.array([1.0, 1.0, 1.0, 1.0]), np.array([1, 1, 1, 1])
    assert props.ece(conf, correct) == pytest.approx(0.0)
    assert props.ece(np.full(10, 0.9), np.array([1] * 5 + [0] * 5)) == pytest.approx(0.4)


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 600)
    logits = rng.normal(0, 1, (600, 3)); logits[np.arange(600), y] += 1.0
    logits *= 6                                                   # far too confident
    T = props.fit_temperature(torch.tensor(logits, dtype=torch.float32), torch.tensor(y))
    assert T > 2
    p0, p1 = torch.softmax(torch.tensor(logits), 1).numpy(), torch.softmax(torch.tensor(logits) / T, 1).numpy()
    c = p0.argmax(1) == y
    assert props.ece(p1.max(1), c) < props.ece(p0.max(1), c)


# --- training ---
def test_train_cpu_logs_per_head_accuracy_and_calibration(tmp_path, export_dir):
    import mlflow
    cfg = props.resolve({"export": str(export_dir), "crop": "door", "backbone": "resnet18", "pretrained": False,
                         "imgsz": 64, "epochs": 2, "batch": 4, "device": "cpu", "workers": 0, "out": str(tmp_path / "runs"),
                         "mlflow": {"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}", "experiment": "test"}})
    r = props.train(cfg, run_name="unit")
    run = mlflow.get_run(r["run_id"])
    for h in props.DEFAULT_HEADS["door"]:
        assert f"val_acc_{h}" in run.data.metrics and f"test_acc_{h}" in run.data.metrics
        assert f"test_ece_{h}" in run.data.metrics and h in r["temperature"]
    assert (pathlib.Path(r["out"]) / "best" / "model.pt").exists() and (pathlib.Path(r["out"]) / "best" / "heads.json").exists()
    assert run.data.params["crop"] == "door"


def test_resolve_defaults_per_crop():
    d = props.resolve({"export": "x", "crop": "door"})
    h = props.resolve({"export": "x", "crop": "handle"})
    assert d["heads"] == props.DEFAULT_HEADS["door"] and h["heads"] == ["handle_type"]
    assert h["margin"] > d["margin"] and d["jitter"] == {"shift": 0.1, "scale": 0.15}
    assert props.resolve({"export": "x", "crop": "door", "heads": ["glass"]})["heads"] == ["glass"]
    with pytest.raises(ValueError):
        props.resolve({"export": "x", "crop": "door", "heads": ["handle_type"]})


def test_workers_jitter_differs_per_worker_and_epoch(export_dir):
    ds = props.CropDataset(export_dir, "train", "door", imgsz=32, margin=0.1, min_px=8, jitter={"shift": 0.2, "scale": 0.2}, seed=0)
    gen = torch.Generator().manual_seed(0)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, generator=gen, worker_init_fn=props.reseed_worker)
    e1, e2 = [x for x, _ in dl], [x for x, _ in dl]
    assert not torch.equal(e1[0], e2[0])            # a new jitter for the same crop in the next epoch


# --- held-out property combos (door room v4): test accuracy also per group ---
def test_train_reports_seen_and_unseen_combos_on_test(tmp_path):
    import mlflow
    root = make_export(tmp_path / "export")
    coco = json.loads((root / "coco" / "test.json").read_text())
    for a in coco["annotations"]:
        a["attributes"] = dict(a["attributes"], heldout=a["object"] in ("door_b", "handle_b"))
    (root / "coco" / "test.json").write_text(json.dumps(coco))
    cfg = props.resolve({"export": str(root), "crop": "door", "backbone": "resnet18", "pretrained": False,
                         "imgsz": 64, "epochs": 1, "batch": 4, "device": "cpu", "workers": 0, "out": str(tmp_path / "runs"),
                         "mlflow": {"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}", "experiment": "test"}})
    r = props.train(cfg, run_name="unit")
    for h in props.DEFAULT_HEADS["door"]:
        assert f"test_acc_seen_{h}" in r["test"] and f"test_acc_unseen_{h}" in r["test"]
        # 4 seen (door_a) + 4 unseen (door_b) crops: the overall accuracy is their mean
        assert r["test"][f"test_acc_{h}"] == pytest.approx((r["test"][f"test_acc_seen_{h}"] + r["test"][f"test_acc_unseen_{h}"]) / 2)


def test_train_no_group_metrics_without_heldout(tmp_path, export_dir):
    cfg = props.resolve({"export": str(export_dir), "crop": "door", "backbone": "resnet18", "pretrained": False,
                         "imgsz": 64, "epochs": 1, "batch": 4, "device": "cpu", "workers": 0, "out": str(tmp_path / "runs"),
                         "mlflow": {"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}", "experiment": "test"}})
    r = props.train(cfg, run_name="unit")
    assert not any("_seen_" in k or "_unseen_" in k for k in r["test"])


# --- scored on the detector's own boxes (Madelief 2026-11-02: true boxes and detector boxes) ---
def test_score_on_detector_boxes(tmp_path):
    root = make_export(tmp_path / "export")
    cfg = props.resolve({"export": str(root), "crop": "door", "backbone": "resnet18", "pretrained": False,
                         "imgsz": 64, "epochs": 1, "batch": 4, "device": "cpu", "workers": 0, "out": str(tmp_path / "runs"),
                         "mlflow": {"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}", "experiment": "test"}})
    r = props.train(cfg, run_name="unit")
    # detector output per image id: door_a found (slightly off), door_b missed in image 0, plus a false door far from any door
    preds = {}
    for i in range(4):
        boxes = [[12, 11, 12 + 48, 11 + 99], [70, 0, 80, 5]]
        scores, labels = [0.9, 0.8], [0, 0]
        if i > 0:
            boxes.append([91, 10, 91 + 50, 10 + 100]); scores.append(0.85); labels.append(0)
        preds[i] = {"boxes": np.array(boxes, float), "scores": np.array(scores), "labels": np.array(labels)}
    s = props.score_on_boxes(r["out"], root, "test", preds, score_min=0.3, iou=0.5)
    assert s["crops"] == 8 and s["matched"] == 7 and s["missed"] == 1 and s["false"] == 4
    for h in props.DEFAULT_HEADS["door"]:
        assert 0.0 <= s[f"acc_{h}"] <= 1.0
    # against the true-box accuracy the same keys exist, so the two can be compared side by side
    assert set(k for k in s if k.startswith("acc_") and not k.endswith(("_rule", "_combined"))) == {f"acc_{h}" for h in props.DEFAULT_HEADS["door"]}
