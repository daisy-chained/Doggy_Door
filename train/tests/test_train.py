# Unit tests for train/doordet (no GPU, no model downloads): train/.venv/bin/python -m pytest train/tests
import json, pathlib, sys
import numpy as np
import pytest
import torch
from PIL import Image

TRAIN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))
from doordet import adapters, config, data, evaluate, loop

W, H = 64, 48
BOXES = [  # (category_id, [x, y, w, h]) per image, COCO ids as the exporter writes them (1 door, 2 handle)
    [(1, [5, 4, 30, 40]), (2, [28, 20, 3, 6])],
    [(1, [30, 2, 20, 44])],
]


def make_export(root):
    """Minimal exporter output: images/<split>/*.png + coco/<split>.json for train/val/test."""
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        coco = {"images": [], "categories": [{"id": 1, "name": "door"}, {"id": 2, "name": "handle"}], "annotations": []}
        for i, boxes in enumerate(BOXES):
            fn = f"images/{split}/{i:06d}.png"
            Image.fromarray(np.full((H, W, 3), 40 * i, np.uint8)).save(root / fn)
            coco["images"].append({"id": i, "file_name": fn, "width": W, "height": H, "scene": i, "view": 0, "frame": f"{i:06d}"})
            for c, b in boxes:
                coco["annotations"].append({"id": len(coco["annotations"]), "image_id": i, "category_id": c, "bbox": b,
                                            "area": b[2] * b[3], "iscrowd": 0, "segmentation": [], "truncated": False,
                                            "size_px": b[2:], "attributes": {}})
        (root / "coco").mkdir(exist_ok=True)
        (root / "coco" / f"{split}.json").write_text(json.dumps(coco))
    (root / "export.json").write_text(json.dumps({"source": {"path": "/x", "git": {"commit": "abc"}}, "seed": 0}))
    return root


@pytest.fixture
def export_dir(tmp_path):
    return make_export(tmp_path / "export")


# --- config ---
def test_config_defaults_and_overrides(tmp_path, export_dir):
    p = tmp_path / "c.yaml"
    p.write_text(f"export: {export_dir}\nmodel: {{family: hf, checkpoint: some/ckpt}}\nimgsz: 320\n")
    cfg = config.load(p, overrides=["epochs=3", "lr=0.001"])
    assert cfg["imgsz"] == 320 and cfg["epochs"] == 3 and cfg["lr"] == 0.001
    assert cfg["batch"] > 0 and "seed" in cfg and "mlflow" in cfg  # defaults filled in


def test_config_rejects_unknown_key(tmp_path, export_dir):
    p = tmp_path / "c.yaml"
    p.write_text(f"export: {export_dir}\nmodel: {{family: hf, checkpoint: x}}\nimgsize: 320\n")
    with pytest.raises(ValueError, match="imgsize"):
        config.load(p)


# --- data ---
def test_dataset_items_and_label_mapping(export_dir):
    ds = data.CocoDetection(export_dir, "train")
    assert len(ds) == 2
    assert ds.id2label == {0: "door", 1: "handle"}
    img, target = ds[0]
    assert img.size == (W, H)
    assert target["image_id"] == 0
    assert [a["category_id"] for a in target["annotations"]] == [0, 1]  # 0-based for the model
    assert target["annotations"][1]["bbox"] == [28, 20, 3, 6]


# --- prediction -> COCO + eval ---
def test_to_coco_results_converts_xyxy_and_labels():
    preds = {7: {"boxes": np.array([[5, 4, 35, 44]], float), "scores": np.array([0.9]), "labels": np.array([1])}}
    res = evaluate.to_coco_results(preds)
    assert res == [{"image_id": 7, "category_id": 2, "bbox": [5.0, 4.0, 30.0, 40.0], "score": 0.9}]


def perfect_preds(gt_json, drop_class=None):
    gt = json.loads(gt_json.read_text())
    preds = {im["id"]: {"boxes": [], "scores": [], "labels": []} for im in gt["images"]}
    for a in gt["annotations"]:
        if a["category_id"] == drop_class:
            continue
        x, y, w, h = a["bbox"]
        p = preds[a["image_id"]]
        p["boxes"].append([x, y, x + w, y + h]); p["scores"].append(0.99); p["labels"].append(a["category_id"] - 1)
    return {k: {kk: np.array(vv, float) for kk, vv in v.items()} for k, v in preds.items()}


def test_coco_metrics_perfect_and_per_class(export_dir):
    gt = export_dir / "coco" / "val.json"
    m = evaluate.coco_metrics(gt, perfect_preds(gt))
    assert m["map"] == pytest.approx(1.0) and m["map_50"] == pytest.approx(1.0)
    assert m["ap_door"] == pytest.approx(1.0) and m["ap_handle"] == pytest.approx(1.0)
    assert "map_small" in m
    m = evaluate.coco_metrics(gt, perfect_preds(gt, drop_class=2))
    assert m["ap_door"] == pytest.approx(1.0) and m["ap_handle"] == pytest.approx(0.0)


def test_coco_metrics_no_predictions(export_dir):
    gt = export_dir / "coco" / "val.json"
    empty = {i: {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0)} for i in (0, 1)}
    assert evaluate.coco_metrics(gt, empty)["map"] == 0.0


# --- adapters ---
def test_adapter_registry():
    assert "hf" in adapters.FAMILIES
    with pytest.raises(ValueError, match="nope"):
        adapters.get("nope")


class FakeAdapter:
    """Predicts the ground truth after the first epoch; loss = a learnable scalar. Stands in for a real model."""
    def __init__(self, cfg, id2label):
        self.model = torch.nn.Linear(1, 1)
        self.id2label = id2label

    def collate(self, batch):
        return {"targets": [t for _, t in batch], "sizes": [img.size[::-1] for img, _ in batch]}

    def loss(self, inputs, device):
        return (self.model.weight - 1).pow(2).sum()

    def predict(self, inputs, device):
        out = {}
        for t in inputs["targets"]:
            b = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in t["annotations"]]
            out[t["image_id"]] = {"boxes": np.array(b, float).reshape(-1, 4), "scores": np.full(len(b), 0.9),
                                  "labels": np.array([a["category_id"] for a in t["annotations"]])}
        return out

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "weights.txt").write_text("x")


# --- loop ---
def test_training_loop_logs_to_mlflow_and_saves_best(tmp_path, export_dir, monkeypatch):
    import mlflow
    monkeypatch.setitem(adapters.FAMILIES, "fake", FakeAdapter)
    cfg = config.resolve({"export": str(export_dir), "model": {"family": "fake"}, "epochs": 2, "batch": 2,
                          "device": "cpu", "workers": 0, "out": str(tmp_path / "runs"),
                          "mlflow": {"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}", "experiment": "test"}})
    result = loop.train(cfg, run_name="unit")
    run = mlflow.get_run(result["run_id"])
    assert run.data.params["model.family"] == "fake" and run.data.params["imgsz"] == str(cfg["imgsz"])
    assert run.data.params["export.git_commit"] == "abc"
    assert run.data.metrics["val_map"] == pytest.approx(1.0)
    assert "train_loss" in run.data.metrics and "test_map" in run.data.metrics  # test split scored once, at the end
    assert (pathlib.Path(result["out"]) / "best" / "weights.txt").exists()
    assert result["best_epoch"] in (0, 1)
