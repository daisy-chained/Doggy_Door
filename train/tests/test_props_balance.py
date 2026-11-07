# Balanced sampling for the property heads (Claude, 2026-11-08, research-based: Doggy Door exception, logged in the project note).
# Why: the state head gets closed glass doors right ~1 in 4 (v5h test 0.24, val 0.27); the train split has 36 closed glass doors vs 180
# open glass, 502 closed / 1118 open non-glass. Standard fix for a rare group: oversample it (WeightedRandomSampler).
#   props.DEFAULTS["balance"] = None (off: plain shuffle, runs as before). A list of attribute names, e.g. ["glass", "state"], turns it on.
#   CropDataset keeps the annotation attributes per item: ds.attrs[i] (dict; {} when the annotation has none).
#   props.balance_weights(attrs, keys) -> float numpy array, one weight per item, summing to 1: every group (tuple of attrs.get(k) for k in
#     keys; a missing key is None) gets the same total weight, split evenly over its items.
#   train(): with balance set, the train loader draws len(train) items per epoch with replacement by those weights (seeded); val / test
#     loaders are unchanged. The run logs param "balance".
import pathlib, sys
import numpy as np
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from doordet import props
from test_props import make_export


def test_default_off():
    assert props.DEFAULTS["balance"] is None
    assert props.resolve({"export": "x", "crop": "door"})["balance"] is None
    assert props.resolve({"export": "x", "crop": "door", "balance": ["glass", "state"]})["balance"] == ["glass", "state"]


def test_balance_weights_equal_per_group():
    attrs = [{"glass": False, "state": "open"}] * 6 + [{"glass": True, "state": "closed"}] + [{"glass": False, "state": "closed"}] * 3 + [{}]
    w = props.balance_weights(attrs, ["glass", "state"])
    assert isinstance(w, np.ndarray) and w.shape == (len(attrs),) and w.sum() == pytest.approx(1.0)
    groups = {}
    for a, x in zip(attrs, w):
        groups.setdefault((a.get("glass"), a.get("state")), []).append(x)
    assert len(groups) == 4
    for xs in groups.values():
        assert sum(xs) == pytest.approx(0.25) and max(xs) == pytest.approx(min(xs))


def test_dataset_keeps_attrs(tmp_path):
    ds = props.CropDataset(make_export(tmp_path / "e"), "train", "door", 64, 0.1, 4)
    assert len(ds.attrs) == len(ds.items) and all(isinstance(a, dict) for a in ds.attrs)


def test_train_with_balance(tmp_path):
    import mlflow
    export_dir = make_export(tmp_path / "e")
    cfg = props.resolve({"export": str(export_dir), "crop": "door", "backbone": "resnet18", "pretrained": False, "imgsz": 64, "epochs": 1,
                         "batch": 4, "device": "cpu", "workers": 0, "out": str(tmp_path / "runs"), "balance": ["state"],
                         "mlflow": {"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}", "experiment": "test"}})
    r = props.train(cfg, run_name="unit")
    assert "state" in mlflow.get_run(r["run_id"]).data.params["balance"]
