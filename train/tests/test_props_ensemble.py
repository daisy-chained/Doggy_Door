# Deep ensemble for the property heads (Madelief 2026-11-12: "recalibrate first if that doesnt work then ensemble"; recalibration did
# not reach the targets on held-out door types, dev log 2026-11-12). Technical choices = Claude decisions (research-based, project note):
# 5 members trained alike with seeds 0..4 (Lakshminarayanan et al. 2017), members' softmax averaged, then ONE temperature per head fitted
# on val on the pooled probabilities ("pool, then calibrate", Rahaman & Thiery 2021).
#
# Module train/doordet/ensemble.py:
#   ensemble.pool(logits_list) -> tensor N x C: log of the mean of softmax(logits) over the members (log clamped at 1e-12), so
#     softmax(pool(...)) == the mean member probability and softmax(pool(...) / T) is the calibrated ensemble.
#   ensemble.Ensemble(models): nn.Module; forward(x) -> {head: pool([m(x)[head] for m in models])}. Drop-in for props.Model.
#   ensemble.build(member_dirs, out_dir, device="cpu") -> dict written as out_dir/ensemble.json:
#     {"members": [str abs paths], "temperature": {head: T}, "test": {...}}; members must share heads, backbone, imgsz, crop, margin,
#     input (else ValueError). Val / test sets: CropDataset of the first member's config.json export (no jitter), as props.train.
#     T per head = props.fit_temperature(pooled val logits, labels) (same min_calibration rule as train: < min_calibration val items or
#     none wrong -> 1.0). Also writes out_dir/best/heads.json (first member's meta with the ensemble temperature) and out_dir/config.json
#     (first member's config.json with "ensemble": [member paths]).
#     test keys per head h: test_ece_<h>, test_acc_<h>, test_sure_share_<h> (share with calibrated conf >= 0.9), test_sure_acc_<h>
#     (accuracy among those; nan when none), and test_ece_seen_<h> / test_ece_unseen_<h> when the test set has held-out items.
#   props.load(run_dir): when run_dir/ensemble.json exists -> (Ensemble of the members, the out_dir's heads.json meta) — so
#     props_server.py and score_on_boxes work on an ensemble unchanged.
# CLI: train/.venv/bin/python train/build_ensemble.py OUT MEMBER_DIR...
import json, math, pathlib, sys
import numpy as np
import pytest
import torch

TRAIN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))
from doordet import props, ensemble
from test_props import make_export


def test_pool_is_log_mean_probability():
    a = torch.tensor([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]]); b = torch.tensor([[-1.0, 3.0, 0.0], [5.0, 0.0, 0.0]])
    p = ensemble.pool([a, b])
    want = (torch.softmax(a, 1) + torch.softmax(b, 1)) / 2
    assert p.shape == (2, 3)
    assert torch.allclose(torch.softmax(p, 1), want, atol=1e-6)
    assert torch.allclose(p.exp().sum(1), torch.ones(2), atol=1e-6)


def test_pool_one_member_matches_softmax():
    a = torch.randn(5, 2)
    assert torch.allclose(torch.softmax(ensemble.pool([a]), 1), torch.softmax(a, 1), atol=1e-6)


def test_pool_disagreement_lowers_confidence():
    sure_a, sure_b = torch.tensor([[6.0, 0.0]]), torch.tensor([[0.0, 6.0]])
    agree = torch.softmax(ensemble.pool([sure_a, sure_a]), 1).max()
    disagree = torch.softmax(ensemble.pool([sure_a, sure_b]), 1).max()
    assert agree > 0.99 and disagree == pytest.approx(0.5, abs=1e-3)


def test_ensemble_module_pools_per_head():
    torch.manual_seed(0)
    heads = {"opening": ["push", "pull", "slide"], "glass": ["no", "yes"]}
    ms = [props.Model("resnet18", heads, pretrained=False).eval() for _ in range(3)]
    e = ensemble.Ensemble(ms).eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out, each = e(x), [m(x) for m in ms]
    assert set(out) == set(heads)
    for h in heads:
        assert torch.allclose(out[h], ensemble.pool([o[h] for o in each]), atol=1e-5)


def _member(tmp_path, root, seed, **kw):
    cfg = props.resolve({"export": str(root), "crop": "door", "backbone": "resnet18", "pretrained": False, "imgsz": 64, "epochs": 1,
                         "batch": 4, "device": "cpu", "workers": 0, "seed": seed, "out": str(tmp_path / "runs"), "min_calibration": 1,
                         "mlflow": {"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}", "experiment": "test"}, **kw})
    return props.train(cfg, run_name=f"m{seed}")["out"]


@pytest.fixture
def members(tmp_path):
    root = make_export(tmp_path / "export")
    coco = json.loads((root / "coco" / "test.json").read_text())
    for a in coco["annotations"]:
        a["attributes"] = dict(a["attributes"], heldout=a["object"] in ("door_b", "handle_b"))
    (root / "coco" / "test.json").write_text(json.dumps(coco))
    return [_member(tmp_path, root, s) for s in (0, 1)]


def test_build_writes_ensemble_and_scores(tmp_path, members):
    out = tmp_path / "runs" / "ens"
    r = ensemble.build(members, out)
    saved = json.loads((out / "ensemble.json").read_text())
    assert saved["members"] == [str(pathlib.Path(m).resolve()) for m in members]
    assert (out / "best" / "heads.json").exists() and (out / "config.json").exists()
    assert json.loads((out / "config.json").read_text())["ensemble"] == saved["members"]
    for h in props.DEFAULT_HEADS["door"]:
        assert saved["temperature"][h] > 0 and r["temperature"][h] == saved["temperature"][h]
        for k in ("test_ece_", "test_acc_", "test_sure_share_", "test_sure_acc_", "test_ece_seen_", "test_ece_unseen_"):
            assert k + h in saved["test"]
        assert 0.0 <= saved["test"][f"test_sure_share_{h}"] <= 1.0
    assert json.loads((out / "best" / "heads.json").read_text())["temperature"] == saved["temperature"]


def test_load_ensemble_is_drop_in(tmp_path, members):
    out = tmp_path / "runs" / "ens"
    ensemble.build(members, out)
    model, meta = props.load(out)
    assert isinstance(model, ensemble.Ensemble) and not model.training
    assert meta["imgsz"] == 64 and meta["input"] == "rgb" and "min_px" in meta
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        o = model(x)
        each = [props.load(m)[0](x) for m in members]
    for h in meta["heads"]:
        assert torch.allclose(torch.softmax(o[h], 1), torch.stack([torch.softmax(e[h], 1) for e in each]).mean(0), atol=1e-5)


def test_build_rejects_mismatched_members(tmp_path):
    root = make_export(tmp_path / "export")
    a = _member(tmp_path, root, 0)
    b = _member(tmp_path, root, 1, imgsz=96)
    with pytest.raises(ValueError):
        ensemble.build([a, b], tmp_path / "runs" / "bad")
