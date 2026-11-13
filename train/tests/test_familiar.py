# "This door looks unfamiliar" check for the door property model (Madelief 2026-11-13: confidence on new door types, option A of
# A + B + C). The property heads are ~81 % sure but ~43 % right on held-out door types; recalibration and a 5-model ensemble did not
# fix that. Technical choices = Claude decisions (research-based, project note):
#   - score = relative Mahalanobis distance on the backbone features (Ren et al. 2021, "A simple fix to Mahalanobis distance for
#     improving near-OOD detection"; Lee et al. 2018 for the class-conditional Gaussians with one shared covariance). New door types
#     are new COMBINATIONS of known properties (near-OOD), the case the relative version was made for.
#   - the Gaussian "classes" are the door types seen in training = the tuple of the run's head labels (e.g. push / right / glass).
#   - threshold = the `keep` quantile (0.95) of the val scores: val has only familiar doors, so ~5 % of familiar doors get flagged.
#   - flagged doors count as "not sure": a >= 0.9 answer on a flagged door is no longer "sure".
#
# Module train/doordet/familiar.py (numpy + torch):
#   group_ids(labels, heads) -> (np int array N, list of combos): labels = {head: int array N}; combo = tuple of the head labels in
#     `heads` order; ids number the combos in first-seen order; combos = list of tuples (as lists of ints when saved).
#   fit(feats, groups, reg=1e-3) -> stats dict of float64 numpy arrays: "means" K x D (one per group id 0..K-1), "mu0" D,
#     "prec" D x D = inverse of the shared within-group covariance, "prec0" D x D = inverse of the covariance of all features;
#     each covariance gets reg * mean(diag) added on the diagonal before inverting.
#   score(stats, feats, relative=True) -> np array N: md_k(x) = (x - means[k])^T prec (x - means[k]); md0(x) = (x - mu0)^T prec0
#     (x - mu0); relative=True -> min_k md_k - md0, relative=False -> min_k md_k. Higher = more unfamiliar.
#   threshold(val_scores, keep=0.95) -> float: the `keep` quantile of val_scores (np.quantile).
#   auroc(scores, positive) -> float: P(score of a positive > score of a negative), ties count half; nan if a class is empty.
#   features(model, loader, device) -> (feats N x D float tensor on cpu, {head: label tensor N}): model.backbone(x) (props.Model).
#   build(run_dir, out_dir, device="cpu", keep=0.95, reg=1e-3, max_train=None, batch=64) -> dict, also written as
#     out_dir/familiar.json: {"run": str abs run_dir, "method": "rmd", "keep", "reg", "threshold", "combos": [[...], ...],
#     "test": {...}} and the stats as out_dir/familiar.npz. Single props run only (an ensemble run -> ValueError). Train / val / test
#     = props.CropDataset of the run's config.json export, crop, imgsz, margin, min_px, input, need = the run's heads, no jitter;
#     max_train: use at most that many train items (seeded subset, seed 0).
#     test keys: test_auroc (positive = held-out item), test_flag_seen, test_flag_unseen (share flagged), and per head h, for
#     part in (seen, unseen): test_sure_share_<part>_<h>, test_sure_acc_<part>_<h> (calibrated conf >= 0.9 with the run's
#     temperature; acc nan when none) and test_gated_sure_share_<part>_<h>, test_gated_sure_acc_<part>_<h> (the same, but flagged
#     items are never sure).
#   load(out_dir) -> Checker with .threshold, .score(feats) (relative, np array) and .flag(feats) (bool array, score > threshold).
# CLI: train/.venv/bin/python train/build_familiar.py RUN_DIR OUT_DIR [--device cuda] [--keep 0.95] [--max-train N]
import json, math, pathlib, sys
import numpy as np
import pytest
import torch

TRAIN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))
from doordet import props, familiar
from test_props import make_export


def blobs(rng, centres, n=200, d=8, scale=0.3):
    xs, gs = [], []
    for k, c in enumerate(centres):
        xs.append(rng.normal(0, scale, (n, d)) + np.asarray(c, float)); gs.append(np.full(n, k))
    return np.concatenate(xs), np.concatenate(gs)


def test_group_ids_numbers_combos():
    labels = {"opening": np.array([0, 2, 0, 1]), "glass": np.array([1, 0, 1, 1])}
    ids, combos = familiar.group_ids(labels, ["opening", "glass"])
    assert list(ids) == [0, 1, 0, 2]
    assert [tuple(c) for c in combos] == [(0, 1), (2, 0), (1, 1)]


def test_fit_shapes_and_means():
    rng = np.random.default_rng(0)
    c = [np.eye(8)[0] * 3, np.eye(8)[1] * 3, np.eye(8)[2] * 3]
    x, g = blobs(rng, c)
    st = familiar.fit(x, g)
    assert st["means"].shape == (3, 8) and st["mu0"].shape == (8,) and st["prec"].shape == st["prec0"].shape == (8, 8)
    assert np.allclose(st["means"], c, atol=0.1) and np.allclose(st["mu0"], np.mean(c, 0), atol=0.1)


def test_score_formula():
    rng = np.random.default_rng(1)
    x, g = blobs(rng, [np.zeros(8), np.ones(8) * 2])
    st = familiar.fit(x, g)
    q = rng.normal(0, 1, (5, 8))
    md = np.stack([np.einsum("nd,de,ne->n", q - m, st["prec"], q - m) for m in st["means"]]).min(0)
    md0 = np.einsum("nd,de,ne->n", q - st["mu0"], st["prec0"], q - st["mu0"])
    assert np.allclose(familiar.score(st, q, relative=False), md)
    assert np.allclose(familiar.score(st, q), md - md0)


def test_new_combination_scores_high():
    # three known types on the corners of a square; the fourth corner = a new combination of known "properties"
    rng = np.random.default_rng(2)
    x, g = blobs(rng, [[0, 0] + [0] * 6, [3, 0] + [0] * 6, [0, 3] + [0] * 6])
    st = familiar.fit(x, g)
    known = familiar.score(st, blobs(rng, [[0, 0] + [0] * 6], n=100)[0])
    new = familiar.score(st, blobs(rng, [[3, 3] + [0] * 6], n=100)[0])
    t = familiar.threshold(known, 0.95)
    assert (new > t).mean() > 0.9
    assert familiar.auroc(np.concatenate([known, new]), np.r_[np.zeros(100, bool), np.ones(100, bool)]) > 0.95


def test_threshold_is_quantile():
    s = np.arange(100.0)
    assert familiar.threshold(s, 0.95) == pytest.approx(np.quantile(s, 0.95))
    assert (s > familiar.threshold(s, 0.9)).mean() == pytest.approx(0.1, abs=0.02)


def test_auroc():
    pos = np.array([False, False, True, True])
    assert familiar.auroc(np.array([0.1, 0.2, 0.8, 0.9]), pos) == 1.0
    assert familiar.auroc(np.array([0.9, 0.8, 0.2, 0.1]), pos) == 0.0
    assert familiar.auroc(np.ones(4), pos) == 0.5
    assert math.isnan(familiar.auroc(np.ones(3), np.zeros(3, bool)))


@pytest.fixture
def run(tmp_path):
    root = make_export(tmp_path / "export")
    coco = json.loads((root / "coco" / "test.json").read_text())
    for a in coco["annotations"]:
        a["attributes"] = dict(a["attributes"], heldout=a["object"] in ("door_b", "handle_b"))
    (root / "coco" / "test.json").write_text(json.dumps(coco))
    cfg = props.resolve({"export": str(root), "crop": "door", "backbone": "resnet18", "pretrained": False, "imgsz": 64, "epochs": 1,
                         "batch": 4, "device": "cpu", "workers": 0, "seed": 0, "out": str(tmp_path / "runs"), "min_calibration": 1,
                         "mlflow": {"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}", "experiment": "test"}})
    return props.train(cfg, run_name="m0")["out"]


def test_features_are_backbone_output(run):
    model, meta = props.load(run)
    ds = props.CropDataset(json.loads((pathlib.Path(run) / "config.json").read_text())["export"], "val", "door", 64, 0.1, 16)
    loader = torch.utils.data.DataLoader(ds, batch_size=4)
    f, lab = familiar.features(model, loader, "cpu")
    assert f.shape == (len(ds), 512) and set(meta["heads"]) <= set(lab)
    with torch.no_grad():
        assert torch.allclose(f[:1], model.backbone(ds[0][0][None]), atol=1e-5)


def test_build_writes_and_scores(tmp_path, run):
    out = tmp_path / "fam"
    r = familiar.build(run, out)
    saved = json.loads((out / "familiar.json").read_text())
    assert (out / "familiar.npz").exists()
    assert saved["method"] == "rmd" and saved["keep"] == 0.95 and saved["run"] == str(pathlib.Path(run).resolve())
    assert math.isfinite(saved["threshold"]) and saved["threshold"] == r["threshold"] and len(saved["combos"]) >= 1
    t = saved["test"]
    for k in ("test_auroc", "test_flag_seen", "test_flag_unseen"):
        assert k in t
    assert 0.0 <= t["test_flag_seen"] <= 1.0 and 0.0 <= t["test_flag_unseen"] <= 1.0
    _, meta = props.load(run)
    for h in meta["heads"]:
        for part in ("seen", "unseen"):
            for k in ("sure_share", "sure_acc", "gated_sure_share", "gated_sure_acc"):
                assert f"test_{k}_{part}_{h}" in t
            assert t[f"test_gated_sure_share_{part}_{h}"] <= t[f"test_sure_share_{part}_{h}"] + 1e-9


def test_load_checker(tmp_path, run):
    out = tmp_path / "fam"
    familiar.build(run, out)
    ck = familiar.load(out)
    f = np.random.default_rng(0).normal(0, 1, (3, 512))
    s = ck.score(f)
    assert s.shape == (3,) and np.array_equal(ck.flag(f), s > ck.threshold)


def test_build_rejects_ensemble(tmp_path, run):
    from doordet import ensemble
    ens = tmp_path / "ens"
    ensemble.build([run, run], ens)
    with pytest.raises(ValueError):
        familiar.build(ens, tmp_path / "fam")
