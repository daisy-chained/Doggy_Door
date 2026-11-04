# Unit tests for late fusion (Madelief 2026-11-04: RGB detector + depth detector, boxes merged): weighted box fusion
# (Solovyev et al. 2021, arXiv 1910.13302) on two runs' predictions: train/.venv/bin/python -m pytest train/tests
import pathlib, sys
import numpy as np
import pytest

TRAIN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))
from doordet import fusion


def p(boxes, scores, labels):
    return {"boxes": np.array(boxes, float).reshape(-1, 4), "scores": np.array(scores, float), "labels": np.array(labels, int)}


def test_same_box_from_both_models_becomes_one():
    out = fusion.wbf([p([[0, 0, 10, 10]], [0.8], [0]), p([[0, 0, 10, 10]], [0.6], [0])])
    assert len(out["scores"]) == 1
    assert out["boxes"][0].tolist() == pytest.approx([0, 0, 10, 10])
    assert out["scores"][0] == pytest.approx(0.7)


def test_box_from_one_model_only_is_scored_down():
    out = fusion.wbf([p([[0, 0, 10, 10]], [0.8], [0]), p([], [], [])])
    assert out["scores"][0] == pytest.approx(0.4)  # WBF: score x (models that found it) / (models)


def test_coordinates_are_score_weighted():
    out = fusion.wbf([p([[0, 0, 10, 10]], [0.9], [0]), p([[2, 0, 12, 10]], [0.3], [0])], iou=0.5)
    assert out["boxes"][0][0] == pytest.approx((0 * 0.9 + 2 * 0.3) / 1.2)


def test_labels_never_merge_and_far_boxes_stay_apart():
    out = fusion.wbf([p([[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.9], [0, 0]), p([[0, 0, 10, 10]], [0.9], [1])])
    assert sorted(out["labels"].tolist()) == [0, 0, 1]


def test_model_weights():
    out = fusion.wbf([p([[0, 0, 10, 10]], [0.8], [0]), p([[0, 0, 10, 10]], [0.2], [0])], weights=[3, 1])
    assert out["scores"][0] == pytest.approx((3 * 0.8 + 1 * 0.2) / 4)


def test_one_model_box_in_cluster_twice_counts_once():
    # two overlapping boxes from model A alone: still found by 1 of 2 models
    out = fusion.wbf([p([[0, 0, 10, 10], [0, 0, 10, 11]], [0.8, 0.8], [0, 0]), p([], [], [])], iou=0.5)
    assert len(out["scores"]) == 1 and out["scores"][0] == pytest.approx(0.4)


def test_fuse_runs_over_all_images():
    a = {0: p([[0, 0, 10, 10]], [0.8], [0]), 1: p([], [], [])}
    b = {0: p([[0, 0, 10, 10]], [0.6], [0]), 2: p([[5, 5, 9, 9]], [0.5], [1])}
    out = fusion.fuse_runs([a, b])
    assert set(out) == {0, 1, 2}
    assert out[0]["scores"][0] == pytest.approx(0.7) and out[2]["scores"][0] == pytest.approx(0.25)
    assert len(out[1]["scores"]) == 0


def test_low_score_duplicates_do_not_drag_the_score_down():
    # DETR-style output: one confident query per object plus low-score duplicates around it
    out = fusion.wbf([p([[0, 0, 10, 10], [0, 0, 10, 10.5], [0, 0, 10.5, 10]], [0.9, 0.02, 0.02], [0, 0, 0]), p([[0, 0, 10, 10]], [0.9], [0])])
    assert len(out["scores"]) == 1 and out["scores"][0] == pytest.approx(0.9)
