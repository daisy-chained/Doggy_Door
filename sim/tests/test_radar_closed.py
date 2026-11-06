# Learned radar sanity check (2026-11-06, Claude, research-based): the learned doorway detector also "found" 32 % (imaging) /
# 67 % (single-chip) of closed doors, because closed doors were ignore boxes (never punished). Fix: closed doors in view become
# negatives (no box: background), and the scores also report a strict precision where a found closed door counts as wrong.
# python3 -m pytest sim/tests/test_radar_closed.py
import json, pathlib, sys
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import radar_doorways as rd, radar_export as rx


def ann(i, state, in_view, image_id=0):
    ignored = state != "open" or not in_view
    return {"id": i, "image_id": image_id, "category_id": 1, "bbox": [0, 0, 10, 3], "area": 30, "iscrowd": int(ignored),
            "attributes": {"door": f"door_{i}", "state": state, "in_view": in_view, **({"ignored": True} if ignored else {})}}


def coco():
    return {"images": [{"id": 0}], "categories": [{"id": 1, "name": "doorway"}],
            "annotations": [ann(1, "open", True), ann(2, "closed", True), ann(3, "closed", False), ann(4, "open", False)]}


def test_closed_in_view_doors_become_negatives():
    out = rx.closed_as_negatives(coco())
    kept = {a["attributes"]["door"]: a for a in out["annotations"]}
    assert "door_2" not in kept  # closed + in view: no box at all = background, punished if detected
    assert kept["door_1"]["iscrowd"] == 0  # open + in view: target as before
    assert kept["door_3"]["iscrowd"] == 1 and kept["door_4"]["iscrowd"] == 1  # out of view / hidden: still ignore


def test_closed_as_negatives_leaves_the_input_alone():
    c = coco()
    rx.closed_as_negatives(c)
    assert len(c["annotations"]) == 4


def test_relabel_export_folder(tmp_path):
    src = tmp_path / "radar_x_v1"
    (src / "coco").mkdir(parents=True)
    (src / "images" / "test").mkdir(parents=True)
    (src / "images" / "test" / "a.png").write_bytes(b"png")
    for sp in ("train", "val", "test"):
        (src / "coco" / f"{sp}.json").write_text(json.dumps(coco()))
    (src / "export.json").write_text(json.dumps({"model": "imaging", "views": 1, "labels": {"open_only": True}}))
    out = rx.relabel_closed(src, tmp_path / "radar_x_v1_neg")
    for sp in ("train", "val", "test"):
        assert len(json.loads((out / "coco" / f"{sp}.json").read_text())["annotations"]) == 3
    assert (out / "images" / "test" / "a.png").read_bytes() == b"png"  # images shared, not re-rendered
    info = json.loads((out / "export.json").read_text())
    assert info["model"] == "imaging" and info["labels"]["closed"] == "negative" and info["relabelled_from"] == str(src)
    with pytest.raises(FileExistsError):
        rx.relabel_closed(src, out)


def test_strict_precision_counts_found_closed_doors_as_wrong():
    rows = [{"door": "a", "found": True, "state": "open", "in_view": True},
            {"door": "b", "found": True, "state": "closed", "in_view": True},
            {"door": "c", "found": False, "state": "open", "in_view": True},
            {"door": None, "found": True, "false_positive": True}]
    m = rd.detection_metrics(rows)
    assert m["precision"] == pytest.approx(2 / 3)  # as before: a found closed door is no false positive
    assert m["precision_strict"] == pytest.approx(1 / 3)  # strict: only found OPEN doors are right
    assert m["closed_found"] == 1
    assert m["f1_strict"] == pytest.approx(2 * 0.5 * (1 / 3) / (0.5 + 1 / 3))
