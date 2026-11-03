# Unit tests for ignoring nearly vanished doors in smoke (exporter --ignore-below): python3 -m pytest sim/tests
import json, pathlib, sys
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(SIM / "tests"))
from doorsim import export
from test_export import make_dataset

T = {"door_a": 0.02, "handle_a": 0.5, "door_b": None}  # door_b: no smoke_transmission (frame without smoke)


def smoky(root):
    for f in (root / "frames").iterdir():
        m = json.loads((f / "meta.json").read_text())
        for n, t in T.items():
            if t is not None:
                m["objects"][n]["smoke_transmission"] = t
        (f / "meta.json").write_text(json.dumps(m))
    return root


def anns(out, split):
    return json.loads((out / "coco" / f"{split}.json").read_text())["annotations"]


def test_no_ignore_by_default(tmp_path):
    src = smoky(make_dataset(tmp_path / "ds"))
    export.export(src, tmp_path / "e")
    for sp in export.SPLITS:
        assert all(a["iscrowd"] == 0 and not a["attributes"].get("ignored") for a in anns(tmp_path / "e", sp))


def test_ignore_below_marks_train_and_val_only(tmp_path):
    src = smoky(make_dataset(tmp_path / "ds"))
    info = export.export(src, tmp_path / "e", ignore_below=0.05)
    for sp in ("train", "val"):
        for a in anns(tmp_path / "e", sp):
            gone = a["object"] == "door_a"  # 0.02 < 0.05; handle_a 0.5 stays; door_b has no smoke value
            assert a["iscrowd"] == int(gone) and bool(a["attributes"].get("ignored")) == gone
    # test sets keep counting them (that is the visibility curve)
    assert all(a["iscrowd"] == 0 for a in anns(tmp_path / "e", "test"))
    assert info["ignore"] == {"below": 0.05, "splits": ["train", "val"]}
    assert json.loads((tmp_path / "e" / "export.json").read_text())["ignore"] == info["ignore"]


def test_ignored_boxes_not_in_yolo_labels(tmp_path):
    src = smoky(make_dataset(tmp_path / "ds"))
    export.export(src, tmp_path / "e", ignore_below=0.05)
    rows = [l for f in (tmp_path / "e" / "labels" / "train").iterdir() for l in f.read_text().splitlines()]
    plain = tmp_path / "p"
    export.export(src, plain)
    rows_plain = [l for f in (plain / "labels" / "train").iterdir() for l in f.read_text().splitlines()]
    n_train = len(list((plain / "labels" / "train").iterdir()))
    assert len(rows) == len(rows_plain) - n_train  # one door_a row less per image


def test_ignore_splits_setting(tmp_path):
    src = smoky(make_dataset(tmp_path / "ds"))
    export.export(src, tmp_path / "e", ignore_below=0.05, ignore_splits=("train",))
    assert any(a["iscrowd"] for a in anns(tmp_path / "e", "train"))
    assert not any(a["iscrowd"] for a in anns(tmp_path / "e", "val"))
