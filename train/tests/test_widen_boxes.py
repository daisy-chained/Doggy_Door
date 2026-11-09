# Wider hinge boxes (Madelief 2026-11-09, "bigger + wider hinges"): hinge boxes are ~2-3 px wide, narrower than one cell of the
# detector's finest grid (8 px), so they are hard to learn. A training copy of an export gets hinge boxes widened to a minimum width;
# scoring stays on the ORIGINAL export (point score: centre distance, so a wider box is not punished).
#   widen_boxes.widen_bbox(bbox xywh, min_w, img_w) -> xywh: width < min_w -> min_w, same centre, shifted to stay inside
#     [0, img_w] (min_w > img_w -> x 0, width img_w); height / y unchanged; a box already >= min_w is returned unchanged.
#   widen_boxes.widen_coco(coco, name, min_w) -> new coco dict (input untouched): every annotation of category `name` (ignore
#     boxes too, so ignore regions keep covering the part) widened, "area" = w * h; other classes and fields unchanged. A name
#     that is not a category -> ValueError.
#   python train/widen_boxes.py SRC OUT [--class hinge] [--min-width 8]: OUT gets a symlink to every entry of SRC except
#     coco/ and export.json; coco/{train,val,test}.json widened; export.json = SRC's plus "widen": {"class", "min_width",
#     "from": SRC absolute}. OUT that exists -> error (no overwrite).
# train/.venv/bin/python -m pytest train/tests/test_widen_boxes.py
import copy, json, pathlib, subprocess, sys
import pytest

TRAIN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))
import widen_boxes as wb


def test_widen_bbox():
    assert wb.widen_bbox([99, 43, 2, 14], 8, 640) == pytest.approx([96, 43, 8, 14])   # centre 100 kept
    assert wb.widen_bbox([10, 5, 20, 30], 8, 640) == [10, 5, 20, 30]                  # wide enough: unchanged
    assert wb.widen_bbox([0.5, 5, 2, 14], 8, 640) == pytest.approx([0, 5, 8, 14])     # left edge: shifted in
    assert wb.widen_bbox([638, 5, 2, 14], 8, 640) == pytest.approx([632, 5, 8, 14])   # right edge: shifted in
    assert wb.widen_bbox([3, 5, 2, 14], 8, 6) == pytest.approx([0, 5, 6, 14])         # wider than the image


def coco():
    return {"images": [{"id": 0, "width": 640, "height": 480, "file_name": "images/train/000000.png"}],
            "categories": [{"id": 1, "name": "door"}, {"id": 2, "name": "handle"}, {"id": 3, "name": "hinge"}],
            "annotations": [
                {"id": 1, "image_id": 0, "category_id": 3, "bbox": [99, 43, 2, 14], "area": 28, "iscrowd": 0, "object": "hinge_a"},
                {"id": 2, "image_id": 0, "category_id": 3, "bbox": [300, 40, 1, 10], "area": 10, "iscrowd": 1},
                {"id": 3, "image_id": 0, "category_id": 2, "bbox": [200, 200, 4, 3], "area": 12, "iscrowd": 0},
                {"id": 4, "image_id": 0, "category_id": 1, "bbox": [100, 40, 3, 300], "area": 900, "iscrowd": 0}]}


def test_widen_coco_only_the_class():
    src = coco(); keep = copy.deepcopy(src)
    out = wb.widen_coco(src, "hinge", 8)
    assert src == keep
    a = {x["id"]: x for x in out["annotations"]}
    assert a[1]["bbox"] == pytest.approx([96, 43, 8, 14]) and a[1]["area"] == pytest.approx(112) and a[1]["object"] == "hinge_a"
    assert a[2]["bbox"] == pytest.approx([296.5, 40, 8, 10]) and a[2]["iscrowd"] == 1   # ignore box widened too
    assert a[3] == keep["annotations"][2] and a[4] == keep["annotations"][3]           # small handle, thin door untouched
    assert out["images"] == keep["images"] and out["categories"] == keep["categories"]
    with pytest.raises(ValueError):
        wb.widen_coco(src, "window", 8)


def test_cli_makes_a_linked_copy(tmp_path):
    src = tmp_path / "exp"
    (src / "coco").mkdir(parents=True); (src / "images" / "train").mkdir(parents=True)
    (src / "images" / "train" / "000000.png").write_bytes(b"png")
    (src / "data.yaml").write_text("x: 1\n")
    (src / "export.json").write_text(json.dumps({"source": {"seed": 6}}))
    for sp in ("train", "val", "test"):
        (src / "coco" / f"{sp}.json").write_text(json.dumps(coco()))
    out = tmp_path / "exp_wide"
    run = lambda: subprocess.run([sys.executable, str(TRAIN / "widen_boxes.py"), str(src), str(out), "--min-width", "8"],
                                 capture_output=True, text=True)
    r = run()
    assert r.returncode == 0, r.stderr
    assert (out / "images").is_symlink() and (out / "data.yaml").is_symlink() and not (out / "coco").is_symlink()
    assert (out / "images" / "train" / "000000.png").read_bytes() == b"png"
    for sp in ("train", "val", "test"):
        g = json.loads((out / "coco" / f"{sp}.json").read_text())
        assert next(x for x in g["annotations"] if x["id"] == 1)["bbox"] == pytest.approx([96, 43, 8, 14])
    e = json.loads((out / "export.json").read_text())
    assert e["source"] == {"seed": 6} and e["widen"] == {"class": "hinge", "min_width": 8, "from": str(src.resolve())}
    assert json.loads((src / "coco" / "train.json").read_text()) == coco()   # source untouched
    assert run().returncode != 0   # no overwrite
