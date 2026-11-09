# Hinge strip, option B (Madelief 2026-11-10 ~01:05, "union of visible knuckles"): per door the 2-3 px knuckle boxes are hard to
# see (on half the doors the detector puts nothing at all), so a training + scoring copy of an export replaces the knuckle boxes of
# each door with ONE box: the union of that door's visible knuckle boxes, widened to a minimum width (default 8 px, same centre,
# kept inside the image; widen_boxes.widen_bbox). Only what is visible, like the other labels; class stays "hinge".
#   strip_hinges.strip_coco(coco, min_w=8) -> new coco dict (input untouched):
#     - hinge annotations are grouped by (image_id, attributes.door, iscrowd); each group becomes one annotation with
#       bbox = union of the group's boxes (then widened), area = w * h, iscrowd = the group's, category = hinge,
#       object = "hinge_strip_<door>", attributes = {"door": <door>, "knuckles": <number of boxes merged>};
#     - hinge annotations without attributes.door stay single (only widened), so nothing is lost;
#     - all other annotations unchanged (same dicts, same order before the strips);
#     - annotation ids renumbered 1..N (unique); no "hinge" category -> ValueError.
#   python train/strip_hinges.py SRC OUT [--min-width 8]: like widen_boxes: OUT symlinks every entry of SRC except coco/ and
#     export.json; coco/{train,val,test}.json stripped; export.json = SRC's + "strip": {"class": "hinge", "min_width", "from"}.
#     OUT that exists -> error. (Training reads COCO; the symlinked YOLO labels/ keep the knuckles.)
# train/.venv/bin/python -m pytest train/tests/test_strip_hinges.py
import copy, json, pathlib, subprocess, sys
import pytest

TRAIN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))
import strip_hinges as sh

CATS = [{"id": 1, "name": "door"}, {"id": 2, "name": "handle"}, {"id": 3, "name": "hinge"}, {"id": 4, "name": "window"}]


def a(i, img, cat, box, obj, crowd=0, **attrs):
    return {"id": i, "image_id": img, "category_id": cat, "bbox": box, "area": box[2] * box[3], "iscrowd": crowd, "object": obj,
            "attributes": attrs}


def coco():
    return {"images": [{"id": 0, "width": 640, "height": 480}, {"id": 1, "width": 640, "height": 480}],
            "categories": CATS,
            "annotations": [
                a(1, 0, 1, [380, 0, 60, 480], "door_a"),
                a(2, 0, 3, [428, 430, 5, 32], "hinge_a_0", door="door_a"),
                a(3, 0, 3, [410, 194, 4, 23], "hinge_a_1", door="door_a"),
                a(4, 0, 3, [397, 26, 3, 19], "hinge_a_2", door="door_a"),
                a(5, 0, 2, [300, 240, 6, 4], "handle_a", door="door_a"),
                a(6, 0, 3, [20, 100, 2, 6], "hinge_b_0", door="door_b"),          # door b: one visible knuckle
                a(7, 0, 3, [30, 300, 2, 6], "hinge_b_1", crowd=1, door="door_b"),  # + one ignored knuckle
                a(8, 1, 3, [637, 50, 2, 10], "hinge_a_0", door="door_a"),         # other image, at the right edge
                a(9, 1, 3, [100, 60, 2, 8], "loose_hinge"),                        # no door attribute
            ]}


def hinges(c):
    return [x for x in c["annotations"] if x["category_id"] == 3]


def test_union_per_door_widened():
    out = sh.strip_coco(coco())
    s = next(x for x in hinges(out) if x["image_id"] == 0 and x["attributes"].get("door") == "door_a")
    # union x 397..433 (w 36, wide enough), y 26..462 (h 436)
    assert s["bbox"] == pytest.approx([397, 26, 36, 436]) and s["area"] == pytest.approx(36 * 436)
    assert s["object"] == "hinge_strip_door_a" and s["attributes"] == {"door": "door_a", "knuckles": 3} and s["iscrowd"] == 0
    assert len([x for x in hinges(out) if x["image_id"] == 0 and x["attributes"].get("door") == "door_a"]) == 1


def test_single_knuckle_and_crowd_kept_apart():
    out = sh.strip_coco(coco())
    b = [x for x in hinges(out) if x["attributes"].get("door") == "door_b"]
    live, crowd = [x for x in b if not x["iscrowd"]], [x for x in b if x["iscrowd"]]
    assert len(live) == 1 and len(crowd) == 1
    assert live[0]["bbox"] == pytest.approx([17, 100, 8, 6]) and live[0]["attributes"]["knuckles"] == 1   # widened around 21
    assert crowd[0]["bbox"] == pytest.approx([27, 300, 8, 6])


def test_edge_and_loose_hinges():
    out = sh.strip_coco(coco())
    img1 = [x for x in hinges(out) if x["image_id"] == 1]
    edge = next(x for x in img1 if x["attributes"].get("door") == "door_a")
    assert edge["bbox"] == pytest.approx([632, 50, 8, 10])                  # shifted inside the 640 px image
    loose = [x for x in img1 if x.get("object") == "loose_hinge"]
    assert len(loose) == 1 and loose[0]["bbox"] == pytest.approx([97, 60, 8, 8])


def test_others_unchanged_ids_unique_input_untouched():
    src = coco(); keep = copy.deepcopy(src)
    out = sh.strip_coco(src)
    assert src == keep
    others = [x for x in out["annotations"] if x["category_id"] != 3]
    want = [x for x in keep["annotations"] if x["category_id"] != 3]
    assert [{k: v for k, v in x.items() if k != "id"} for x in others] == [{k: v for k, v in x.items() if k != "id"} for x in want]
    ids = [x["id"] for x in out["annotations"]]
    assert sorted(ids) == list(range(1, len(ids) + 1))
    assert len(hinges(out)) == 5                                           # a strip, b live, b crowd, img1 a, loose
    assert out["images"] == keep["images"] and out["categories"] == keep["categories"]


def test_min_width_option_and_missing_class():
    out = sh.strip_coco(coco(), min_w=40)
    s = next(x for x in hinges(out) if x["image_id"] == 0 and x["attributes"].get("door") == "door_a")
    assert s["bbox"] == pytest.approx([395, 26, 40, 436])                   # centre 415 kept
    c = coco(); c["categories"] = [x for x in CATS if x["name"] != "hinge"]
    with pytest.raises(ValueError):
        sh.strip_coco(c)


def test_cli(tmp_path):
    src = tmp_path / "src"; (src / "coco").mkdir(parents=True); (src / "images").mkdir()
    for sp in ("train", "val", "test"):
        (src / "coco" / f"{sp}.json").write_text(json.dumps(coco()))
    (src / "export.json").write_text(json.dumps({"source": "x"}))
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(TRAIN / "strip_hinges.py"), str(src), str(out)], check=True)
    assert (out / "images").is_symlink() and not (out / "coco").is_symlink()
    assert len(hinges(json.loads((out / "coco" / "test.json").read_text()))) == 5
    exp = json.loads((out / "export.json").read_text())
    assert exp["source"] == "x" and exp["strip"] == {"class": "hinge", "min_width": 8, "from": str(src.resolve())}
    r = subprocess.run([sys.executable, str(TRAIN / "strip_hinges.py"), str(src), str(out)], capture_output=True)
    assert r.returncode != 0
