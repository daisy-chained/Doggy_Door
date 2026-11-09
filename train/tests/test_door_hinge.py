# Per-door hinge score (Madelief 2026-11-09, option A: "is there a hinge on this door, yes / no" instead of counting knuckles).
# evaluate.door_hinge_metrics(gt_json, preds, score_min=0.5, margin=0.1) -> dict:
#   positive door = a door annotation (class "door") in the image with >= 1 visible, non-crowd hinge annotation whose
#                   attributes.door == the door's "object" name;
#   negative door = a door in the image without any such hinge (push side, sliding door, hinges out of frame);
#   doors whose only hinges are crowd (ignored) are left out of both.
#   found      = a hinge prediction with score >= score_min point-hits one of that door's hinges (same rule as point_metrics:
#                centre within max(MIN_RADIUS, half the GT's longer side));
#   false yes  = on a negative door, a hinge prediction >= score_min whose centre lies inside the door box grown by
#                margin * box width left and right, and that does not point-hit any hinge GT of the image.
# keys: door_hinge_recall, door_hinge_false_yes, door_hinge_n_pos, door_hinge_n_neg,
#       door_hinge_recall_<state> for every door attributes.state seen on positives (open / closed / ...).
import json, sys, pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from doordet import evaluate

CATS = [{"id": 1, "name": "door"}, {"id": 2, "name": "handle"}, {"id": 3, "name": "hinge"}, {"id": 4, "name": "window"}]


def ann(i, img, cat, obj, box, crowd=0, **attrs):
    return {"id": i, "image_id": img, "category_id": cat, "object": obj, "bbox": box, "area": box[2] * box[3], "iscrowd": crowd,
            "attributes": attrs}


def gt_file(tmp_path, anns, n_images=1):
    p = tmp_path / "gt.json"
    p.write_text(json.dumps({"images": [{"id": i, "width": 640, "height": 480} for i in range(n_images)],
                             "categories": CATS, "annotations": anns}))
    return str(p)


def hinge_pred(*centres_scores):
    """preds for image 0: hinge boxes (label = category_id - 1 = 2) 4 x 20 px around the given centres."""
    boxes = [[x - 2, y - 10, x + 2, y + 10] for (x, y), _ in centres_scores]
    return {0: {"boxes": boxes, "scores": [s for _, s in centres_scores], "labels": [2] * len(boxes)}}


def door_with_hinges(n_hinges=3, state="closed", crowd=0, start_id=1, door="door_a", x=100):
    anns = [ann(start_id, 0, 1, door, [x, 0, 200, 480], state=state)]
    for k in range(n_hinges):
        anns.append(ann(start_id + 1 + k, 0, 3, f"hinge_{door[-1]}_{k}", [x + 190, 50 + 150 * k, 4, 20], crowd=crowd, door=door))
    return anns


def test_one_knuckle_found_is_enough(tmp_path):
    gt = gt_file(tmp_path, door_with_hinges(3))
    m = evaluate.door_hinge_metrics(gt, hinge_pred(((292, 60), 0.9)))      # only the top knuckle
    assert m["door_hinge_n_pos"] == 1 and m["door_hinge_recall"] == 1.0


def test_no_hit_or_low_score_is_a_miss(tmp_path):
    gt = gt_file(tmp_path, door_with_hinges(3))
    assert evaluate.door_hinge_metrics(gt, hinge_pred(((292, 60), 0.3)))["door_hinge_recall"] == 0.0      # below score_min
    assert evaluate.door_hinge_metrics(gt, hinge_pred(((150, 60), 0.9)))["door_hinge_recall"] == 0.0      # wrong place
    assert evaluate.door_hinge_metrics(gt, hinge_pred(((292, 60), 0.3)), score_min=0.2)["door_hinge_recall"] == 1.0


def test_door_without_visible_hinges_is_negative(tmp_path):
    gt = gt_file(tmp_path, door_with_hinges(0, state="closed"))
    m = evaluate.door_hinge_metrics(gt, hinge_pred())
    assert m["door_hinge_n_pos"] == 0 and m["door_hinge_n_neg"] == 1 and m["door_hinge_false_yes"] == 0.0
    m = evaluate.door_hinge_metrics(gt, hinge_pred(((150, 200), 0.9)))     # hinge "seen" on a push-side door
    assert m["door_hinge_false_yes"] == 1.0


def test_false_yes_uses_margin_and_ignores_far_predictions(tmp_path):
    gt = gt_file(tmp_path, door_with_hinges(0))                            # door x 100..300, margin 0.1 -> 80..320
    assert evaluate.door_hinge_metrics(gt, hinge_pred(((310, 200), 0.9)))["door_hinge_false_yes"] == 1.0
    assert evaluate.door_hinge_metrics(gt, hinge_pred(((400, 200), 0.9)))["door_hinge_false_yes"] == 0.0
    assert evaluate.door_hinge_metrics(gt, hinge_pred(((310, 200), 0.9)), margin=0.0)["door_hinge_false_yes"] == 0.0


def test_real_hinge_of_neighbour_door_is_not_a_false_yes(tmp_path):
    # door a (positive) x 100..300 with hinges at x ~292; door b (negative) x 280..480 overlaps that edge
    anns = door_with_hinges(3, door="door_a", x=100) + [ann(10, 0, 1, "door_b", [280, 0, 200, 480], state="closed")]
    m = evaluate.door_hinge_metrics(gt_file(tmp_path, anns), hinge_pred(((292, 60), 0.9)))
    assert m["door_hinge_recall"] == 1.0 and m["door_hinge_n_neg"] == 1 and m["door_hinge_false_yes"] == 0.0


def test_crowd_only_doors_are_left_out(tmp_path):
    m = evaluate.door_hinge_metrics(gt_file(tmp_path, door_with_hinges(3, crowd=1)), hinge_pred())
    assert m["door_hinge_n_pos"] == 0 and m["door_hinge_n_neg"] == 0


def test_recall_per_state_and_string_image_ids(tmp_path):
    anns = door_with_hinges(3, state="closed", door="door_a", x=0) + door_with_hinges(3, state="open", start_id=20, door="door_b", x=300)
    preds = hinge_pred(((192, 60), 0.9))                                    # finds door a only
    preds = {"0": preds[0]}                                                 # predictions keyed by str id also work
    m = evaluate.door_hinge_metrics(gt_file(tmp_path, anns), preds)
    assert m["door_hinge_n_pos"] == 2 and m["door_hinge_recall"] == 0.5
    assert m["door_hinge_recall_closed"] == 1.0 and m["door_hinge_recall_open"] == 0.0


def test_empty_counts_give_zero_not_errors(tmp_path):
    m = evaluate.door_hinge_metrics(gt_file(tmp_path, []), {})
    assert m["door_hinge_n_pos"] == 0 and m["door_hinge_recall"] == 0.0 and m["door_hinge_false_yes"] == 0.0
