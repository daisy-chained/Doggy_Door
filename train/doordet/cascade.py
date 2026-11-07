"""Door property cascade, step A (Madelief 2026-11-07): decide the door type first, then ask the hinged-only questions.
Type from the handle seen inside the door box (flush pull -> sliding, lever / knob -> hinged), else from a type classifier;
sliding -> opening "slide", hinge side "none"; hinged -> push vs pull from the opening head, left / right from the handle rule, else the hinge_side head."""
import json, pathlib
import numpy as np
import torch

from . import props
from .props import CATEGORY, CropDataset, crop_box, input_tensor, largest_handle_inside  # largest_handle_inside re-exported


def type_from_handle(handle_type):
    """Door type implied by a handle type: flush_pull -> "sliding", lever / knob -> "hinged", no handle -> None."""
    return None if handle_type is None else ("sliding" if handle_type == "flush_pull" else "hinged")


def hinges_near(door_xywh, hinge_xywhs, pad=0.1):
    """Indices of the hinge boxes whose centre lies in the door box widened by pad x its width per side (x) and inside its y range."""
    x, y, w, h = door_xywh
    out = []
    for i, (hx, hy, hw, hh) in enumerate(hinge_xywhs):
        cx, cy = hx + hw / 2, hy + hh / 2
        if x - pad * w <= cx <= x + w + pad * w and y <= cy <= y + h:
            out.append(i)
    return out


def decide(p_sliding, handle_type, p_push_pull, rule_hinge, p_left_right, hinge_xs=None, door_cx=None):
    """One door's cascade answer. p_push_pull = (P(push), P(pull)), p_left_right = (P(left), P(right)); rule_hinge: "left" / "right" / None.
    -> {"door_type", "type_source" (handle | classifier), "opening", "hinge_side", "hinge_source" (rule | classifier | None)}
    Step B: hinge_xs (centre x of the hinge boxes near this door, may be empty) + door_cx add "opening_source" and "hinges_seen":
    hinges seen on a hinged door -> "pull", hinge side from their mean x vs door_cx ("hinges"); none seen -> as step A; sliding -> None, None."""
    b = hinge_xs is not None
    dt = type_from_handle(handle_type)
    source = "handle" if dt else "classifier"
    dt = dt or ("sliding" if p_sliding >= 0.5 else "hinged")
    if dt == "sliding":
        res = {"door_type": dt, "type_source": source, "opening": "slide", "hinge_side": "none", "hinge_source": None}
        return {**res, "opening_source": None, "hinges_seen": None} if b else res
    opening, osrc = ("push" if p_push_pull[0] >= p_push_pull[1] else "pull"), "classifier"
    if b and len(hinge_xs):
        opening, osrc = "pull", "hinges"
        hinge, hsrc = ("left" if float(np.mean(hinge_xs)) < door_cx else "right"), "hinges"
    elif rule_hinge is not None:
        hinge, hsrc = rule_hinge, "rule"
    else:
        hinge, hsrc = ("left" if p_left_right[0] >= p_left_right[1] else "right"), "classifier"
    res = {"door_type": dt, "type_source": source, "opening": opening, "hinge_side": hinge, "hinge_source": hsrc}
    return {**res, "opening_source": osrc, "hinges_seen": bool(b and len(hinge_xs))} if b else res


def _mean(hits):
    return float(np.mean(hits)) if len(hits) else 0.0


def scores(decisions, truth):
    """Cascade accuracies from per-door decisions and true {"opening", "hinge_side"} (empty subsets give 0.0).
    True door type = sliding when the true opening is slide. push_pull / left_right: over true hinged doors the cascade also called hinged."""
    n = len(decisions)
    tt = ["sliding" if t["opening"] == "slide" else "hinged" for t in truth]
    by = lambda src: [i for i, d in enumerate(decisions) if d["type_source"] == src]
    ok_type = [d["door_type"] == t for d, t in zip(decisions, tt)]
    both = [i for i in range(n) if tt[i] == "hinged" and decisions[i]["door_type"] == "hinged"]
    res = {"doors": n,
            "acc_door_type": _mean(ok_type),
            "acc_opening": _mean([d["opening"] == t["opening"] for d, t in zip(decisions, truth)]),
            "acc_hinge_side": _mean([d["hinge_side"] == t["hinge_side"] for d, t in zip(decisions, truth)]),
            "type_from_handle": len(by("handle")) / n if n else 0.0,
            "acc_door_type_handle": _mean([ok_type[i] for i in by("handle")]),
            "acc_door_type_classifier": _mean([ok_type[i] for i in by("classifier")]),
            "acc_push_pull_hinged": _mean([decisions[i]["opening"] == truth[i]["opening"] for i in both]),
            "acc_left_right_hinged": _mean([decisions[i]["hinge_side"] == truth[i]["hinge_side"] for i in both])}
    if any("hinges_seen" in d for d in decisions):   # step B: hinge boxes were used
        seen = [i for i in both if decisions[i].get("hinges_seen")]
        res["hinges_seen"] = len(seen) / len(both) if both else 0.0
        res["acc_pull_when_hinges_seen"] = _mean([decisions[i]["opening"] == truth[i]["opening"] for i in seen])
        res["acc_push_pull_hinged_strict"] = _mean([("pull" if decisions[i].get("hinges_seen") else "push") == truth[i]["opening"] for i in both])
    return res


def _probs(out, meta, head, names):
    """Softmax (temperature of the run) of one head over the given class names, renormalised -> N x len(names) numpy array."""
    idx = [meta["heads"][head].index(c) for c in names]
    p = torch.softmax(out[head] / meta["temperature"].get(head, 1.0), 1)[:, idx]
    return (p / p.sum(1, keepdim=True)).numpy()


def _run_out(run, doors):
    """Run the model of a props run over the door boxes [(path, xywh)] (crop like the dataset, no jitter) -> (outputs per head, meta)."""
    model, meta = props.load(run, "cpu")
    xs = [input_tensor(path, crop_box(box, meta["margin"], None, None), meta["imgsz"], meta["input"]) for path, box in doors]
    if not xs:
        return {h: torch.zeros(0, len(c)) for h, c in meta["heads"].items()}, meta
    outs = []
    with torch.no_grad():
        for i in range(0, len(xs), 32):
            outs.append(model(torch.stack(xs[i:i + 32])))
    return {h: torch.cat([o[h].float() for o in outs]) for h in outs[0]}, meta


def score(door_run, handle_run, export_dir, split, preds=None, score_min=0.3, iou=0.5, type_run=None, hinges=False):
    """Cascade vs flat door classifier, on the true boxes (preds None: GT doors and GT handles) or on the detector's boxes
    (preds {image id: {boxes xyxy, scores, labels}}: detector doors matched to GT doors like props.score_on_boxes, detector handles).
    -> scores(...) + flat_acc_opening / flat_acc_hinge_side (the door run's plain argmax on the same doors); with preds also "matched".
    hinges=True (step B): hinge boxes (GT category 3, or the detector's hinge boxes) near each door feed decide(); scores gain the hinge keys."""
    from .evaluate import match
    _, dmeta = props.load(door_run, "cpu")
    ds = CropDataset(export_dir, split, "door", dmeta["imgsz"], dmeta["margin"], dmeta["min_px"], input=dmeta["input"])
    names = props.HEADS["door"]
    coco = json.loads((ds.root / "coco" / f"{split}.json").read_text())
    gt_handles = {}
    for a in coco["annotations"]:
        if a["category_id"] == CATEGORY["handle"] and not a.get("iscrowd"):
            gt_handles.setdefault(a["image_id"], []).append(a["bbox"])
    gt_hinges = {}
    for a in coco["annotations"]:
        if a["category_id"] == CATEGORY["hinge"] and not a.get("iscrowd"):
            gt_hinges.setdefault(a["image_id"], []).append(a["bbox"])
    by_image = {}
    for i, image_id in enumerate(ds.image_ids):
        by_image.setdefault(image_id, []).append(i)
    doors, handles, truth, hboxes = [], [], [], []   # per scored door: (path, xywh), its handles, true labels, hinge boxes of its image
    if preds is None:
        for i, (fn, bbox, y) in enumerate(ds.items):
            doors.append((ds.root / fn, bbox)); handles.append(gt_handles.get(ds.image_ids[i], [])); hboxes.append(gt_hinges.get(ds.image_ids[i], []))
            truth.append({h: names[h][y[h]] for h in ("opening", "hinge_side")})
    else:
        for image_id in sorted(by_image):
            p = preds.get(image_id) or preds.get(str(image_id)) or {"boxes": [], "scores": [], "labels": []}
            sc, lb = np.asarray(p["scores"], float), np.asarray(p["labels"])
            xyxy = np.asarray(p["boxes"], float).reshape(-1, 4)
            cat = lambda c: [j for j in range(len(sc)) if sc[j] >= score_min and lb[j] == CATEGORY[c] - 1]
            xywh = lambda ks: [[xyxy[k][0], xyxy[k][1], xyxy[k][2] - xyxy[k][0], xyxy[k][3] - xyxy[k][1]] for k in ks]
            keep = cat("door")
            dbox, hbox, nbox = xywh(keep), xywh(cat("handle")), xywh(cat("hinge"))
            items = by_image[image_id]
            hit = match([ds.items[i][1] for i in items], dbox, sc[keep], iou)
            for g, j in hit.items():
                fn, _, y = ds.items[items[g]]
                doors.append((ds.root / fn, dbox[j])); handles.append(hbox); hboxes.append(nbox)
                truth.append({h: names[h][y[h]] for h in ("opening", "hinge_side")})
    # handle type of the largest handle inside each door (the handle run's own crop settings, no min_px filter)
    pick = [largest_handle_inside(d[1], hs) for d, hs in zip(doors, handles)]
    idx = [i for i, k in enumerate(pick) if k is not None]
    handle_type = [None] * len(doors)
    if idx:
        out, hmeta = _run_out(handle_run, [(doors[i][0], handles[i][pick[i]]) for i in idx])
        for i, c in zip(idx, out["handle_type"].argmax(1).tolist()):
            handle_type[i] = hmeta["heads"]["handle_type"][c]
    out, meta = _run_out(door_run, doors)
    if len(doors):
        p_pp, p_lr = _probs(out, meta, "opening", ["push", "pull"]), _probs(out, meta, "hinge_side", ["left", "right"])
        if type_run:
            tout, tmeta = _run_out(type_run, doors)
            p_sl = torch.softmax(tout["door_type"] / tmeta["temperature"].get("door_type", 1.0), 1)[:, tmeta["heads"]["door_type"].index("sliding")].numpy()
        else:
            p_sl = torch.softmax(out["opening"] / meta["temperature"].get("opening", 1.0), 1)[:, meta["heads"]["opening"].index("slide")].numpy()
        def hinge_kw(i):
            if not hinges:
                return {}
            x, _, w, _ = doors[i][1]
            return {"hinge_xs": [hboxes[i][k][0] + hboxes[i][k][2] / 2 for k in hinges_near(doors[i][1], hboxes[i])], "door_cx": x + w / 2}
        dec = [decide(float(p_sl[i]), handle_type[i], tuple(p_pp[i]), props.hinge_from_handle(doors[i][1], handles[i]), tuple(p_lr[i]), **hinge_kw(i))
               for i in range(len(doors))]
    else:
        dec = []
    res = scores(dec, truth)
    for h in ("opening", "hinge_side"):
        pred = out[h].argmax(1).tolist() if len(doors) else []
        res[f"flat_acc_{h}"] = _mean([meta["heads"][h][c] == t[h] for c, t in zip(pred, truth)])
    if preds is not None:
        res["matched"] = len(doors)
    return res
