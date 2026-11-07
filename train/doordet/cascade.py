"""Door property cascade, step A (Madelief 2026-11-07): decide the door type first, then ask the hinged-only questions.
Type from the handle seen inside the door box (flush pull -> sliding, lever / knob -> hinged), else from a type classifier;
sliding -> opening "slide", hinge side "none"; hinged -> push vs pull from the opening head, left / right from the handle rule, else the hinge_side head."""
import json, math, pathlib
import numpy as np
import torch

from . import door_angle, props
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


def decide(p_sliding, handle_type, p_push_pull, rule_hinge, p_left_right, hinge_xs=None, door_cx=None, closed=None, geom=None):
    """One door's cascade answer. p_push_pull = (P(push), P(pull)), p_left_right = (P(left), P(right)); rule_hinge: "left" / "right" / None.
    -> {"door_type", "type_source" (handle | classifier), "opening", "hinge_side", "hinge_source" (rule | classifier | None)}
    Step B: hinge_xs (centre x of the hinge boxes near this door, may be empty) + door_cx add "opening_source" and "hinges_seen":
    hinges seen on a hinged door -> "pull", hinge side from their mean x vs door_cx ("hinges"); none seen -> as step A; sliding -> None, None.
    closed (None | True | False): False (open door) -> hinges ignored for opening and hinge side (as step A, opening_source "classifier"), hinges_seen still reported.
    geom ({"panel_off", "near_wall"} from door_angle.geometry, or None) with a known state (closed True / False) on a hinged door overrides only the opening and adds
    "opening_source": closed: hinges seen -> pull ("hinges"); else near_wall < -0.02 (door set in a recess) -> push, else pull ("recess"); near_wall None -> classifier.
    open: panel_off > 0 (panel in front of the wall) -> pull, < 0 -> push ("panel"); None -> classifier. closed None -> geometry not used. Hinge side unchanged."""
    b = hinge_xs is not None
    g = geom is not None
    dt = type_from_handle(handle_type)
    source = "handle" if dt else "classifier"
    dt = dt or ("sliding" if p_sliding >= 0.5 else "hinged")
    if dt == "sliding":
        res = {"door_type": dt, "type_source": source, "opening": "slide", "hinge_side": "none", "hinge_source": None}
        return {**res, "opening_source": None, "hinges_seen": None} if b else {**res, "opening_source": None} if g else res
    opening, osrc = ("push" if p_push_pull[0] >= p_push_pull[1] else "pull"), "classifier"
    if b and len(hinge_xs) and closed is not False:
        opening, osrc = "pull", "hinges"
        hinge, hsrc = ("left" if float(np.mean(hinge_xs)) < door_cx else "right"), "hinges"
    elif rule_hinge is not None:
        hinge, hsrc = rule_hinge, "rule"
    else:
        hinge, hsrc = ("left" if p_left_right[0] >= p_left_right[1] else "right"), "classifier"
    if g and closed is not None:
        if closed:
            if b and len(hinge_xs):
                opening, osrc = "pull", "hinges"
            elif geom.get("near_wall") is not None:
                opening, osrc = ("push" if geom["near_wall"] < -0.02 else "pull"), "recess"
        elif geom.get("panel_off") not in (None, 0.0):
            opening, osrc = ("pull" if geom["panel_off"] > 0 else "push"), "panel"
    res = {"door_type": dt, "type_source": source, "opening": opening, "hinge_side": hinge, "hinge_source": hsrc}
    if b:
        return {**res, "opening_source": osrc, "hinges_seen": bool(len(hinge_xs))}
    return {**res, "opening_source": osrc} if g else res


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
    if all("state" in t for t in truth) and truth:   # closed-doors-only: per true state
        for st in ("closed", "open"):
            at = [i for i in range(n) if truth[i]["state"] == st]
            res[f"acc_opening_{st}"] = _mean([decisions[i]["opening"] == truth[i]["opening"] for i in at])
            res[f"acc_push_pull_hinged_{st}"] = _mean([decisions[i]["opening"] == truth[i]["opening"] for i in at if i in both])
    if any("hinges_seen" in d for d in decisions):   # step B: hinge boxes were used
        seen = [i for i in both if decisions[i].get("hinges_seen")]
        res["hinges_seen"] = len(seen) / len(both) if both else 0.0
        res["acc_pull_when_hinges_seen"] = _mean([decisions[i]["opening"] == truth[i]["opening"] for i in seen])
        res["acc_push_pull_hinged_strict"] = _mean([("pull" if decisions[i].get("hinges_seen") else "push") == truth[i]["opening"] for i in both])
    if any(d.get("opening_source") for d in decisions):   # angle: push / pull accuracy per source of the opening answer
        for src in ("hinges", "recess", "panel", "classifier"):
            at = [i for i in both if decisions[i].get("opening_source") == src]
            if at:
                res[f"acc_push_pull_by_{src}"] = _mean([decisions[i]["opening"] == truth[i]["opening"] for i in at])
                res[f"share_push_pull_by_{src}"] = len(at) / len(both)
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


FEATURES = ["p_slide", "p_push", "p_left", "p_closed", "h_lever", "h_flush", "h_knob", "h_none", "rule_side", "hinges_n", "hinges_side",
            "panel_off", "near_wall", "box_w", "box_h"]   # the cues of the cascade as one row per door (learned_tree reads them in this order)


def _door_data(door_run, handle_run, export_dir, split, preds, score_min, iou, type_run, hinges, state_run, state_gt, angle):
    """Everything the cascade knows per door, computed once for score() and cue_table(): the door list (true boxes, or detector boxes
    matched to GT doors like props.score_on_boxes), their handles, hinge boxes, true labels, handle type, head probabilities, state and
    door_angle geometry. -> dict of parallel lists / arrays (+ "out", "meta" of the door run, "dec" = decide() per door)."""
    from .evaluate import match
    _, dmeta = props.load(door_run, "cpu")
    ds = CropDataset(export_dir, split, "door", dmeta["imgsz"], dmeta["margin"], dmeta["min_px"], input=dmeta["input"])
    names = props.HEADS["door"]
    coco = json.loads((ds.root / "coco" / f"{split}.json").read_text())
    gt_handles, gt_hinges = {}, {}
    for a in coco["annotations"]:
        if a.get("iscrowd"):
            continue
        if a["category_id"] == CATEGORY["handle"]:
            gt_handles.setdefault(a["image_id"], []).append(a["bbox"])
        if a["category_id"] == CATEGORY["hinge"]:
            gt_hinges.setdefault(a["image_id"], []).append(a["bbox"])
    by_image = {}
    for i, image_id in enumerate(ds.image_ids):
        by_image.setdefault(image_id, []).append(i)
    doors, handles, truth, hboxes = [], [], [], []   # per scored door: (path, xywh), its handles, true labels, hinge boxes of its image
    if preds is None:
        for i, (fn, bbox, y) in enumerate(ds.items):
            doors.append((ds.root / fn, bbox)); handles.append(gt_handles.get(ds.image_ids[i], [])); hboxes.append(gt_hinges.get(ds.image_ids[i], []))
            truth.append({h: names[h][y[h]] for h in ("opening", "hinge_side", "state") if h in y})
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
                truth.append({h: names[h][y[h]] for h in ("opening", "hinge_side", "state") if h in y})
    # handle type of the largest handle inside each door (the handle run's own crop settings, no min_px filter)
    pick = [largest_handle_inside(d[1], hs) for d, hs in zip(doors, handles)]
    idx = [i for i, k in enumerate(pick) if k is not None]
    handle_type = [None] * len(doors)
    if idx:
        out, hmeta = _run_out(handle_run, [(doors[i][0], handles[i][pick[i]]) for i in idx])
        for i, c in zip(idx, out["handle_type"].argmax(1).tolist()):
            handle_type[i] = hmeta["heads"]["handle_type"][c]
    out, meta = _run_out(door_run, doors)
    d = {"doors": doors, "handles": handles, "truth": truth, "handle_type": handle_type, "out": out, "meta": meta, "dec": [], "closed": [None] * len(doors),
         "p_closed": [math.nan] * len(doors), "geoms": [None] * len(doors), "hinge_xs": [[] for _ in doors], "rule": [None] * len(doors),
         "p_sl": np.zeros(0), "p_pp": np.zeros((0, 2)), "p_lr": np.zeros((0, 2))}
    if not len(doors):
        return d
    d["p_pp"], d["p_lr"] = _probs(out, meta, "opening", ["push", "pull"]), _probs(out, meta, "hinge_side", ["left", "right"])
    if type_run:
        tout, tmeta = _run_out(type_run, doors)
        d["p_sl"] = torch.softmax(tout["door_type"] / tmeta["temperature"].get("door_type", 1.0), 1)[:, tmeta["heads"]["door_type"].index("sliding")].numpy()
    else:
        d["p_sl"] = torch.softmax(out["opening"] / meta["temperature"].get("opening", 1.0), 1)[:, meta["heads"]["opening"].index("slide")].numpy()
    if hinges:
        d["hinge_xs"] = [[hboxes[i][k][0] + hboxes[i][k][2] / 2 for k in hinges_near(doors[i][1], hboxes[i])] for i in range(len(doors))]
    if state_gt:
        d["closed"] = [t.get("state") == "closed" for t in truth]
    elif state_run:
        sout, smeta = _run_out(state_run, doors)
        pc = _probs(sout, smeta, "state", ["closed", "open"])[:, 0]
        d["p_closed"], d["closed"] = pc.tolist(), (pc >= 0.5).tolist()
    if angle:
        cache = {}
        for i, (path, box) in enumerate(doors):
            if path not in cache:
                cache[path] = door_angle.depth_for(path)
            if cache[path] is not None:
                d["geoms"][i] = door_angle.geometry(*cache[path], box)
    d["rule"] = [props.hinge_from_handle(doors[i][1], handles[i]) for i in range(len(doors))]
    d["dec"] = [decide(float(d["p_sl"][i]), handle_type[i], tuple(d["p_pp"][i]), d["rule"][i], tuple(d["p_lr"][i]), closed=d["closed"][i],
                       geom=d["geoms"][i] if angle else None,
                       **({"hinge_xs": d["hinge_xs"][i], "door_cx": doors[i][1][0] + doors[i][1][2] / 2} if hinges else {}))
                for i in range(len(doors))]
    return d


def score(door_run, handle_run, export_dir, split, preds=None, score_min=0.3, iou=0.5, type_run=None, hinges=False, state_run=None, state_gt=False, angle=False):
    """Cascade vs flat door classifier, on the true boxes (preds None: GT doors and GT handles) or on the detector's boxes
    (preds {image id: {boxes xyxy, scores, labels}}: detector doors matched to GT doors like props.score_on_boxes, detector handles).
    -> scores(...) + flat_acc_opening / flat_acc_hinge_side (the door run's plain argmax on the same doors); with preds also "matched".
    hinges=True (step B): hinge boxes (GT category 3, or the detector's hinge boxes) near each door feed decide(); scores gain the hinge keys.
    Closed-doors-only: closed per door = the true state (state_gt) or the state run's head (P(closed) >= 0.5), else None; passed to decide(); acc_state = share of doors
    whose closed / open used equals the true state (needs a state label).
    angle=True: door_angle.geometry per scored door (its box, true or detector) from its frame's sensor depth (cached per image; no depth.png -> None) goes to decide(geom=)."""
    d = _door_data(door_run, handle_run, export_dir, split, preds, score_min, iou, type_run, hinges, state_run, state_gt, angle)
    doors, truth, dec, out, meta = d["doors"], d["truth"], d["dec"], d["out"], d["meta"]
    res = scores(dec, truth)
    if (state_gt or state_run) and dec and all("state" in t for t in truth):
        res["acc_state"] = _mean([c == (t["state"] == "closed") for c, t in zip(d["closed"], truth)])
    for h in ("opening", "hinge_side"):
        pred = out[h].argmax(1).tolist() if len(doors) else []
        res[f"flat_acc_{h}"] = _mean([meta["heads"][h][c] == t[h] for c, t in zip(pred, truth)])
    if preds is not None:
        res["matched"] = len(doors)
    return res


def cue_table(door_run, handle_run, export_dir, split, preds=None, score_min=0.3, iou=0.5, type_run=None, state_run=None, hinges=True, angle=True, with_decisions=False):
    """The cascade's cues as one row per scored door (same doors as score()), for the learned tree: -> (rows, truth) with rows = dicts keyed by FEATURES
    (P(slide) / P(push) / P(left) as decide sees them, P(closed) from the state run, handle type one-hot, handle rule side, hinge boxes near the door,
    door_angle geometry, door box size) and truth = [{"opening", "hinge_side"[, "state"]}]. Missing cues are NaN (no state run, no depth / wall).
    with_decisions=True also returns the hand cascade's decide() answers: -> (rows, truth, decisions)."""
    d = _door_data(door_run, handle_run, export_dir, split, preds, score_min, iou, type_run, hinges, state_run, False, angle)
    side = {"right": 1.0, "left": -1.0}
    rows = []
    for i, (_, box) in enumerate(d["doors"]):
        hx = d["hinge_xs"][i]
        g = d["geoms"][i] or {}
        num = lambda v: math.nan if v is None else float(v)
        ht = d["handle_type"][i]
        rows.append({"p_slide": float(d["p_sl"][i]), "p_push": float(d["p_pp"][i][0]), "p_left": float(d["p_lr"][i][0]), "p_closed": d["p_closed"][i],
                     "h_lever": float(ht == "lever"), "h_knob": float(ht == "knob"), "h_flush": float(ht == "flush_pull"), "h_none": float(ht is None),
                     "rule_side": side.get(d["rule"][i], 0.0), "hinges_n": float(len(hx)),
                     "hinges_side": 0.0 if not hx else (-1.0 if float(np.mean(hx)) < box[0] + box[2] / 2 else 1.0),
                     "panel_off": num(g.get("panel_off")), "near_wall": num(g.get("near_wall")), "box_w": float(box[2]), "box_h": float(box[3])})
    rows = [{f: r[f] for f in FEATURES} for r in rows]
    return (rows, d["truth"], d["dec"]) if with_decisions else (rows, d["truth"])
