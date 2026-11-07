"""Door property + handle type classifier on box crops: shared backbone, one linear head per property, temperature-scaled confidence.
Input: the rgb crop, the depth-derived dha crop (depth / height / normal angle, data.encode_depth) or both stacked (rgb_dha, 6 channels)."""
import datetime, json, math, pathlib, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import REPO
from .data import load_input
from .loop import flatten, git_info

HEADS = {"door": {"opening": ["push", "pull", "slide"], "hinge_side": ["left", "right", "none"], "glass": ["no", "yes"],
                  "door_type": ["hinged", "sliding"]},  # door_type: opt-in head (cascade step A), not in the default heads
         "handle": {"handle_type": ["lever", "knob", "flush_pull"]}}
DEFAULT_HEADS = {"door": ["opening", "hinge_side", "glass"]}  # crops not listed: all their heads
CATEGORY = {"door": 1, "handle": 2}
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
DHA_MEAN, DHA_STD = 0.5, 0.25  # dha channels are 0..255 maps, not photos
CHANNELS = {"rgb": 3, "dha": 3, "rgb_dha": 6}
BACKBONES = ("resnet18", "resnet34", "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0")
PER_CROP = {"imgsz": {"door": 224, "handle": 96}, "margin": {"door": 0.1, "handle": 0.5}, "min_px": {"door": 16, "handle": 6}}
DEFAULTS = {
    "export": None, "crop": None,  # required: exporter output folder, "door" or "handle"
    "input": "rgb",  # rgb | dha | rgb_dha (dha: depth, height above floor, normal angle)
    "heads": None,  # None -> the crop's default heads (all, except the opt-in door_type)
    "backbone": "resnet18", "pretrained": True, "imgsz": None, "margin": None, "min_px": None,  # None -> per-crop default
    "jitter": {"shift": 0.1, "scale": 0.15},  # train-time crop shift / scale, fractions of the long box side
    "epochs": 20, "batch": 32, "lr": 3e-4, "weight_decay": 1e-4, "amp": False, "workers": 4, "seed": 0, "device": "cuda",
    "min_calibration": 20,  # val crops needed per head to fit a temperature (else T = 1)
    "out": str(REPO / "runs"), "name": None,
    "mlflow": {"tracking_uri": f"sqlite:///{REPO / 'mlflow.db'}", "experiment": "doggy-door-props", "tags": {}},
}
NESTED = ("jitter", "mlflow")


def resolve(d):
    unknown = sorted(set(d) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown config key(s): {', '.join(unknown)}")
    cfg = {k: (dict(v) if k in NESTED else v) for k, v in DEFAULTS.items()}
    for k, v in d.items():
        if k in NESTED:
            bad = sorted(set(v) - set(DEFAULTS[k]))
            if bad:
                raise ValueError(f"unknown config key(s) in {k}: {', '.join(bad)}")
            cfg[k].update(v)
        else:
            cfg[k] = v
    if not cfg["export"]:
        raise ValueError("config key 'export' is required")
    if cfg["crop"] not in HEADS:
        raise ValueError(f"config key 'crop' must be one of {', '.join(HEADS)}")
    if cfg["input"] not in CHANNELS:
        raise ValueError(f"unknown input {cfg['input']!r}; available: {', '.join(CHANNELS)}")
    if cfg["backbone"] not in BACKBONES:
        raise ValueError(f"unknown backbone {cfg['backbone']!r}; available: {', '.join(BACKBONES)}")
    avail = list(HEADS[cfg["crop"]])
    cfg["heads"] = list(DEFAULT_HEADS.get(cfg["crop"], avail)) if cfg["heads"] is None else list(cfg["heads"])
    bad = [h for h in cfg["heads"] if h not in avail]
    if bad:
        raise ValueError(f"heads {bad} do not belong to crop {cfg['crop']!r}; available: {', '.join(avail)}")
    for k, per in PER_CROP.items():
        if cfg[k] is None:
            cfg[k] = per[cfg["crop"]]
    return cfg


def labels(attrs, crop):
    """COCO attributes -> {head: class name}."""
    if crop == "handle":
        return {"handle_type": attrs.get("handle_type")}
    return {"opening": attrs.get("opening"), "hinge_side": attrs.get("hinge_side") or "none", "glass": "yes" if attrs.get("glass") else "no",
            "door_type": attrs.get("door_type") or ("sliding" if attrs.get("opening") == "slide" else "hinged")}


def crop_box(bbox, margin, rng, jitter):
    """Square crop (x0, y0, x1, y1) around an xywh box, side = long side x (1 + 2 margin); jitter shifts / scales it. Not clipped."""
    x, y, w, h = bbox
    cx, cy, long = x + w / 2, y + h / 2, max(w, h)
    side = long * (1 + 2 * margin)
    if jitter and rng is not None:
        cx += rng.uniform(-jitter["shift"], jitter["shift"]) * long
        cy += rng.uniform(-jitter["shift"], jitter["shift"]) * long
        side *= rng.uniform(1 - jitter["scale"], 1 + jitter["scale"])
    return cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2


def crop_tensor(img, box, imgsz, mean=MEAN, std=STD):
    """Crop box (x0, y0, x1, y1) of a PIL image -> normalised 3 x imgsz x imgsz tensor."""
    # transform maps output pixels to source coordinates, so outside the image the crop is zero padded
    s = (box[2] - box[0]) / imgsz
    crop = img.transform((imgsz, imgsz), Image.AFFINE, (s, 0, box[0], 0, s, box[1]), Image.BILINEAR, fillcolor=(0, 0, 0))
    x = (np.asarray(crop, np.float32) / 255 - mean) / std
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


def input_tensor(path, box, imgsz, input, K_cache=None):
    """Export image path -> crop tensor of the run's input: rgb (ImageNet norm), dha (DHA_MEAN / DHA_STD) or both stacked."""
    parts = []
    if input in ("rgb", "rgb_dha"):
        parts.append(crop_tensor(load_input(path, "rgb"), box, imgsz))
    if input in ("dha", "rgb_dha"):
        parts.append(crop_tensor(load_input(path, "dha", K_cache), box, imgsz, DHA_MEAN, DHA_STD))
    return torch.cat(parts)


class CropDataset(Dataset):
    """Crops of door or handle boxes from a COCO export; item = (normalised C x imgsz x imgsz tensor, {head: class index}), C = CHANNELS[input]."""
    def __init__(self, export_dir, split, crop, imgsz, margin, min_px, jitter=None, seed=0, input="rgb"):
        self.root, self.imgsz, self.margin, self.jitter, self.input, self._K = pathlib.Path(export_dir), imgsz, margin, jitter, input, {}
        self.rng = np.random.default_rng(seed)
        coco = json.loads((self.root / "coco" / f"{split}.json").read_text())
        files = {im["id"]: im["file_name"] for im in coco["images"]}
        self.items, self.heldout, self.image_ids = [], [], []  # heldout: the item's property combo was never in training (sim worlds with held-out doors)
        for a in coco["annotations"]:
            if a["category_id"] != CATEGORY[crop] or min(a["bbox"][2:]) < min_px or a.get("iscrowd"):  # ignore boxes: nearly vanished in smoke
                continue
            attrs = a.get("attributes") or {}
            lab = labels(attrs, crop)
            if all(v in HEADS[crop][h] for h, v in lab.items()):
                self.items.append((files[a["image_id"]], a["bbox"], {h: HEADS[crop][h].index(v) for h, v in lab.items()}))
                self.heldout.append(bool(attrs.get("heldout")))
                self.image_ids.append(a["image_id"])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fn, bbox, y = self.items[i]
        box = crop_box(bbox, self.margin, self.rng, self.jitter)  # one box for every image of the input
        return input_tensor(self.root / fn, box, self.imgsz, self.input, self._K), y


def largest_handle_inside(door_xywh, handle_xywhs):
    """Index of the largest handle box whose centre lies inside the door box (None if there is none); shared by the hinge rule and the cascade."""
    x, y, w, h = door_xywh
    inside = [i for i, b in enumerate(handle_xywhs) if x <= b[0] + b[2] / 2 <= x + w and y <= b[1] + b[3] / 2 <= y + h]
    return max(inside, key=lambda i: handle_xywhs[i][2] * handle_xywhs[i][3]) if inside else None


def hinge_from_handle(door_xywh, handle_xywhs):
    """Hinge side from the handle position (Madelief 2026-11-07: the hinge is opposite the handle, which sits on the lock edge).
    The largest handle box whose centre lies inside the door box decides: handle left of the door centre -> "right", else "left"; none inside -> None."""
    x, y, w, h = door_xywh
    i = largest_handle_inside(door_xywh, handle_xywhs)
    if i is None:
        return None
    b = handle_xywhs[i]
    return "right" if b[0] + b[2] / 2 < x + w / 2 else "left"


def combine_hinge(classifier_label, rule_label):
    """Classifier's "none" (sliding) stays; otherwise the handle rule when it has an answer, else the classifier."""
    return classifier_label if classifier_label == "none" or rule_label is None else rule_label


def hinge_rule_true_boxes(export_dir, split, classifier_pred=None, min_px=None):
    """Handle rule on the GT boxes: the CropDataset's door items (same filters) and the non-crowd GT handles of their image.
    -> {"rule_coverage_hinge_side" (share of hinged doors the rule answers), "acc_hinge_side_rule" (over those answered; 0.0 if none)};
    with classifier_pred (hinge_side class names in dataset order) also "acc_hinge_side_combined" over all doors."""
    # min_px = the run's (so the door list lines up with the classifier's predictions); margin / imgsz don't change which items are kept
    ds = CropDataset(export_dir, split, "door", PER_CROP["imgsz"]["door"], PER_CROP["margin"]["door"], min_px or PER_CROP["min_px"]["door"])
    coco = json.loads((ds.root / "coco" / f"{split}.json").read_text())
    handles = {}
    for a in coco["annotations"]:
        if a["category_id"] == CATEGORY["handle"] and not a.get("iscrowd"):
            handles.setdefault(a["image_id"], []).append(a["bbox"])
    names = HEADS["door"]["hinge_side"]
    rules = [hinge_from_handle(it[1], handles.get(im, [])) for it, im in zip(ds.items, ds.image_ids)]
    truth = [names[it[2]["hinge_side"]] for it in ds.items]
    return _hinge_scores(rules, truth, classifier_pred)


def _hinge_scores(rules, truth, classifier_pred=None):
    """Rule / combined hinge-side numbers from per-door rule answers (None = no handle), true labels and optional classifier names."""
    hinged = [i for i, t in enumerate(truth) if t != "none"]
    answered = [i for i in hinged if rules[i] is not None]
    out = {"rule_coverage_hinge_side": len(answered) / len(hinged) if hinged else 0.0,
           "acc_hinge_side_rule": float(np.mean([rules[i] == truth[i] for i in answered])) if answered else 0.0}
    if classifier_pred is not None and truth:
        out["acc_hinge_side_combined"] = float(np.mean([combine_hinge(c, r) == t for c, r, t in zip(classifier_pred, rules, truth)]))
    return out


def reseed_worker(worker_id):
    """DataLoader workers get a copy of the dataset's jitter generator, and are restarted every epoch: reseed it from
    torch's per-worker seed (new every epoch, fixed by the loader's generator) so jitter differs between workers and epochs."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)


def widen_stem(conv, in_ch):
    """First conv with in_ch inputs: the rgb weights are kept, extra channels start as the mean of the rgb weights (the usual way to add depth channels to a pretrained net)."""
    if in_ch == conv.in_channels:
        return conv
    new = nn.Conv2d(in_ch, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups, conv.bias is not None)
    with torch.no_grad():
        new.weight[:, :conv.in_channels] = conv.weight
        new.weight[:, conv.in_channels:] = conv.weight.mean(1, keepdim=True)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


class Model(nn.Module):
    """torchvision backbone (classifier removed, global pool) + one linear layer per head."""
    def __init__(self, backbone, heads, pretrained, in_ch=3):
        super().__init__()
        import torchvision.models as tvm
        net = getattr(tvm, backbone)(weights="DEFAULT" if pretrained else None)
        if backbone.startswith("resnet"):
            net.conv1 = widen_stem(net.conv1, in_ch)
            dim, net.fc = net.fc.in_features, nn.Identity()
        else:
            net.features[0][0] = widen_stem(net.features[0][0], in_ch)
            dim = next(m for m in net.classifier if isinstance(m, nn.Linear)).in_features  # the first Linear sees the pooled features
            net.classifier = nn.Identity()
        self.backbone = net
        self.heads = nn.ModuleDict({h: nn.Linear(dim, len(c)) for h, c in heads.items()})

    def forward(self, x):
        f = self.backbone(x)
        return {h: m(f) for h, m in self.heads.items()}


def ece(conf, correct, bins=10):
    """Expected calibration error: equal-width confidence bins, weighted by bin count."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.sum() * abs(conf[m].mean() - correct[m].mean())
    return float(total / max(len(conf), 1))


def fit_temperature(logits, labels):
    """T > 0 minimising the NLL of logits / T (temperature scaling, Guo et al. 2017)."""
    logits, labels = logits.detach().double(), labels.detach().long()
    log_t = torch.zeros(1, dtype=torch.double, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.detach().exp().clamp(1e-2, 1e2))


def predict(model, loader, device, heads):
    """-> {head: (logits, labels)} over a loader."""
    model.eval()
    lg, lb = {h: [] for h in heads}, {h: [] for h in heads}
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            for h in heads:
                lg[h].append(out[h].float().cpu()); lb[h].append(y[h])
    return {h: (torch.cat(lg[h]), torch.cat(lb[h])) for h in heads}


def train(cfg, run_name=None):
    import mlflow
    seed, crop, heads = cfg["seed"], cfg["crop"], cfg["heads"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device(cfg["device"])
    export = pathlib.Path(cfg["export"])
    run_name = run_name or cfg["name"] or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = pathlib.Path(cfg["out"]) / run_name
    out.mkdir(parents=True, exist_ok=True)

    kw = {k: cfg[k] for k in ("imgsz", "margin", "min_px", "input")}
    sets = {s: CropDataset(export, s, crop, jitter=cfg["jitter"] if s == "train" else None, seed=seed, **kw) for s in ("train", "val", "test")}
    gen = torch.Generator().manual_seed(seed)
    mk = lambda s, shuffle: DataLoader(sets[s], batch_size=cfg["batch"], shuffle=shuffle, num_workers=cfg["workers"], generator=gen if shuffle else None,
                                       worker_init_fn=reseed_worker)
    loaders = {s: mk(s, s == "train") for s in sets}
    classes = {h: HEADS[crop][h] for h in heads}
    model = Model(cfg["backbone"], classes, cfg["pretrained"], CHANNELS[cfg["input"]]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = max(cfg["epochs"] * len(loaders["train"]), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    amp = cfg["amp"] and device.type == "cuda"

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    export_info = json.loads((export / "export.json").read_text()) if (export / "export.json").exists() else {}
    src = export_info.get("source") or {}
    best = {"value": -1.0, "epoch": None, "state": None}
    meta = lambda temps: {"heads": classes, "backbone": cfg["backbone"], "imgsz": cfg["imgsz"], "crop": crop, "margin": cfg["margin"], "input": cfg["input"], "temperature": temps}
    (out / "best").mkdir(exist_ok=True)
    with mlflow.start_run(run_name=run_name) as run:
        params = dict(flatten(cfg))
        params.update({"export.git_commit": (src.get("git") or {}).get("commit"), "export.seed": export_info.get("seed"),
                       "export.source": src.get("path"), **git_info()})
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.set_tags({k: str(v) for k, v in cfg["mlflow"]["tags"].items()})
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
        mlflow.log_artifact(str(out / "config.json"))

        for epoch in range(cfg["epochs"]):
            model.train()
            losses = []
            for x, y in loaders["train"]:
                with torch.autocast(device.type, enabled=amp):
                    o = model(x.to(device))
                    loss = sum(F.cross_entropy(o[h].float(), y[h].to(device)) for h in heads)
                opt.zero_grad(set_to_none=True)
                loss.backward(); opt.step(); sched.step()
                losses.append(loss.item())
            if losses:
                mlflow.log_metric("train_loss", float(np.mean(losses)), step=epoch)
            if len(sets["val"]):
                pr = predict(model, loaders["val"], device, heads)
                acc = {h: (lg.argmax(1) == lb).float().mean().item() for h, (lg, lb) in pr.items()}
                mean = float(np.mean(list(acc.values())))
                mlflow.log_metrics({**{f"val_acc_{h}": a for h, a in acc.items()}, "val_acc_mean": mean}, step=epoch)
                print(f"epoch {epoch} loss {np.mean(losses) if losses else float('nan'):.4f} val_acc {mean:.4f}", flush=True)
                if mean > best["value"]:
                    best.update(value=mean, epoch=epoch, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                    torch.save(best["state"], out / "best" / "model.pt")
        if best["state"] is None:  # no val items: keep the last weights
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best["state"], out / "best" / "model.pt")
        model.load_state_dict(best["state"])

        temps, test = {}, {}
        if len(sets["val"]):
            pr = predict(model, loaders["val"], device, heads)
            # too few val crops (or none wrong) gives a meaningless temperature (it runs to the clamp): keep T = 1
            temps = {h: fit_temperature(lg, lb) if len(lb) >= cfg["min_calibration"] and (lg.argmax(1) != lb).any() else 1.0
                     for h, (lg, lb) in pr.items()}
            mlflow.log_metrics({f"temperature_{h}": t for h, t in temps.items()})
        (out / "best" / "heads.json").write_text(json.dumps(meta(temps), indent=2))
        if len(sets["test"]):
            held = torch.tensor(sets["test"].heldout, dtype=torch.bool)
            for h, (lg, lb) in predict(model, loaders["test"], device, heads).items():
                for tag, T in (("", temps.get(h, 1.0)), ("_raw", 1.0)):
                    p = torch.softmax(lg / T, 1)
                    conf, pred = p.max(1)
                    test[f"test_ece{tag}_{h}"] = ece(conf.numpy(), (pred == lb).numpy())
                test[f"test_acc_{h}"] = (lg.argmax(1) == lb).float().mean().item()
                if held.any():  # the test loader does not shuffle: logits are in dataset order
                    hit = (lg.argmax(1) == lb).float()
                    for tag, grp in (("seen", ~held), ("unseen", held)):
                        if grp.any():
                            test[f"test_acc_{tag}_{h}"] = hit[grp].mean().item()
            mlflow.log_metrics(test)
            print("test " + "  ".join(f"{k} {v:.4f}" for k, v in test.items()), flush=True)
        return {"run_id": run.info.run_id, "out": str(out), "best_epoch": best["epoch"], "best_val_acc": best["value"],
                "temperature": temps, "test": test}


def load(run_dir, device="cpu"):
    """Trained props run -> (model in eval mode, heads.json meta: heads, backbone, imgsz, crop, margin, input, temperature; config.json keys min_px)."""
    run_dir = pathlib.Path(run_dir)
    meta = json.loads((run_dir / "best" / "heads.json").read_text())
    meta["min_px"] = json.loads((run_dir / "config.json").read_text())["min_px"]
    meta["input"] = meta.get("input", "rgb")  # runs from before the input option
    model = Model(meta["backbone"], meta["heads"], pretrained=False, in_ch=CHANNELS[meta["input"]])
    model.load_state_dict(torch.load(run_dir / "best" / "model.pt", map_location="cpu"))
    return model.to(device).eval(), meta


def score_on_boxes(run_dir, export_dir, split, preds, score_min=0.3, iou=0.5):
    """Classifier accuracy on the detector's own boxes (Madelief 2026-11-02: score on true boxes and on detector boxes).
    Per image the detector's boxes of the crop's category (score >= score_min) are matched to the GT crops the dataset keeps;
    a matched box is cropped like the dataset does (no jitter) and compared with its GT's labels.
    -> {"crops" GT items, "matched", "missed", "false" (unmatched predictions), "acc_<head>", "ece_<head>" over the matched;
    door crops with a hinge_side head also the handle rule from the detector's handles: rule_coverage_hinge_side, acc_hinge_side_rule, acc_hinge_side_combined}"""
    from .evaluate import match
    model, meta = load(run_dir, "cpu")
    crop, heads, imgsz = meta["crop"], list(meta["heads"]), meta["imgsz"]
    ds = CropDataset(export_dir, split, crop, imgsz, meta["margin"], meta["min_px"], input=meta["input"])
    by_image = {}
    for i, image_id in enumerate(ds.image_ids):
        by_image.setdefault(image_id, []).append(i)
    xs, ys, rules, matched, false = [], [], [], 0, 0
    hinge = crop == "door" and "hinge_side" in heads
    for image_id in sorted(set(by_image) | {int(k) for k in preds}):
        items = by_image.get(image_id, [])
        p = preds.get(image_id) or preds.get(str(image_id)) or {"boxes": [], "scores": [], "labels": []}
        keep = [j for j, (s, l) in enumerate(zip(p["scores"], p["labels"])) if s >= score_min and l == CATEGORY[crop] - 1]
        boxes = np.asarray(p["boxes"], float).reshape(-1, 4)[keep]
        xywh = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
        if hinge:  # the detector's own handles of this image
            hk = [j for j, (s, l) in enumerate(zip(p["scores"], p["labels"])) if s >= score_min and l == CATEGORY["handle"] - 1]
            hb = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in np.asarray(p["boxes"], float).reshape(-1, 4)[hk]]
        hit = match([ds.items[i][1] for i in items], xywh, np.asarray(p["scores"], float)[keep], iou)
        false += len(keep) - len(hit)
        if not hit:
            continue
        path = ds.root / ds.items[items[0]][0]
        for g, j in hit.items():
            xs.append(input_tensor(path, crop_box(xywh[j], meta["margin"], None, None), imgsz, meta["input"], ds._K))
            ys.append(ds.items[items[g]][2])
            if hinge:
                rules.append(hinge_from_handle(xywh[j], hb))
            matched += 1
    out = {"crops": len(ds), "matched": matched, "missed": len(ds) - matched, "false": false}
    if xs:
        with torch.no_grad():
            o = model(torch.stack(xs))
        for h in heads:
            y = torch.tensor([t[h] for t in ys])
            conf, pred = torch.softmax(o[h] / meta["temperature"].get(h, 1.0), 1).max(1)
            out[f"acc_{h}"] = (pred == y).float().mean().item()
            out[f"ece_{h}"] = ece(conf.numpy(), (pred == y).numpy())
            if h == "hinge_side" and hinge:
                names = HEADS["door"]["hinge_side"]
                out.update(_hinge_scores(rules, [names[i] for i in y.tolist()], [names[i] for i in pred.tolist()]))
    return out
