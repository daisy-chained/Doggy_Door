"""Door property + handle type classifier on box crops: shared backbone, one linear head per property, temperature-scaled confidence."""
import datetime, json, math, pathlib, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import REPO
from .loop import flatten, git_info

HEADS = {"door": {"opening": ["push", "pull", "slide"], "hinge_side": ["left", "right", "none"], "glass": ["no", "yes"]},
         "handle": {"handle_type": ["lever", "knob", "flush_pull"]}}
CATEGORY = {"door": 1, "handle": 2}
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
BACKBONES = ("resnet18", "resnet34", "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0")
PER_CROP = {"imgsz": {"door": 224, "handle": 96}, "margin": {"door": 0.1, "handle": 0.5}, "min_px": {"door": 16, "handle": 6}}
DEFAULTS = {
    "export": None, "crop": None,  # required: exporter output folder, "door" or "handle"
    "heads": None,  # None -> all heads of the crop
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
    if cfg["backbone"] not in BACKBONES:
        raise ValueError(f"unknown backbone {cfg['backbone']!r}; available: {', '.join(BACKBONES)}")
    avail = list(HEADS[cfg["crop"]])
    cfg["heads"] = avail if cfg["heads"] is None else list(cfg["heads"])
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
    return {"opening": attrs.get("opening"), "hinge_side": attrs.get("hinge_side") or "none", "glass": "yes" if attrs.get("glass") else "no"}


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


class CropDataset(Dataset):
    """Crops of door or handle boxes from a COCO export; item = (normalised 3 x imgsz x imgsz tensor, {head: class index})."""
    def __init__(self, export_dir, split, crop, imgsz, margin, min_px, jitter=None, seed=0):
        self.root, self.imgsz, self.margin, self.jitter = pathlib.Path(export_dir), imgsz, margin, jitter
        self.rng = np.random.default_rng(seed)
        coco = json.loads((self.root / "coco" / f"{split}.json").read_text())
        files = {im["id"]: im["file_name"] for im in coco["images"]}
        self.items, self.heldout = [], []  # heldout: the item's property combo was never in training (sim worlds with held-out doors)
        for a in coco["annotations"]:
            if a["category_id"] != CATEGORY[crop] or min(a["bbox"][2:]) < min_px:
                continue
            attrs = a.get("attributes") or {}
            lab = labels(attrs, crop)
            if all(v in HEADS[crop][h] for h, v in lab.items()):
                self.items.append((files[a["image_id"]], a["bbox"], {h: HEADS[crop][h].index(v) for h, v in lab.items()}))
                self.heldout.append(bool(attrs.get("heldout")))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fn, bbox, y = self.items[i]
        box = crop_box(bbox, self.margin, self.rng, self.jitter)
        img = Image.open(self.root / fn).convert("RGB")
        # transform maps output pixels to source coordinates, so outside the image the crop is zero padded
        s = (box[2] - box[0]) / self.imgsz
        crop = img.transform((self.imgsz, self.imgsz), Image.AFFINE, (s, 0, box[0], 0, s, box[1]), Image.BILINEAR, fillcolor=(0, 0, 0))
        x = (np.asarray(crop, np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(x).permute(2, 0, 1).contiguous(), y


def reseed_worker(worker_id):
    """DataLoader workers get a copy of the dataset's jitter generator, and are restarted every epoch: reseed it from
    torch's per-worker seed (new every epoch, fixed by the loader's generator) so jitter differs between workers and epochs."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)


class Model(nn.Module):
    """torchvision backbone (classifier removed, global pool) + one linear layer per head."""
    def __init__(self, backbone, heads, pretrained):
        super().__init__()
        import torchvision.models as tvm
        net = getattr(tvm, backbone)(weights="DEFAULT" if pretrained else None)
        if backbone.startswith("resnet"):
            dim, net.fc = net.fc.in_features, nn.Identity()
        else:
            dim = next(m for m in reversed(net.classifier) if isinstance(m, nn.Linear)).in_features
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

    kw = {k: cfg[k] for k in ("imgsz", "margin", "min_px")}
    sets = {s: CropDataset(export, s, crop, jitter=cfg["jitter"] if s == "train" else None, seed=seed, **kw) for s in ("train", "val", "test")}
    gen = torch.Generator().manual_seed(seed)
    mk = lambda s, shuffle: DataLoader(sets[s], batch_size=cfg["batch"], shuffle=shuffle, num_workers=cfg["workers"], generator=gen if shuffle else None,
                                       worker_init_fn=reseed_worker)
    loaders = {s: mk(s, s == "train") for s in sets}
    classes = {h: HEADS[crop][h] for h in heads}
    model = Model(cfg["backbone"], classes, cfg["pretrained"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = max(cfg["epochs"] * len(loaders["train"]), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    amp = cfg["amp"] and device.type == "cuda"

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    export_info = json.loads((export / "export.json").read_text()) if (export / "export.json").exists() else {}
    src = export_info.get("source") or {}
    best = {"value": -1.0, "epoch": None, "state": None}
    meta = lambda temps: {"heads": classes, "backbone": cfg["backbone"], "imgsz": cfg["imgsz"], "crop": crop, "margin": cfg["margin"], "temperature": temps}
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
