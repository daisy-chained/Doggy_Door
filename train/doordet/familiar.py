"""'This door looks unfamiliar' check: relative Mahalanobis distance (Ren et al. 2021) on the backbone features, groups = seen door types."""
import json, pathlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from . import props


def group_ids(labels, heads):
    """-> (group id per item, combos): a combo is the tuple of the head labels, ids number the combos in first-seen order."""
    n = len(labels[heads[0]])
    seen, combos, ids = {}, [], np.zeros(n, int)
    for i in range(n):
        c = tuple(int(labels[h][i]) for h in heads)
        if c not in seen:
            seen[c] = len(combos); combos.append(c)
        ids[i] = seen[c]
    return ids, combos


def _prec(x, reg):
    cov = np.atleast_2d(np.cov(x, rowvar=False, bias=True))
    return np.linalg.pinv(cov + reg * np.mean(np.diag(cov)) * np.eye(len(cov)), hermitian=True)  # pinv: tiny sets give singular covariances


def fit(feats, groups, reg=1e-3):
    x, groups = np.asarray(feats, np.float64), np.asarray(groups)
    ks = range(int(groups.max()) + 1)
    means = np.stack([x[groups == k].mean(0) for k in ks])
    within = x - means[groups]  # shared covariance = pooled within-group scatter
    return {"means": means, "mu0": x.mean(0), "prec": _prec(within, reg), "prec0": _prec(x, reg)}


def _md(x, mu, prec):
    d = x - mu
    return np.einsum("nd,de,ne->n", d, prec, d)


def score(stats, feats, relative=True):
    """min_k md_k - md0 (relative) or min_k md_k; higher = more unfamiliar."""
    x = np.asarray(feats, np.float64)
    s = np.stack([_md(x, m, stats["prec"]) for m in stats["means"]]).min(0)
    return s - _md(x, stats["mu0"], stats["prec0"]) if relative else s


def threshold(val_scores, keep=0.95):
    return float(np.quantile(val_scores, keep))


def auroc(scores, positive):
    scores, positive = np.asarray(scores, np.float64), np.asarray(positive, bool)
    p, n = scores[positive], scores[~positive]
    if not len(p) or not len(n):
        return float("nan")
    return float(((p[:, None] > n[None]).sum() + 0.5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n)))


def features(model, loader, device):
    """-> (backbone features N x D on cpu, {head: label tensor N})."""
    model.eval()
    fs, ys = [], {}
    with torch.no_grad():
        for x, y in loader:
            fs.append(model.backbone(x.to(device)).float().cpu())
            for h, v in y.items():
                ys.setdefault(h, []).append(torch.as_tensor(v))
    return torch.cat(fs), {h: torch.cat(v) for h, v in ys.items()}


def build(run_dir, out_dir, device="cpu", keep=0.95, reg=1e-3, max_train=None, batch=64):
    run_dir, out_dir = pathlib.Path(run_dir).resolve(), pathlib.Path(out_dir)
    if (run_dir / "ensemble.json").exists():
        raise ValueError(f"{run_dir} is an ensemble run; the check needs one model with a backbone")
    model, meta = props.load(run_dir, device)
    if not hasattr(model, "backbone"):
        raise ValueError(f"{run_dir} has no backbone")
    cfg = json.loads((run_dir / "config.json").read_text())
    heads = list(meta["heads"])
    kw = {"imgsz": meta["imgsz"], "margin": meta["margin"], "min_px": cfg["min_px"], "input": meta["input"], "need": heads}
    sets = {s: props.CropDataset(cfg["export"], s, meta["crop"], jitter=None, **kw) for s in ("train", "val", "test")}
    if max_train is not None and len(sets["train"]) > max_train:
        sets["train"] = Subset(sets["train"], np.sort(np.random.default_rng(0).permutation(len(sets["train"]))[:max_train]).tolist())
    feats = {s: features(model, DataLoader(d, batch_size=batch, shuffle=False), device) for s, d in sets.items() if len(d)}
    if "train" not in feats or "val" not in feats:
        raise ValueError("need non-empty train and val splits")
    groups, combos = group_ids({h: feats["train"][1][h].numpy() for h in heads}, heads)
    stats = fit(feats["train"][0].numpy(), groups, reg)
    thr = threshold(score(stats, feats["val"][0].numpy()), keep)
    test = {}
    if "test" in feats:
        f, lab = feats["test"]
        held = np.array(sets["test"].heldout, bool)
        s = score(stats, f.numpy())
        flag = s > thr
        test["test_auroc"] = auroc(s, held)
        for part, grp in (("seen", ~held), ("unseen", held)):
            test[f"test_flag_{part}"] = float(flag[grp].mean()) if grp.any() else float("nan")
        temps = meta.get("temperature") or {}
        for h in heads:
            with torch.no_grad():
                conf, pred = torch.softmax(model.heads[h](f.to(device)).float().cpu() / temps.get(h, 1.0), 1).max(1)
            hit = (pred == lab[h]).numpy()
            sure = conf.numpy() >= 0.9
            for part, grp in (("seen", ~held), ("unseen", held)):
                for tag, ok in (("", sure), ("gated_", sure & ~flag)):
                    m = ok & grp
                    test[f"test_{tag}sure_share_{part}_{h}"] = float(m.sum() / grp.sum()) if grp.any() else float("nan")
                    test[f"test_{tag}sure_acc_{part}_{h}"] = float(hit[m].mean()) if m.any() else float("nan")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"run": str(run_dir), "method": "rmd", "keep": keep, "reg": reg, "threshold": thr,
              "combos": [list(c) for c in combos], "test": test}
    (out_dir / "familiar.json").write_text(json.dumps(result, indent=2))
    np.savez(out_dir / "familiar.npz", **stats)
    return result


class Checker:
    def __init__(self, stats, thr):
        self.stats, self.threshold = stats, thr

    def score(self, feats):
        return score(self.stats, feats)

    def flag(self, feats):
        return self.score(feats) > self.threshold


def load(out_dir):
    out_dir = pathlib.Path(out_dir)
    with np.load(out_dir / "familiar.npz") as z:
        stats = {k: z[k] for k in z.files}
    return Checker(stats, json.loads((out_dir / "familiar.json").read_text())["threshold"])
