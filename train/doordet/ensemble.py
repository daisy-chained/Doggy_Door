"""Deep ensemble of props classifiers: members' softmax averaged, then one temperature per head fitted on val (pool, then calibrate)."""
import json, pathlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import props


def pool(logits_list):
    """log of the mean member probability: softmax(pool(...)) is the mean softmax, softmax(pool(...) / T) the calibrated ensemble."""
    return torch.log(torch.stack([torch.softmax(l, 1) for l in logits_list]).mean(0).clamp_min(1e-12))


class Ensemble(nn.Module):
    """Drop-in for props.Model: forward(x) -> {head: pooled log-probabilities}."""
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outs = [m(x) for m in self.models]
        return {h: pool([o[h] for o in outs]) for h in outs[0]}


def build(member_dirs, out_dir, device="cpu"):
    member_dirs = [pathlib.Path(m).resolve() for m in member_dirs]
    out_dir = pathlib.Path(out_dir)
    loaded = [props.load(m, device) for m in member_dirs]
    metas = [meta for _, meta in loaded]
    for d, meta in zip(member_dirs[1:], metas[1:]):
        for k in ("heads", "backbone", "imgsz", "crop", "margin", "input"):
            if meta[k] != metas[0][k]:
                raise ValueError(f"member {d} differs from {member_dirs[0]} in {k}: {meta[k]!r} vs {metas[0][k]!r}")
    meta0, models = metas[0], [m for m, _ in loaded]
    first_dir = member_dirs[0]
    cfg = json.loads((first_dir / "config.json").read_text())
    heads = list(meta0["heads"])
    kw = {"imgsz": meta0["imgsz"], "margin": meta0["margin"], "min_px": cfg["min_px"], "input": meta0["input"], "need": heads}
    sets = {s: props.CropDataset(cfg["export"], s, meta0["crop"], jitter=None, **kw) for s in ("val", "test")}
    loader = lambda s: DataLoader(sets[s], batch_size=cfg["batch"], shuffle=False, num_workers=cfg["workers"])

    def pooled(s):
        per = [props.predict(m, loader(s), device, heads) for m in models]
        return {h: (pool([p[h][0] for p in per]), per[0][h][1]) for h in heads}

    temps = {h: 1.0 for h in heads}
    if len(sets["val"]):
        for h, (lg, lb) in pooled("val").items():
            if len(lb) >= cfg["min_calibration"] and (lg.argmax(1) != lb).any():
                temps[h] = props.fit_temperature(lg, lb)
    test = {}
    if len(sets["test"]):
        held = torch.tensor(sets["test"].heldout, dtype=torch.bool)
        for h, (lg, lb) in pooled("test").items():
            conf, pred = torch.softmax(lg / temps[h], 1).max(1)
            hit = (pred == lb).float()
            sure = conf >= 0.9
            test[f"test_ece_{h}"] = props.ece(conf.numpy(), hit.numpy())
            test[f"test_acc_{h}"] = hit.mean().item()
            test[f"test_sure_share_{h}"] = sure.float().mean().item()
            test[f"test_sure_acc_{h}"] = hit[sure].mean().item() if sure.any() else float("nan")
            if held.any():
                for tag, grp in (("seen", ~held), ("unseen", held)):
                    if grp.any():
                        test[f"test_ece_{tag}_{h}"] = props.ece(conf[grp].numpy(), hit[grp].numpy())
    members = [str(d) for d in member_dirs]
    result = {"members": members, "temperature": temps, "test": test}
    (out_dir / "best").mkdir(parents=True, exist_ok=True)
    (out_dir / "ensemble.json").write_text(json.dumps(result, indent=2))
    heads_json = json.loads((first_dir / "best" / "heads.json").read_text())
    heads_json["temperature"] = temps
    (out_dir / "best" / "heads.json").write_text(json.dumps(heads_json, indent=2))
    (out_dir / "config.json").write_text(json.dumps({**cfg, "ensemble": members}, indent=2))
    return result
