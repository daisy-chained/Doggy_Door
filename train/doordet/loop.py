"""Training loop: AdamW + warmup/cosine, AMP, val mAP per epoch, best checkpoint, one test eval, MLflow logging."""
import contextlib, datetime, json, math, os, pathlib, random, subprocess, time
import numpy as np
import torch
from torch.utils.data import DataLoader

from . import adapters, evaluate
from .data import CocoDetection

REPO = pathlib.Path(__file__).resolve().parents[2]


def flatten(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def git_info():
    try:
        run = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()
        return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--porcelain"))}
    except Exception:
        return {}


def autocast(cfg, device):
    if not (cfg["amp"] and device.type == "cuda"):
        return contextlib.nullcontext(), None
    bf16 = torch.cuda.is_bf16_supported()
    return torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float16), (None if bf16 else torch.amp.GradScaler())


def predict_all(adapter, loader, device, cfg):
    adapter.model.eval()
    preds, t0, n = {}, time.time(), 0
    ctx, _ = autocast(cfg, device)
    with torch.no_grad(), ctx:
        for inputs in loader:
            preds.update(adapter.predict(inputs, device))
            n += len(inputs.get("image_ids", inputs.get("sizes", [])))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return preds, n / max(time.time() - t0, 1e-9)


def train(cfg, run_name=None, on_eval=None):
    import mlflow
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device(cfg["device"])
    export = pathlib.Path(cfg["export"])
    run_name = run_name or cfg["name"] or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = pathlib.Path(cfg["out"]) / run_name
    out.mkdir(parents=True, exist_ok=True)

    sets = {s: CocoDetection(export, s) for s in ("train", "val", "test")}
    adapter = adapters.get(cfg["model"]["family"])(cfg, sets["train"].id2label)
    adapter.model.to(device)
    gen = torch.Generator().manual_seed(seed)
    mk = lambda s, shuffle: DataLoader(sets[s], batch_size=cfg["batch"], shuffle=shuffle, num_workers=cfg["workers"],
                                       collate_fn=adapter.collate, generator=gen if shuffle else None)
    loaders = {"train": mk("train", True), "val": mk("val", False), "test": mk("test", False)}

    opt = torch.optim.AdamW([p for p in adapter.model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = max(cfg["epochs"] * len(loaders["train"]), 1)
    warm = cfg["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    ctx, scaler = autocast(cfg, device)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    export_info = json.loads((export / "export.json").read_text()) if (export / "export.json").exists() else {}
    src = export_info.get("source") or {}
    sel = cfg["select_metric"]
    best = {"value": -1.0, "map": -1.0, "epoch": None, "state": None}
    with mlflow.start_run(run_name=run_name) as run:
        params = dict(flatten(cfg))
        params.update({"export.git_commit": (src.get("git") or {}).get("commit"), "export.seed": export_info.get("seed"),
                       "export.source": src.get("path"), **git_info()})
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.set_tags({k: str(v) for k, v in cfg["mlflow"]["tags"].items()})
        if (export / "export.json").exists():
            mlflow.log_artifact(str(export / "export.json"))
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
        mlflow.log_artifact(str(out / "config.json"))

        t0, speed = time.time(), 0.0
        for epoch in range(cfg["epochs"]):
            adapter.model.train()
            losses = []
            for inputs in loaders["train"]:
                with autocast(cfg, device)[0]:
                    loss = adapter.loss(inputs, device)
                opt.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward(); scaler.unscale_(opt)
                else:
                    loss.backward()
                if cfg["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), cfg["grad_clip"])
                if scaler:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                sched.step()
                losses.append(loss.item())
            mlflow.log_metric("train_loss", float(np.mean(losses)), step=epoch)
            if (epoch + 1) % cfg["eval_every"] == 0 or epoch == cfg["epochs"] - 1:
                preds, speed = predict_all(adapter, loaders["val"], device, cfg)
                m = evaluate.coco_metrics(sets["val"].gt_path, preds)
                mlflow.log_metrics({f"val_{k}": v for k, v in m.items()}, step=epoch)
                print(f"epoch {epoch} loss {np.mean(losses):.4f} val_map {m['map']:.4f}", flush=True)
                if m[sel] >= best["value"]:
                    best.update(value=m[sel], map=m["map"], epoch=epoch,
                                state={k: v.detach().cpu().clone() for k, v in adapter.model.state_dict().items()})
                    adapter.save(out / "best")
                if on_eval:
                    on_eval(epoch, m)
        train_seconds = time.time() - t0
        mlflow.log_metrics({"train_seconds": train_seconds, "imgs_per_sec_eval": speed})

        if best["state"] is not None:
            adapter.model.load_state_dict(best["state"])
        test = None
        if cfg["test_eval"]:
            preds, _ = predict_all(adapter, loaders["test"], device, cfg)
            test = evaluate.coco_metrics(sets["test"].gt_path, preds)
            mlflow.log_metrics({f"test_{k}": v for k, v in test.items()})
            print(f"test_map {test['map']:.4f}  train {train_seconds:.0f}s  eval {speed:.1f} img/s", flush=True)
        return {"run_id": run.info.run_id, "out": str(out), "best_epoch": best["epoch"], "best_metric": sel,
                "best_value": best["value"], "best_val_map": best["map"], "test": test}
