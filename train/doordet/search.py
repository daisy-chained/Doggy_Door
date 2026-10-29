"""Auto trainer: per hardware budget (GFLOPs or ms per image), an Optuna search over models x resolution x hyperparameters."""
import copy, json, pathlib
import optuna
import torch
import yaml

from . import config, loop

DEFAULTS = {
    "name": None, "base": None, "speed_table": None, "budgets": None, "space": None,  # required
    "objective": "map", "overrides": {}, "budget_metric": "gflops",  # what budgets limit: gflops or ms (per image)
    "pruner": {"type": "median", "grace_epochs": 5},
    "stop": {"patience": 10, "min_trials": 15, "min_delta": 0.005, "max_hours": 24},
    "keep_checkpoints": 3, "seed": 0, "storage": "sqlite:///optuna.db",
}
NESTED = ("pruner", "stop")


def _path(p):
    p = pathlib.Path(p)
    return str(p if p.is_absolute() else config.REPO / p)


def load(path, overrides=None):
    d = yaml.safe_load(pathlib.Path(path).read_text()) or {}
    for o in overrides or []:
        key, _, val = o.partition("=")
        node = d
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(val)
    unknown = sorted(set(d) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown search key(s): {', '.join(unknown)}")
    missing = [k for k in ("name", "base", "speed_table", "budgets", "space") if not d.get(k)]
    if missing:
        raise ValueError(f"missing search key(s): {', '.join(missing)}")
    s = {k: (dict(v) if k in NESTED else v) for k, v in DEFAULTS.items()}
    for k, v in d.items():
        if k in NESTED:
            bad = sorted(set(v) - set(DEFAULTS[k]))
            if bad:
                raise ValueError(f"unknown search key(s) in {k}: {', '.join(bad)}")
            s[k].update(v)
        else:
            s[k] = v
    if s["objective"] not in config.METRICS:
        raise ValueError(f"unknown objective {s['objective']!r}; available: {', '.join(config.METRICS)}")
    if s["budget_metric"] not in ("gflops", "ms"):
        raise ValueError(f"unknown budget_metric {s['budget_metric']!r}; available: gflops, ms")
    if s["pruner"]["type"] not in ("median", "hyperband"):
        raise ValueError(f"unknown pruner type {s['pruner']['type']!r}; available: median, hyperband")
    if s["keep_checkpoints"] < 1:
        raise ValueError("keep_checkpoints must be at least 1 (the winner's weights are needed for its test score)")
    s["base"], s["speed_table"] = _path(s["base"]), _path(s["speed_table"])
    if s["storage"].startswith("sqlite:///") and not s["storage"].startswith("sqlite:////"):
        s["storage"] = "sqlite:///" + _path(s["storage"][len("sqlite:///"):])
    return s


def imgsizes(space):
    i = space["imgsz"]
    return range(i["low"], i["high"] + 1, i["step"])


def feasible(table, space, budget, metric="gflops"):
    """{checkpoint: sorted allowed imgsz}: measured, and (if the budget is set) metric (gflops or ms) <= budget."""
    cost = {(e["family"], e["checkpoint"], e["imgsz"]): e[metric] for e in table["entries"] if e.get("ms") is not None}
    out = {}
    for m in space["models"]:
        ok = [z for z in imgsizes(space) if (m["family"], m["checkpoint"], z) in cost
              and (budget is None or cost[(m["family"], m["checkpoint"], z)] <= budget)]
        if ok:
            out[m["checkpoint"]] = sorted(ok)
    return out


def suggest(trial, space, allowed):
    ckpt = trial.suggest_categorical("model.checkpoint", list(allowed))
    family = next(m["family"] for m in space["models"] if m["checkpoint"] == ckpt)
    flt = lambda name: trial.suggest_float(name, space[name]["low"], space[name]["high"], log=space[name].get("log", False))
    w = space["warmup_steps"]
    return {"model.family": family, "model.checkpoint": ckpt,
            "imgsz": trial.suggest_categorical(f"imgsz|{ckpt}", allowed[ckpt]),
            "lr": flt("lr"), "weight_decay": flt("weight_decay"),
            "warmup_steps": trial.suggest_int("warmup_steps", w["low"], w["high"]),
            "batch": trial.suggest_categorical("batch", space["batch"])}


class PatienceStop:
    """Study callback: stop after `patience` finished trials without a gain > min_delta (never before min_trials)."""
    def __init__(self, patience, min_trials, min_delta):
        self.patience, self.min_trials, self.min_delta = patience, min_trials, min_delta

    def __call__(self, study, trial):
        done = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
        best, improved = None, []
        for t in study.get_trials(deepcopy=False, states=done):
            gain = t.state == done[0] and (best is None or t.value > best + self.min_delta)
            if gain:
                best = t.value
            improved.append(gain)
        if len(improved) >= self.min_trials and not any(improved[-self.patience:]):
            study.stop()


def keep_best(study, k):
    done = [t for t in study.get_trials(deepcopy=False) if t.user_attrs.get("out")]
    complete = sorted((t for t in done if t.state == optuna.trial.TrialState.COMPLETE), key=lambda t: -t.value)
    keep = {t.number for t in complete[:k]}
    for t in done:
        if t.number not in keep:
            _rmtree(pathlib.Path(t.user_attrs["out"]) / "best")


def _rmtree(p):
    import shutil
    if p.is_dir():
        shutil.rmtree(p)


def _set(cfg, key, val):
    *parents, last = key.split(".")
    for p in parents:
        cfg = cfg.setdefault(p, {})
    cfg[last] = val


def run(s, budget_name, train_fn=None):
    train_fn = train_fn or loop.train
    study_name = f"{s['name']}_{budget_name}"
    p = s["pruner"]
    pruner = (optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=p["grace_epochs"]) if p["type"] == "median" else None)
    base = config.load(s["base"])
    if pruner is None:
        pruner = optuna.pruners.HyperbandPruner(min_resource=max(p["grace_epochs"], 1), max_resource=max(base["epochs"], 1))
    table = json.loads(pathlib.Path(s["speed_table"]).read_text())
    allowed = feasible(table, s["space"], s["budgets"][budget_name], s["budget_metric"])
    if not allowed:
        raise ValueError(f"no model fits budget {budget_name!r} ({s['budgets'][budget_name]} {s['budget_metric']}) in {s['speed_table']}")
    for k, v in s["overrides"].items():
        _set(base, k, v)
    base = config.resolve(base)
    entry = {(e["checkpoint"], e["imgsz"]): e for e in table["entries"]}
    out_dir = pathlib.Path(base["out"]) / "search" / study_name
    study = optuna.create_study(direction="maximize", study_name=study_name, storage=s["storage"], load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=s["seed"]), pruner=pruner)

    def objective(trial):
        o = suggest(trial, s["space"], allowed)
        e = entry[(o["model.checkpoint"], o["imgsz"])]
        name = f"{study_name}_t{trial.number:03d}"
        cfg = copy.deepcopy(base)
        for k, v in o.items():
            _set(cfg, k, v)
        cfg.update(test_eval=False, select_metric=s["objective"], name=name, out=str(out_dir))
        cfg["mlflow"]["tags"] = {k: str(v) for k, v in {
            "study": study_name, "trial": trial.number, "budget": budget_name, "speed_ms": e["ms"], "gflops": e["gflops"],
            "objective": s["objective"]}.items()}
        cfg = config.resolve(cfg)
        trial.set_user_attr("out", str(out_dir / name))
        trial.set_user_attr("speed_ms", e["ms"])
        trial.set_user_attr("gflops", e["gflops"])

        def on_eval(epoch, m):
            trial.report(m[s["objective"]], epoch)
            if trial.should_prune():
                import mlflow
                mlflow.set_tag("optuna_state", "pruned")
                raise optuna.TrialPruned()

        return train_fn(cfg, run_name=name, on_eval=on_eval)["best_value"]

    def free_cuda(study, trial):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    st = s["stop"]
    study.optimize(objective, timeout=st["max_hours"] * 3600, callbacks=[PatienceStop(st["patience"], st["min_trials"], st["min_delta"]), free_cuda],
                   catch=(torch.cuda.OutOfMemoryError,), gc_after_trial=True)
    keep_best(study, s["keep_checkpoints"])

    best = study.best_trial
    cfg = json.loads((pathlib.Path(best.user_attrs["out"]) / "config.json").read_text())  # the trial's own config
    cfg.update(epochs=0, test_eval=True, name=f"{study_name}_test", out=str(out_dir))
    cfg["model"]["checkpoint"] = str(pathlib.Path(best.user_attrs["out"]) / "best")
    cfg["mlflow"]["tags"] = {"study": study_name, "budget": budget_name, "final_test": "true", "best_trial": str(best.number)}
    test = train_fn(config.resolve(cfg), run_name=cfg["name"])["test"]
    print(f"{study_name}: trial {best.number} {s['objective']} {best.value:.4f} test_map {test['map']:.4f}", flush=True)
    return {"study": study_name, "trials": len(study.trials), "best_value": best.value, "best_trial": best.number,
            "best_params": best.params, "test": test}
