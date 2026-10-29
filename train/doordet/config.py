"""Run config: YAML file + key=value overrides, validated against DEFAULTS."""
import pathlib
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULTS = {
    "export": None,  # required: exporter output folder
    "model": {"family": "hf", "checkpoint": None},
    "imgsz": 640,  # images are resized to imgsz x imgsz
    "epochs": 30, "batch": 8, "lr": 1e-4, "weight_decay": 1e-4, "warmup_steps": 100, "grad_clip": 1.0,
    "amp": True, "workers": 4, "seed": 0, "device": None,  # None -> cuda if available
    "out": str(REPO / "runs"), "name": None,
    "mlflow": {"tracking_uri": (REPO / "mlruns").as_uri(), "experiment": "doggy-door"},
    "eval_every": 1, "score_threshold": 0.01,
}
NESTED = ("model", "mlflow")


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
    if cfg["device"] is None:
        import torch
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def load(path, overrides=None):
    d = yaml.safe_load(pathlib.Path(path).read_text()) or {}
    for o in overrides or []:
        key, _, val = o.partition("=")
        node = d
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(val)
    return resolve(d)
