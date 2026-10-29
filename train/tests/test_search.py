# Unit tests for the auto trainer (train/doordet/search.py, speed.py, loop hooks). No GPU, no downloads:
#   train/.venv/bin/python -m pytest train/tests
import json, pathlib, sys
import pytest
import torch
import yaml

TRAIN = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRAIN))
sys.path.insert(0, str(TRAIN / "tests"))
import optuna
from doordet import adapters, config, loop, search, speed
from test_train import FakeAdapter, make_export


@pytest.fixture
def export_dir(tmp_path):
    return make_export(tmp_path / "export")


def fake_cfg(tmp_path, export_dir, **kw):
    d = {"export": str(export_dir), "model": {"family": "fake"}, "epochs": 2, "batch": 2, "device": "cpu", "workers": 0,
         "out": str(tmp_path / "runs"), "mlflow": {"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}", "experiment": "test"}}
    d.update(kw)
    return config.resolve(d)


# --- loop hooks the search needs ---
def test_loop_new_config_defaults():
    d = config.DEFAULTS
    assert d["test_eval"] is True and d["select_metric"] == "map" and d["mlflow"]["tags"] == {}


def test_loop_on_eval_hook_and_no_test_eval(tmp_path, export_dir, monkeypatch):
    import mlflow
    monkeypatch.setitem(adapters.FAMILIES, "fake", FakeAdapter)
    seen = []
    cfg = fake_cfg(tmp_path, export_dir, test_eval=False, mlflow={"tags": {"budget": "any"}})
    r = loop.train(cfg, run_name="hook", on_eval=lambda epoch, m: seen.append((epoch, m["map"], m["ap_handle"])))
    assert [e for e, _, _ in seen] == [0, 1]
    run = mlflow.get_run(r["run_id"])
    assert not any(k.startswith("test_") for k in run.data.metrics)  # test split left alone during a search
    assert run.data.tags["budget"] == "any"
    assert r["test"] is None


def test_loop_on_eval_can_stop_training(tmp_path, export_dir, monkeypatch):
    monkeypatch.setitem(adapters.FAMILIES, "fake", FakeAdapter)
    class Stop(Exception): pass
    def stop(epoch, m):
        raise Stop
    with pytest.raises(Stop):
        loop.train(fake_cfg(tmp_path, export_dir, epochs=5), run_name="stop", on_eval=stop)


def test_loop_select_metric(tmp_path, export_dir, monkeypatch):
    monkeypatch.setitem(adapters.FAMILIES, "fake", FakeAdapter)
    r = loop.train(fake_cfg(tmp_path, export_dir, select_metric="ap_handle"), run_name="sel")
    assert r["best_value"] == pytest.approx(1.0) and r["best_metric"] == "ap_handle"
    with pytest.raises(ValueError, match="nope"):
        config.resolve({"export": str(export_dir), "select_metric": "nope"})


def test_loop_epochs_zero_scores_test_only(tmp_path, export_dir, monkeypatch):
    # used by the search for the one test score of the winning trial (weights loaded from model.checkpoint)
    monkeypatch.setitem(adapters.FAMILIES, "fake", FakeAdapter)
    r = loop.train(fake_cfg(tmp_path, export_dir, epochs=0), run_name="test_only")
    assert r["test"]["map"] == pytest.approx(1.0)


# --- speed + compute per image ---
def test_speed_measure_counts_flops():
    model = torch.nn.Conv2d(3, 8, 3, padding=1, bias=False)
    r = speed.measure(model, {"input": torch.zeros(1, 3, 32, 32)}, torch.device("cpu"), warmup=1, n=3)
    assert r["ms"] > 0
    assert r["gflops"] == pytest.approx(2 * 8 * 3 * 9 * 32 * 32 / 1e9)  # 2 flops per multiply-add, batch 1


def test_fake_and_hf_adapters_have_example_inputs():
    assert hasattr(adapters.FAMILIES["hf"], "example_inputs")


# --- search config ---
SPACE = {"models": [{"family": "fake", "checkpoint": "fake-a"}, {"family": "fake", "checkpoint": "fake-b"}],
         "imgsz": {"low": 320, "high": 448, "step": 32}, "lr": {"low": 1e-5, "high": 1e-3, "log": True},
         "weight_decay": {"low": 1e-6, "high": 1e-3, "log": True}, "warmup_steps": {"low": 0, "high": 50},
         "batch": [2, 4]}


def table(entries):
    return {"device": "cpu", "entries": [{"family": "fake", "checkpoint": c, "imgsz": s, "ms": ms, "gflops": ms * 2}
                                         for c, s, ms in entries]}


def write_search(tmp_path, base, **kw):
    d = {"name": "unit", "base": str(base), "speed_table": str(tmp_path / "speed.json"),
         "budgets": {"fast": 10, "any": None}, "space": SPACE, "storage": f"sqlite:///{tmp_path / 'optuna.db'}"}
    d.update(kw)
    p = tmp_path / "search.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def test_search_config_defaults_and_checks(tmp_path):
    base = tmp_path / "base.yaml"; base.write_text("export: x\n")
    s = search.load(write_search(tmp_path, base))
    assert s["objective"] == "map"  # Madelief: overall mAP by default ...
    assert s["budget_metric"] == "gflops"  # Madelief 2026-10-29: budgets in GFLOPs (4060 ms too noisy at batch 1), ms still logged
    assert search.load(write_search(tmp_path, base, budget_metric="ms"))["budget_metric"] == "ms"
    with pytest.raises(ValueError, match="budget_metric"):
        search.load(write_search(tmp_path, base, budget_metric="fps"))
    assert s["pruner"] == {"type": "median", "grace_epochs": 5}  # grace epochs always on, as a default
    assert s["stop"]["patience"] > 0 and s["stop"]["max_hours"] > 0
    assert s["keep_checkpoints"] == 3
    assert search.load(write_search(tmp_path, base, objective="ap_handle"))["objective"] == "ap_handle"  # ... or one AP
    with pytest.raises(ValueError, match="objective"):
        search.load(write_search(tmp_path, base, objective="accuracy"))
    with pytest.raises(ValueError, match="bogus"):
        search.load(write_search(tmp_path, base, bogus=1))


# --- hardware budget -> allowed model x resolution ---
def test_feasible_filters_by_measured_speed():
    sizes = [320, 352, 384, 416, 448]
    t = table([("fake-a", s, 2 + i) for i, s in enumerate(sizes)] + [("fake-b", s, 6 + i) for i, s in enumerate(sizes)])
    assert search.feasible(t, SPACE, 5, "ms") == {"fake-a": [320, 352, 384, 416]}  # fake-b too slow at every size: left out
    assert search.feasible(t, SPACE, 10, "gflops") == {"fake-a": [320, 352, 384, 416]}  # table(): gflops = 2 x ms
    assert search.feasible(t, SPACE, None, "gflops") == {"fake-a": sizes, "fake-b": sizes}
    t2 = table([("fake-a", 320, 1)])  # not measured = not allowed
    assert search.feasible(t2, SPACE, None, "gflops") == {"fake-a": [320]}


def test_suggest_stays_in_space_and_budget():
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=0))
    allowed = {"fake-a": [320, 352, 384], "fake-b": [320]}
    for _ in range(40):
        trial = study.ask()
        o = search.suggest(trial, SPACE, allowed)
        assert set(o) == {"model.family", "model.checkpoint", "imgsz", "lr", "weight_decay", "warmup_steps", "batch"}
        assert o["imgsz"] in allowed[o["model.checkpoint"]]
        assert 1e-5 <= o["lr"] <= 1e-3 and 1e-6 <= o["weight_decay"] <= 1e-3 and 0 <= o["warmup_steps"] <= 50
        assert o["batch"] in (2, 4)
        study.tell(trial, 0.5)


# --- stop rule: until it stops improving, + max time ---
def test_patience_stop():
    values = iter([0.5, 0.6, 0.6, 0.603, 0.601, 0.6, 0.9, 0.9])
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: next(values), n_trials=8,
                   callbacks=[search.PatienceStop(patience=3, min_trials=2, min_delta=0.005)])
    # best 0.6 at trial 1; 0.6, 0.603, 0.601 are within min_delta -> 3 trials without improvement -> stop after trial 4
    assert len(study.trials) == 5


def test_patience_waits_for_min_trials():
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: 0.5, n_trials=10, callbacks=[search.PatienceStop(patience=1, min_trials=6, min_delta=0.0)])
    assert len(study.trials) == 6


# --- checkpoints: keep only the best few per search ---
def test_keep_best_checkpoints(tmp_path):
    study = optuna.create_study(direction="maximize")
    for i, v in enumerate([0.3, 0.9, 0.5, 0.7]):
        study.add_trial(optuna.trial.create_trial(value=v, user_attrs={"out": str(tmp_path / f"t{i}")}))
        (tmp_path / f"t{i}" / "best").mkdir(parents=True)
    pruned = optuna.trial.create_trial(state=optuna.trial.TrialState.PRUNED, user_attrs={"out": str(tmp_path / "tp")})
    study.add_trial(pruned); (tmp_path / "tp" / "best").mkdir(parents=True)
    search.keep_best(study, 2)
    left = sorted(p.parent.name for p in tmp_path.glob("*/best"))
    assert left == ["t1", "t3"]


# --- end to end with the fake model ---
def test_run_search_end_to_end(tmp_path, export_dir, monkeypatch):
    import mlflow
    monkeypatch.setitem(adapters.FAMILIES, "fake", FakeAdapter)
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"export": str(export_dir), "model": {"family": "fake"}, "epochs": 2, "device": "cpu",
                                    "workers": 0, "out": str(tmp_path / "runs"),
                                    "mlflow": {"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}", "experiment": "test"}}))
    (tmp_path / "speed.json").write_text(json.dumps(table([("fake-a", s, 3) for s in (320, 352, 384, 416, 448)]
                                                         + [("fake-b", s, 9) for s in (320, 352, 384, 416, 448)])))
    s = search.load(write_search(tmp_path, base, stop={"patience": 2, "min_trials": 3, "min_delta": 0.0, "max_hours": 1},
                                 pruner={"type": "median", "grace_epochs": 1}, keep_checkpoints=1))
    r = search.run(s, "fast")
    # the fake model is always perfect: no improvement after the first trial -> stops at min_trials
    assert r["trials"] == 3 and r["best_value"] == pytest.approx(1.0)
    assert r["study"] == "unit_fast"
    assert r["test"]["map"] == pytest.approx(1.0)  # the winner's test score, once
    runs = mlflow.search_runs(experiment_names=["test"], output_format="list")
    trial_runs = [x for x in runs if x.data.tags.get("study") == "unit_fast" and "trial" in x.data.tags]
    assert len(trial_runs) == 3
    for x in trial_runs:
        assert x.data.params["model.checkpoint"] == "fake-a"  # fake-b (18 GFLOPs) is over the 10 GFLOPs budget
        assert x.data.tags["budget"] == "fast" and float(x.data.tags["speed_ms"]) == 3 and "gflops" in x.data.tags
    assert len(list((tmp_path / "runs").rglob("best"))) == 1  # keep_checkpoints=1
