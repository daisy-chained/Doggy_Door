# Unit tests for train/doordet/record.py + train/record.py (Madelief 2026-11-04, "good for record keeping"): scores computed
# after training go into MLflow, on the training run they belong to; training-free results get their own runs; re-recording
# the same result replaces it instead of adding a duplicate. train/.venv/bin/python -m pytest train/tests
import json, pathlib, sys
import pytest

TRAIN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))
mlflow = pytest.importorskip("mlflow")
import record as record_cli
from doordet import record


@pytest.fixture
def uri(tmp_path):
    u = f"sqlite:///{tmp_path / 'ml.db'}"
    mlflow.set_tracking_uri(u)
    mlflow.set_experiment("doggy-door")
    with mlflow.start_run(run_name="det_a"):
        mlflow.log_metric("test_map", 0.1)
    with mlflow.start_run(run_name="det_a"):  # a rerun with the same name: the latest one is the one to attach to
        mlflow.log_metric("test_map", 0.4)
    return u


def runs(uri, exp, name=None):
    mlflow.set_tracking_uri(uri)
    r = mlflow.search_runs(experiment_names=[exp], output_format="list")
    return [x for x in r if name is None or x.info.run_name == name]


def test_log_to_latest_training_run(uri):
    record.log_to_run("det_a", {"vis16_map": 0.5}, tracking_uri=uri)
    rs = sorted(runs(uri, "doggy-door", "det_a"), key=lambda r: r.info.start_time)
    assert "vis16_map" not in rs[0].data.metrics and rs[1].data.metrics["vis16_map"] == 0.5


def test_unknown_training_run_raises(uri):
    with pytest.raises(LookupError):
        record.log_to_run("nope", {"x": 1.0}, tracking_uri=uri)


def test_standalone_runs_replace_instead_of_duplicate(uri):
    for v in (0.3, 0.6):
        record.log_standalone("doggy-door-radar", "imaging@10 grid", {"model": "imaging", "views": 10}, {"found_open_in_view": v},
                              key="radar:imaging:10:grid:test", tracking_uri=uri)
    rs = runs(uri, "doggy-door-radar")
    assert len(rs) == 1 and rs[0].data.metrics["found_open_in_view"] == 0.6 and rs[0].data.params["views"] == "10"


def test_cli_levels(uri, tmp_path):
    f = tmp_path / "visibility.jsonl"
    f.write_text(json.dumps({"run": "det_a", "export": "exports/v4_test_vis16_dw", "levels": {"visibility_m": 16.0},
                             "metrics": {"map": 0.52, "ap_door": 0.57}}) + "\n")
    record_cli.main(["levels", str(f), "--tracking-uri", uri])
    m = sorted(runs(uri, "doggy-door", "det_a"), key=lambda r: r.info.start_time)[-1].data.metrics
    assert m["v4_test_vis16_dw.map"] == 0.52 and m["v4_test_vis16_dw.ap_door"] == 0.57


def test_cli_radar_rows_summarised_per_model_views_finder(uri, tmp_path):
    rows = [{"frame": "1", "model": "imaging", "views": 3, "finder": "grid", "door": "a", "found": True, "center_m": 0.05,
             "state": "open", "in_view": True},
            {"frame": "1", "model": "imaging", "views": 3, "finder": "grid", "door": "b", "found": False, "center_m": None,
             "state": "open", "in_view": True},
            {"frame": "1", "model": "imaging", "views": 3, "finder": "grid", "door": None, "false_positive": True, "found": True}]
    f = tmp_path / "v4_test_views_grid_tuned.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    record_cli.main(["radar", str(f), "--tracking-uri", uri])
    rs = runs(uri, "doggy-door-radar")
    assert len(rs) == 1
    m = rs[0].data.metrics
    assert m["recall"] == pytest.approx(0.5) and m["precision"] == pytest.approx(0.5) and m["centre_median_m"] == pytest.approx(0.05)
    assert rs[0].data.params["source"].endswith("v4_test_views_grid_tuned.jsonl")
