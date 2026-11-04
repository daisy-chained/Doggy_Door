"""Record scores in MLflow (Madelief 2026-11-04, "good for record keeping"): scores computed after training go on the
training run they belong to (log_to_run); training-free results get their own run, replaced when re-recorded (log_standalone)."""
import re

TRAIN_EXPERIMENT = "doggy-door"
_BAD = re.compile(r"[^A-Za-z0-9_./ -]")


def default_uri():
    from . import config
    return config.DEFAULTS["mlflow"]["tracking_uri"]


def clean(name):
    return _BAD.sub("_", str(name))


def _client(uri):
    import mlflow
    from mlflow.tracking import MlflowClient
    uri = uri or default_uri()
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def _finite(metrics):
    out = {}
    for k, v in metrics.items():
        if isinstance(v, bool):
            v = float(v)
        if isinstance(v, (int, float)) and v == v and abs(v) != float("inf"):
            out[clean(k)] = float(v)
    return out


def _log(c, run_id, metrics, artifacts):
    from mlflow.entities import Metric
    import time
    ts = int(time.time() * 1000)
    ms = [Metric(k, v, ts, 0) for k, v in _finite(metrics).items()]
    for i in range(0, len(ms), 900):
        c.log_batch(run_id, metrics=ms[i:i + 900])
    for a in artifacts:
        c.log_artifact(run_id, str(a))


def log_to_run(run_name, metrics, tracking_uri=None, artifacts=()):
    """Add metrics to the LATEST run (start_time) named run_name (experiment doggy-door first, then the others)."""
    c = _client(tracking_uri)
    exps = [e for e in c.search_experiments() if e.name == TRAIN_EXPERIMENT] + \
           [e for e in c.search_experiments() if e.name != TRAIN_EXPERIMENT]
    for e in exps:
        found = c.search_runs([e.experiment_id], filter_string=f"attributes.run_name = '{run_name}'",
                              order_by=["attributes.start_time DESC"], max_results=1)
        if found:
            _log(c, found[0].info.run_id, metrics, artifacts)
            return found[0].info.run_id
    raise LookupError(f"no MLflow run named {run_name!r}")


def log_standalone(experiment, name, params, metrics, key, tracking_uri=None, artifacts=(), source="doordet.record"):
    """Own run in `experiment`; an earlier run made here with the same record_key is deleted first (replace, no duplicate)."""
    c = _client(tracking_uri)
    exp = c.get_experiment_by_name(experiment)
    eid = exp.experiment_id if exp else c.create_experiment(experiment)
    for r in c.search_runs([eid], filter_string=f"tags.record_key = '{key}'"):
        c.delete_run(r.info.run_id)
    run = c.create_run(eid, tags={"mlflow.runName": name, "record_key": key, "source": source})
    rid = run.info.run_id
    for k, v in params.items():
        c.log_param(rid, clean(k), str(v))
    _log(c, rid, metrics, artifacts)
    c.set_terminated(rid, "FINISHED")
    return rid
