"""usage: train/.venv/bin/python train/record.py {levels,fusion,radar,radar-learned,backfill} FILES... [--tracking-uri URI]
Put scores into MLflow (see doordet/record.py)."""
import argparse, glob, json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "sim"))
from doordet import record

BINS = [(0, 1, "0-1"), (1, 2, "1-2"), (2, 3, "2-3"), (3, 4, "3-4"), (4, None, "4+")]


def base(p):
    return pathlib.Path(str(p).rstrip("/")).name


def jsonl(path):
    return [json.loads(l) for l in pathlib.Path(path).read_text().splitlines() if l.strip()]


def rec_levels(files, uri):
    by_run, n, skipped = {}, 0, []
    for f in files:
        for row in jsonl(f):
            by_run.setdefault(base(row["run"]), {}).update({f"{base(row['export'])}.{k}": v for k, v in row["metrics"].items()})
    for run, m in by_run.items():
        try:
            record.log_to_run(run, m, tracking_uri=uri)
            n += 1
        except LookupError:
            skipped.append(f"levels: no training run {run}")
    return {"doggy-door (levels)": n}, skipped


def rec_fusion(files, uri):
    n = 0
    for f in files:
        d = json.loads(pathlib.Path(f).read_text())
        runs = [base(r) for r in d["runs"]]
        name = "+".join(runs) + " @ " + base(d["export"])
        params = {"runs": ",".join(runs), "export": d["export"], "split": d.get("split"), "weights": d.get("weights"), "iou": d.get("iou")}
        m = {k: v for k, v in d["metrics"]["fused"].items()}
        labels = [k for k in d["metrics"] if k != "fused"]
        for i, k in enumerate(labels):
            label = runs[i] if k in ("A", "B") and i < len(runs) else k
            if "map" in d["metrics"][k]:
                m[f"{label}.map"] = d["metrics"][k]["map"]
        record.log_standalone("doggy-door-fusion", name, params, m, key=name, tracking_uri=uri, source="train/record.py fusion")
        n += 1
    return {"doggy-door-fusion": n}, []


def radar_summary(rows):
    from doorsim import radar_doorways as rd
    m = rd.detection_metrics(rows)
    gt = [r for r in rows if r.get("door") is not None]
    open_v = [r for r in gt if str(r.get("state")) == "open" and r.get("in_view", True)]
    err = [r["center_m"] for r in open_v if r["found"] and r.get("center_m") is not None]
    if err:
        m["centre_median_m"] = float(np.median(err))
        m["centre_p90_m"] = float(np.percentile(err, 90))
    for lo, hi, lab in BINS:
        sel = [r for r in open_v if r.get("distance_m") is not None and r["distance_m"] >= lo and (hi is None or r["distance_m"] < hi)]
        if sel:
            m[f"found_rate_{lab}m"] = sum(1 for r in sel if r["found"]) / len(sel)
    closed = [r for r in gt if str(r.get("state")) == "closed" and r.get("in_view", True)]
    if closed:
        m["closed_found_rate"] = sum(1 for r in closed if r["found"]) / len(closed)
    return m


def rec_radar(files, uri):
    n = 0
    for f in files:
        fn = pathlib.Path(f).name
        tuned = "tuned" in fn
        split = "val" if "val" in fn else "test"
        groups = {}
        for r in jsonl(f):
            groups.setdefault((r.get("model"), r.get("views", 1), r.get("finder", "line")), []).append(r)
        for (model, views, finder), rows in groups.items():
            name = f"{model}@{views} {finder}{' tuned' if tuned else ''}"
            params = {"model": model, "views": views, "finder": finder, "source": str(f), "tuned": tuned, "split": split}
            record.log_standalone("doggy-door-radar", name, params, radar_summary(rows), key=f"radar:{fn}:{model}:{views}:{finder}:{split}",
                                  tracking_uri=uri, source="train/record.py radar")
            n += 1
    return {"doggy-door-radar": n}, []


def rec_learned(files, uri):
    n, skipped = 0, []
    for f in files:
        d = json.loads(pathlib.Path(f).read_text())
        m = {k: v for k, v in d["metrics"].items() if isinstance(v, (int, float))}
        for k, v in d.get("coco", {}).items():
            if isinstance(v, (int, float)):
                m[f"coco_{k}"] = v
        m["threshold"] = d.get("threshold")
        m["images"] = d.get("images")
        m = {f"radar_test.{k}": v for k, v in m.items() if v is not None}
        try:
            record.log_to_run(base(d["run"]), m, tracking_uri=uri)
            n += 1
        except LookupError:
            skipped.append(f"radar-learned: no training run {base(d['run'])} ({f})")
    return {"doggy-door (radar-learned)": n}, skipped


def expand(paths, pattern="*.json"):
    out = []
    for p in paths:
        p = pathlib.Path(p)
        out += sorted(p.glob(pattern)) if p.is_dir() else [p]
    return out


def backfill(uri):
    R = REPO / "runs"
    jobs = [("levels", rec_levels, sorted(glob.glob(str(R / "levels_*" / "*.jsonl")))),
            ("fusion", rec_fusion, sorted(glob.glob(str(R / "levels_*" / "fusion_*" / "*.json")))),
            ("radar", rec_radar, sorted(glob.glob(str(R / "radar" / "*.jsonl")))),
            ("radar-learned", rec_learned, sorted(glob.glob(str(R / "radar" / "learned_*.json"))))]
    counts, skipped = {}, []
    for name, fn, files in jobs:
        if not files:
            print(f"{name}: no files")
            continue
        c, s = fn(files, uri)
        counts.update(c)
        skipped += s
        print(f"{name}: {len(files)} files -> {c}")
    print("logged:", counts)
    for s in skipped:
        print("skipped:", s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["levels", "fusion", "radar", "radar-learned", "backfill"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--tracking-uri", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "backfill":
        return backfill(a.tracking_uri)
    fn = {"levels": rec_levels, "fusion": rec_fusion, "radar": rec_radar, "radar-learned": rec_learned}[a.cmd]
    files = expand(a.files, "*.jsonl" if a.cmd in ("levels", "radar") else "*.json")
    c, s = fn(files, a.tracking_uri)
    print("logged:", c)
    for x in s:
        print("skipped:", x)


if __name__ == "__main__":
    main()
