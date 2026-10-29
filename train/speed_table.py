"""usage: train/.venv/bin/python train/speed_table.py <search.yaml>   (ms + GFLOPs per image for every model x resolution)"""
import copy, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
from doordet import adapters, config, search, speed
from doordet.data import CocoDetection


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    s = search.load(sys.argv[1])
    base = config.load(s["base"])
    for k, v in s["overrides"].items():
        search._set(base, k, v)
    base = config.resolve(base)
    device = torch.device(base["device"])
    id2label = CocoDetection(base["export"], "train").id2label
    entries = []
    for m in s["space"]["models"]:
        for imgsz in search.imgsizes(s["space"]):
            cfg = copy.deepcopy(base)
            cfg.update(imgsz=imgsz)
            cfg["model"] = {"family": m["family"], "checkpoint": m["checkpoint"]}
            e = {"family": m["family"], "checkpoint": m["checkpoint"], "imgsz": imgsz}
            try:
                adapter = adapters.get(m["family"])(cfg, id2label)
                e.update(speed.measure(adapter.model, adapter.example_inputs(imgsz, device), device, amp=cfg["amp"]))
                print(f"{m['checkpoint']:45s} {imgsz:4d}  {e['ms']:7.2f} ms  {e['gflops']:7.1f} GFLOPs", flush=True)
            except torch.cuda.OutOfMemoryError:
                e.update(ms=None, gflops=None, error="oom")
                print(f"{m['checkpoint']:45s} {imgsz:4d}  oom", flush=True)
            entries.append(e)
            adapter = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
    table = {"device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu", "amp": base["amp"], "entries": entries}
    path = pathlib.Path(s["speed_table"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=2))
    import mlflow
    mlflow.set_tracking_uri(base["mlflow"]["tracking_uri"])
    mlflow.set_experiment("doggy-door-speed")
    with mlflow.start_run(run_name=s["name"]):
        mlflow.log_params({"device": table["device"], "amp": table["amp"]})
        mlflow.log_artifact(str(path))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
