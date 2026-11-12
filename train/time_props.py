"""Inference time of the property models next to the detector (Madelief 2026-11-12: "Please do", time the ensemble, since more things
to look at, like people, come later).

    train/.venv/bin/python train/time_props.py OUT.json PROPS_RUN... [--detector RUN] [--cpu-threads N]

Per props run (a single model or an ensemble folder) and per device (cuda if present, then cpu): median ms for batch 1 (one door) and
batch 4 (four doors at one stop), fp32 and, on cuda, autocast (amp); GFLOPs per crop. The detector (optional) is timed at its own imgsz
the same way as train/speed_table.py (batch 1, amp as in its config). Writes OUT.json and prints a table."""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
from doordet import props, speed


def time_props(run, device, batches=(1, 4)):
    model, meta = props.load(run, "cpu")
    rows = []
    for amp in ((False, True) if device.type == "cuda" else (False,)):
        for b in batches:
            x = torch.zeros(b, props.CHANNELS[meta["input"]], meta["imgsz"], meta["imgsz"], device=device)
            r = speed.measure(model, {"x": x}, device, amp=amp, warmup=5, n=30 if device.type == "cpu" else 50)
            rows.append({"run": str(run), "members": len(getattr(model, "models", [model])), "device": device.type, "amp": amp,
                         "batch": b, "ms": r["ms"], "gflops_per_crop": r["gflops"] / b})
    model.to("cpu")
    return rows


def time_detector(run, device):
    from doordet import adapters, config
    cfg = config.resolve(json.loads((pathlib.Path(run) / "config.json").read_text()))
    adapter = adapters.get(cfg["model"]["family"])(cfg, {0: "door", 1: "handle", 2: "hinge", 3: "window"})
    amp = bool(cfg.get("amp")) and device.type == "cuda"
    r = speed.measure(adapter.model, adapter.example_inputs(cfg["imgsz"], device), device, amp=amp, warmup=5, n=30 if device.type == "cpu" else 50)
    adapter.model.to("cpu")
    return {"run": str(run), "device": device.type, "amp": amp, "batch": 1, "imgsz": cfg["imgsz"], "ms": r["ms"], "gflops": r["gflops"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--detector")
    ap.add_argument("--cpu-threads", type=int, default=0, help="0 = torch default")
    a = ap.parse_args()
    if a.cpu_threads:
        torch.set_num_threads(a.cpu_threads)
    devices = ([torch.device("cuda")] if torch.cuda.is_available() else []) + [torch.device("cpu")]
    table = {"gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None, "cpu_threads": torch.get_num_threads(),
             "props": [], "detector": []}
    for d in devices:
        for run in a.runs:
            for r in time_props(run, d):
                table["props"].append(r)
                print(f"{pathlib.Path(run).name:28s} x{r['members']} {d.type:4s} amp {str(r['amp']):5s} batch {r['batch']}  "
                      f"{r['ms']:8.1f} ms  {r['gflops_per_crop']:6.1f} GFLOPs/crop", flush=True)
        if a.detector:
            r = time_detector(a.detector, d)
            table["detector"].append(r)
            print(f"{'detector ' + pathlib.Path(a.detector).name:28s}    {d.type:4s} amp {str(r['amp']):5s} batch 1  {r['ms']:8.1f} ms  "
                  f"{r['gflops']:6.1f} GFLOPs/frame", flush=True)
        if d.type == "cuda":
            torch.cuda.empty_cache()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(table, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
