"""Inference cost of a model: median ms per image (batch 1) and GFLOPs per image."""
import statistics, time
import torch
from torch.utils.flop_counter import FlopCounterMode

from .loop import autocast


def measure(model, example_inputs, device, amp=False, warmup=10, n=50):
    model.to(device).eval()
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:  # fp32, no autocast
            model(**example_inputs)
        ctx = autocast({"amp": amp}, device)[0]
        times = []
        for i in range(warmup + n):
            sync(); t0 = time.perf_counter()
            with ctx:
                model(**example_inputs)
            sync()
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000)
    return {"ms": statistics.median(times), "gflops": fc.get_total_flops() / 1e9}
