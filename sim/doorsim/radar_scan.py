"""Turn a raw gpu_lidar ring into radar-like ranges (beam widening, range bins, noise). Contract: sim/tests/test_radar_ring.py."""
import numpy as np


def process(ranges, angle_inc, rmin, rmax, beam, res, noise=0.0, rng=None):
    """Radar ranges from a raw ring (see the header of sim/tests/test_radar_ring.py). Same length, inf = no return."""
    r = np.asarray(ranges, dtype=float)
    with np.errstate(invalid="ignore"):
        r = np.where(np.isfinite(r) & (r >= rmin) & (r <= rmax), r, np.inf)
    k = int(np.floor(beam / 2 / angle_inc + 1e-9))     # whole steps within +-beam / 2
    out = r.copy()
    for s in range(1, k + 1):
        np.minimum(out, np.roll(r, s), out=out)
        np.minimum(out, np.roll(r, -s), out=out)
    ok = np.isfinite(out)
    out[ok] = (np.floor(out[ok] / res) + 0.5) * res
    if noise > 0 and ok.any():
        out[ok] += (rng if rng is not None else np.random.default_rng()).normal(0.0, noise, ok.sum())
    out[ok & ((out < rmin) | (out > rmax))] = np.inf
    return out
