"""The radar's raw range x azimuth power map and a top-down (BEV) image of it, for a learned doorway detector
(Madelief 2026-11-04, "just do both"). This is what a real mmWave radar makes before CFAR throws most of it away;
RadarHD (arXiv 2206.09273) learns from it.

ra_map: per-pixel returns (radar.pixel_returns: material, specular lobe, 1/r^4, glass pass-through) summed over elevation into
a fine (range, azimuth) grid, blurred in azimuth with the model's beam (Gaussian, FWHM = az_res_deg: why the single chip is
blurry) and in range with the range resolution, plus a noise floor with exponential (Rayleigh-amplitude) speckle. Power units
as radar.simulate: the bin's pixel power divided by its solid angle (az cell x el cell, el cell capped at the fov), summed over
the elevation cells, so a bin is "detected" by simulate when its power is above radar.NOISE_FLOOR. In dB (no reference: 10 log10
of the linear value; noise floor = 10 log10(NOISE_FLOOR) = -60 dB). Range is the slant range; the BEV takes it as horizontal
(the radar cannot tell elevation; the camera is ~1 m high, so a wall at 3 m is off by ~3 cm).

bev: inverse mapping. Every BEV pixel centre -> the metric position in each source radar's frame (range, azimuth; positions are
horizontal, the camera's roll and pitch are ignored, yaw only) -> bilinear sample of that map's linear power. Several maps
(stacked views): the mean of the linear power over the maps that cover the pixel (speckle averages out, a wall seen by one
view stays at full strength; max of dB would raise the noise floor with every added view). Pixels outside every map's field
of view get the noise floor value.

Image axes: the destination camera at the bottom centre, forward = up, left = left; pixel (row i, col j) has its centre at
forward (H - i - 0.5) x res and left (W / 2 - j - 0.5) x res."""
import math
import numpy as np

try:
    from doorsim import radar
except ImportError:  # run with sim/doorsim itself on the path
    import radar

MAX_R = 8.0                  # m, range of the map (and the BEV's forward extent)
AZ_FOV = 60.0                # deg, half field of view
AZ_STEP = 0.5                # deg, fine grid step (finer than every beam)
FWHM_TO_SIGMA = 1 / (2 * math.sqrt(2 * math.log(2)))
NOISE_DB = 10 * math.log10(radar.NOISE_FLOOR)


def _blur(a, sigma, axis):
    """Gaussian blur of a (sigma in bins) along an axis, kernel summing to 1, zero outside the grid."""
    if sigma <= 0:
        return a
    n = int(math.ceil(4 * sigma))
    k = np.exp(-0.5 * (np.arange(-n, n + 1) / sigma) ** 2)
    k /= k.sum()
    a = np.moveaxis(a, axis, 0)
    pad = np.zeros((a.shape[0] + 2 * n,) + a.shape[1:])
    pad[n:-n] = a
    out = sum(k[i] * pad[i:i + a.shape[0]] for i in range(len(k)))
    return np.moveaxis(out, 0, axis)


def ra_map(depth, seg, depth_noglass, K, camera, model, mats, rng):
    """{"power_db": (n_range, n_az), "ranges": bin centres (m), "az_deg": bin centres (deg, positive = left)}."""
    res = radar.RANGE_RES
    nr, na = int(round(MAX_R / res)), int(round(2 * AZ_FOV / AZ_STEP))
    ranges, az_deg = (np.arange(nr) + 0.5) * res, -AZ_FOV + (np.arange(na) + 0.5) * AZ_STEP
    cam_pos = np.array(camera["world_position"], float)
    R = np.array(radar.effects.frames.rot(*camera["world_rpy"]))
    pts, r, pw = radar.pixel_returns(depth, seg, depth_noglass, K, camera, mats)
    b = (pts - cam_pos) @ R  # camera body frame: x forward, y left, z up (azimuth as in radar.simulate)
    az = np.degrees(np.arctan2(b[:, 1], b[:, 0]))
    ir, ia = np.floor(r / res).astype(int), np.floor((az + AZ_FOV) / AZ_STEP).astype(int)
    ok = (pw > 0) & (ir >= 0) & (ir < nr) & (ia >= 0) & (ia < na)
    grid = np.zeros((nr, na))
    np.add.at(grid, (ir[ok], ia[ok]), pw[ok])
    # per solid angle of a beam cell (azimuth step x the elevation cell), like simulate's beam-filling power; the beam blur
    # below is normalised, so a uniform field keeps this density
    el_cell = math.radians(min(model["el_res_deg"], model["fov_deg"][1]))
    grid /= math.radians(AZ_STEP) * el_cell
    grid = _blur(grid, model["az_res_deg"] * FWHM_TO_SIGMA / AZ_STEP, axis=1)
    grid = _blur(grid, model["range_res"] * FWHM_TO_SIGMA / res, axis=0)
    noise = radar.NOISE_FLOOR * rng.exponential(1.0, grid.shape)  # speckle: exponential power = Rayleigh amplitude
    return {"power_db": 10 * np.log10(grid + noise), "ranges": ranges, "az_deg": az_deg}


def _sample(m, rng_m, az_deg):
    """Bilinear sample of a map's linear power at (range m, azimuth deg) arrays; nan outside the grid."""
    p = 10.0 ** (np.asarray(m["power_db"]) / 10)
    ranges, az = np.asarray(m["ranges"]), np.asarray(m["az_deg"])
    fr = (rng_m - ranges[0]) / (ranges[1] - ranges[0])
    fa = (az_deg - az[0]) / (az[1] - az[0])
    inside = (rng_m >= 0) & (rng_m <= ranges[-1] + (ranges[1] - ranges[0]) / 2) & (np.abs(az_deg) <= AZ_FOV)
    fr, fa = np.clip(fr, 0, len(ranges) - 1), np.clip(fa, 0, len(az) - 1)
    r0, a0 = np.minimum(fr.astype(int), len(ranges) - 2), np.minimum(fa.astype(int), len(az) - 2)
    wr, wa = fr - r0, fa - a0
    v = (p[r0, a0] * (1 - wr) * (1 - wa) + p[r0 + 1, a0] * wr * (1 - wa) + p[r0, a0 + 1] * (1 - wr) * wa
         + p[r0 + 1, a0 + 1] * wr * wa)
    return np.where(inside, v, np.nan)


def _xy_yaw(camera):
    return np.array(camera["world_position"][:2], float), float(camera["world_rpy"][2])


def _to_local(xy, origin, yaw):
    """World horizontal points (..., 2) -> (forward, left) in a frame at origin with the given yaw."""
    d = np.asarray(xy, float) - origin
    c, s = math.cos(yaw), math.sin(yaw)
    return d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c


def bev(maps, cameras, dst_camera, size_m=8.0, res=0.05):
    """Float image (H, W) in dB, H = W = size_m / res, robot (dst_camera) at the bottom centre, forward = up, left = left.
    maps[i] was made at cameras[i]. Combination and outside value: see the module doc."""
    n = int(round(size_m / res))
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    fwd, left = (n - i - 0.5) * res, (n / 2 - j - 0.5) * res
    o, yaw = _xy_yaw(dst_camera)
    c, s = math.cos(yaw), math.sin(yaw)
    world = o + np.stack([fwd * c - left * s, fwd * s + left * c], axis=-1)  # dst frame -> world
    total, count = np.zeros((n, n)), np.zeros((n, n))
    for m, cam in zip(maps, cameras):
        o2, yaw2 = _xy_yaw(cam)
        f2, l2 = _to_local(world, o2, yaw2)
        v = _sample(m, np.hypot(f2, l2), np.degrees(np.arctan2(l2, f2)))
        ok = np.isfinite(v)
        total[ok] += v[ok]
        count[ok] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        img = 10 * np.log10(total / count)
    return np.where(count > 0, img, NOISE_DB)


def bev_box(door_gt, dst_camera, size_m=8.0, res=0.05, thick=0.3):
    """[x0, y0, x1, y1] BEV pixels (float) of a doorway segment: door centre +- width / 2 along the wall (perpendicular to the
    door's horizontal normal), +- thick / 2 across it; the axis-aligned box of that rectangle. None when the centre is behind the
    robot or outside the image; a box partly outside is clipped; None when the clipped box is under 3 px in either direction."""
    n = int(round(size_m / res))
    o, yaw = _xy_yaw(dst_camera)
    centre = np.array(door_gt["center"][:2], float)
    nv = np.array(door_gt["normal"][:2], float)
    nv = nv / np.hypot(*nv)
    tv = np.array([-nv[1], nv[0]])
    f, l = _to_local(centre, o, yaw)
    px = lambda f, l: (n / 2 - l / res, n - f / res)  # (x, y) pixel coordinates (edges: pixel j spans [j, j + 1])
    cx, cy = px(f, l)
    if f <= 0 or not (0 <= cx <= n and 0 <= cy <= n):
        return None
    hw, ht = door_gt["width"] / 2, thick / 2
    corners = [px(*_to_local(centre + a * hw * tv + b * ht * nv, o, yaw)) for a in (-1, 1) for b in (-1, 1)]
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    x0, y0, x1, y1 = (float(np.clip(v, 0, n)) for v in (min(xs), min(ys), max(xs), max(ys)))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    return [x0, y0, x1, y1]


def bev_to_world(px, py, dst_camera, size_m=8.0, res=0.05):
    """[x, y, z] world point of a BEV pixel position (x = column, y = row, float; pixel edges as in bev_box). Exact inverse of
    the mapping of bev / bev_box: yaw only, z = the camera's z."""
    n = int(round(size_m / res))
    fwd, left = (n - py) * res, (n / 2 - px) * res
    o, yaw = _xy_yaw(dst_camera)
    c, s = math.cos(yaw), math.sin(yaw)
    return [float(o[0] + fwd * c - left * s), float(o[1] + fwd * s + left * c), float(dst_camera["world_position"][2])]
