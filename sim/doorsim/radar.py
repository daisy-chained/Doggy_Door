"""Radar proxy (Madelief's design, 2026-11-04): a wide depth + label render turned into radar returns. Coarse angle cells,
specular dropout at grazing angles, a material layer (metal strong; wood / plaster / glass weak; glass lets part of the signal
through), sparse points, range noise. Smoke has no parameter: radar ignores it. Two types: "single" (single-chip, like a TI
IWR6843) and "imaging" (cascade imaging radar).

Pipeline (simulate): per-pixel world points + surface normals from the depth -> incidence angle -> power per material
(returned_power) -> bin into (azimuth cell, elevation cell, range bin) -> sum power per bin -> keep bins above NOISE_FLOOR
(CFAR-lite) -> one noisy point per detection -> random keep -> strongest max_points.
inject() adds the wide depth + label cameras to a v4 world; radar_render.py replays a dataset with them."""
import copy, math
import xml.etree.ElementTree as ET
import numpy as np

try:
    from doorsim import effects, thermal
except ImportError:  # run with sim/doorsim itself on the path
    import effects, thermal

# --- sensor models. Single chip: TI SWRU546E (IWR6843 antenna / resolution numbers). Cascade imaging numbers are
# approximate (typical 4-chip cascade datasheet values, not a specific product). ---
FOV_DEG = [120, 40]          # azimuth x elevation field of view, both types (the render's field of view)
RANGE_RES = 0.05             # m, range bin (both types; ~3 GHz of bandwidth, approximate)
MODELS = {
    # az_res_deg 15 (TI SWRU546E, 4 RX azimuth); el_res_deg ~58 (2 elevation channels) is wider than the 40 deg view -> one elevation cell
    "single": {"az_res_deg": 15.0, "el_res_deg": 58.0, "fov_deg": FOV_DEG, "range_res": RANGE_RES, "range_noise": 0.03,
               "max_points": 200},
    # cascade imaging radar, approximate: ~1.5 deg azimuth, ~15 deg elevation
    "imaging": {"az_res_deg": 1.5, "el_res_deg": 15.0, "fov_deg": FOV_DEG, "range_res": RANGE_RES, "range_noise": 0.02,
                "max_points": 2000},
}
ANGLE_NOISE_FRAC = 1 / 6     # angular noise sigma = this x the cell width (clipped at 2 sigma)
KEEP_FRACTION = 0.7          # random keep of detections (sparse, missed detections)
MAX_RANGE = 15.0             # m, same as the render's far clip

# --- reflection per material, as a gain in dB (0 = strongest, metal) ---
REFLECT_DB = {
    "metal": 0.0,            # strongest reflector (reference)
    "wall": -4.8,            # drywall reflection loss ~4.8 dB at 60 GHz (NYURay, arXiv 2410.03104)
    "wood": -8.0,            # wood, literature-guided (a few dB weaker than drywall)
    "glass": -6.0,           # glass surface reflection, literature-guided
    "floor": -6.0,           # floor (mostly seen at grazing angles, where the lobe cuts it down)
    "cardboard": -12.0,      # weak, literature-guided
    "plastic": -10.0,        # weak, literature-guided
    "plant": -14.0,          # leaves scatter weakly (guess, not from a source)
}
GLASS_PASS_DB = -10.0        # one-way loss through interior glass: 7.5-18 dB (NIST, 60 GHz); the return from behind goes through twice

# --- specular lobe: smooth surfaces reflect away from the radar at grazing angles (DREAM-PCD, arXiv 2309.15374) ---
LOBE_EXPONENT = 4            # lobe = cos(incidence)^n
LOBE_CUTOFF_DEG = 75.0       # exactly no return beyond this incidence angle
UNKNOWN_INCIDENCE_DEG = 45.0  # incidence where the normal can't be estimated (depth edges, image border)

# --- detection (CFAR-lite): a fixed noise floor on the beam-filling power of a bin. Units: sum over the bin's pixels of
# (reflect gain x lobe / r^4) x pixel solid angle, divided by the cell's solid angle. 1e-6 is 36 dB below a drywall wall
# facing the radar at 3 m (4.1e-3), 15 dB below the same wall at 60 deg incidence at 3 m depth, and low enough that a wall
# seen through glass (-20 dB two-way) at 6 m is still detected (2.6e-6). Weak, far or grazing surfaces fall below it. ---
NOISE_FLOOR = 1e-6
NORMAL_JUMP = 0.25           # a depth step larger than this x depth (+ 0.1 m) between neighbours is an edge, not a surface


def camera_K(w, h, hfov):
    """Pinhole intrinsics (9 floats row-major, like the dataset's camera_intrinsics K), square pixels; hfov in radians."""
    f = (w / 2) / math.tan(hfov / 2)
    return [f, 0.0, w / 2, 0.0, f, h / 2, 0.0, 0.0, 1.0]


def materials(attrs):
    """{seg id: material class} for the labelled things in a world's attributes, plus the labelled structure."""
    out = {i: {"floor": "floor", "back_wall": "wall", "room_shell": "wall"}[n] for n, i in thermal.STRUCT_IDS.items()}
    for name, o in attrs["objects"].items():
        if "id" not in o:
            continue
        if o.get("part") == "glass_pane":
            m = "glass"
        elif o.get("class") == 1:
            m = "wood"
        elif o.get("class") == 2:
            m = "metal"
        elif name.startswith("box_stack") or name.startswith(("cabinet", "wardrobe", "drawers", "bookcase", "table", "chair")):
            m = "wood"
        elif name.startswith("box_"):
            m = "cardboard"
        elif name.startswith(("cylinder", "sphere")):
            m = "plastic"
        elif name.startswith(("frame_", "whiteboard")):
            m = "wall"
        elif name.startswith("plant"):
            m = "plant"
        else:
            m = "wall"
        out[o["id"]] = m
    return out


def returned_power(reflect_db, incidence_deg, r):
    """Linear power: 10^(dB/10) x cos(incidence)^n / r^4; exactly 0 beyond LOBE_CUTOFF_DEG. Arrays or scalars."""
    inc = np.asarray(incidence_deg, float)
    lobe = np.where(inc < LOBE_CUTOFF_DEG, np.cos(np.radians(np.minimum(inc, 89.0))) ** LOBE_EXPONENT, 0.0)
    out = 10.0 ** (np.asarray(reflect_db, float) / 10) * lobe / np.maximum(np.asarray(r, float), 1e-3) ** 4
    return float(out) if out.ndim == 0 else out


def _incidence(wp, depth, cam_pos):
    """Angle (deg) between each pixel's ray and the surface normal (from finite differences of the world points);
    UNKNOWN_INCIDENCE_DEG where no normal can be made."""
    ray = wp - cam_pos
    ray = ray / np.linalg.norm(ray, axis=-1, keepdims=True)
    lim = (NORMAL_JUMP * np.nan_to_num(depth) + 0.1)[..., None]

    def diff(axis):
        d = np.full_like(wp, np.nan)
        a = np.diff(wp, axis=axis)
        bad = np.abs(a) > (np.take(lim, range(a.shape[axis]), axis=axis))
        a = np.where(bad.any(-1, keepdims=True), np.nan, a)
        fwd, bwd = d.copy(), d.copy()
        if axis == 0:
            fwd[:-1], bwd[1:] = a, a
        else:
            fwd[:, :-1], bwd[:, 1:] = a, a
        return np.where(np.isnan(fwd), bwd, fwd)

    n = np.cross(diff(1), diff(0))
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / norm
        inc = np.degrees(np.arccos(np.clip(np.abs((n * ray).sum(-1)), 0, 1)))
    return np.where(np.isfinite(inc), inc, UNKNOWN_INCIDENCE_DEG)


def _pixel_solid_angle(shape, K):
    H, W = shape
    u, v = np.meshgrid(np.arange(W) - K[2], np.arange(H) - K[5])
    x, y = u / K[0], v / K[4]
    return 1.0 / (K[0] * K[4]) / (1 + x * x + y * y) ** 1.5


def _cells(res, fov):
    """(number of cells, cell width, start angle) over the fov, centred; a cell wider than the fov is capped at the fov."""
    res = min(res, fov)
    n = max(1, math.ceil(fov / res - 1e-9))
    return n, res, -n * res / 2


def _returns(depth, wp, incidence, solid, db, cam_pos, valid):
    """Per-pixel (range, power) arrays for the valid pixels; db = per-pixel reflect gain (dB)."""
    r = np.linalg.norm(wp - cam_pos, axis=-1)
    p = returned_power(db, incidence, np.where(valid, r, 1.0)) * solid
    return r, p


def pixel_returns(depth, seg, depth_noglass, K, camera, mats):
    """The radar's per-pixel returns before any binning or thresholding: (world points (N, 3), range from the radar (N,) in m,
    power (N,)) for every valid pixel of the surface layer, then of the layer behind glass (two-way pass-through loss,
    reflecting like a wall), concatenated. Power = reflect gain x lobe / r^4 x pixel solid angle (linear; zero = no return).
    depth / depth_noglass: planar z-depth in m (0 / NaN = no return), seg: label image, camera: {"world_position", "world_rpy"},
    mats: materials(attrs). Shared by simulate (detections) and radar_heatmap (raw range x azimuth power map)."""
    depth = np.asarray(depth, float)
    depth_noglass = np.asarray(depth_noglass, float)
    seg = np.asarray(seg)
    cam_pos = np.array(camera["world_position"], float)
    solid = _pixel_solid_angle(depth.shape, K)

    names = np.empty(seg.shape, object)
    names[:] = "wall"
    for i in np.unique(seg):
        names[seg == i] = mats.get(int(i), "wall")
    db = np.vectorize(lambda m: REFLECT_DB[m], otypes=[float])(names)

    layers = []  # (world points, range, power) per return layer
    for d, level in ((depth, db), ):
        wp = effects.world_points(d, K, camera)
        valid = np.isfinite(wp[..., 0])
        inc = _incidence(wp, d, cam_pos)
        r, p = _returns(d, wp, inc, solid, level, cam_pos, valid)
        layers.append((wp[valid], r[valid], p[valid]))
    glass = names == "glass"
    if glass.any():  # the return from behind the glass: two-way loss through the pane, reflecting like a wall
        wp = effects.world_points(depth_noglass, K, camera)
        valid = np.isfinite(wp[..., 0]) & glass
        inc = _incidence(wp, depth_noglass, cam_pos)
        r, p = _returns(depth_noglass, wp, inc, solid, REFLECT_DB["wall"] + 2 * GLASS_PASS_DB, cam_pos, valid)
        layers.append((wp[valid], r[valid], p[valid]))
    return (np.concatenate([l[0] for l in layers]), np.concatenate([l[1] for l in layers]),
            np.concatenate([l[2] for l in layers]))


def simulate(depth, seg, depth_noglass, K, camera, model, mats, rng):
    """Radar detections from one wide render. depth / depth_noglass: planar z-depth in m (0 / NaN = no return),
    seg: label image, camera: {"world_position", "world_rpy"}, model: MODELS[...], mats: materials(attrs).
    Returns {"points": (N, 3) world xyz, "power": (N,)} (power in NOISE_FLOOR units: beam-filling power of the bin)."""
    cam_pos = np.array(camera["world_position"], float)
    R = np.array(effects.frames.rot(*camera["world_rpy"]))
    pts, rng_m, pw = pixel_returns(depth, seg, depth_noglass, K, camera, mats)

    empty = {"points": np.zeros((0, 3)), "power": np.zeros(0)}
    keep = (pw > 0) & (rng_m < MAX_RANGE)
    if not keep.any():
        return empty
    pts, rng_m, pw = pts[keep], rng_m[keep], pw[keep]

    # angles in the camera body frame (x forward, y left, z up)
    b = (pts - cam_pos) @ R
    az = np.degrees(np.arctan2(b[:, 1], b[:, 0]))
    el = np.degrees(np.arctan2(b[:, 2], np.hypot(b[:, 0], b[:, 1])))
    naz, waz, az0 = _cells(model["az_res_deg"], model["fov_deg"][0])
    nel, wel, el0 = _cells(model["el_res_deg"], model["fov_deg"][1])
    nr = int(math.ceil(MAX_RANGE / model["range_res"]))
    ia, ie = np.floor((az - az0) / waz).astype(int), np.floor((el - el0) / wel).astype(int)
    ir = np.floor(rng_m / model["range_res"]).astype(int)
    ok = (ia >= 0) & (ia < naz) & (ie >= 0) & (ie < nel) & (ir < nr)
    ia, ie, ir, az, el, rng_m, pw = ia[ok], ie[ok], ir[ok], az[ok], el[ok], rng_m[ok], pw[ok]
    if len(pw) == 0:
        return empty
    key = (ia * nel + ie) * nr + ir
    uk, inv = np.unique(key, return_inverse=True)
    tot = np.bincount(inv, pw)
    cell_sr = math.radians(waz) * math.radians(wel)  # beam filling: power / the cell's solid angle
    det = tot / cell_sr > NOISE_FLOOR
    if not det.any():
        return empty
    mean = lambda x: np.bincount(inv, pw * x)[det] / tot[det]
    r_d, az_d, el_d = mean(rng_m), mean(az), mean(el)
    power = tot[det] / cell_sr / NOISE_FLOOR
    n = len(r_d)
    clip = lambda s: np.clip(rng.normal(0, s, n), -2 * s, 2 * s)  # angular noise, clipped at 2 sigma
    az_d = az_d + clip(waz * ANGLE_NOISE_FRAC)
    el_d = el_d + clip(wel * ANGLE_NOISE_FRAC)
    r_d = r_d + rng.normal(0, model["range_noise"], n)
    kept = rng.random(n) < KEEP_FRACTION
    r_d, az_d, el_d, power = r_d[kept], az_d[kept], el_d[kept], power[kept]
    order = np.argsort(-power, kind="stable")[:model["max_points"]]
    r_d, az_d, el_d, power = r_d[order], az_d[order], el_d[order], power[order]
    a, e = np.radians(az_d), np.radians(el_d)
    body = np.stack([r_d * np.cos(e) * np.cos(a), r_d * np.cos(e) * np.sin(a), r_d * np.sin(e)], axis=-1)
    return {"points": body @ R.T + cam_pos, "power": power}


# --- wide cameras ---
WIDE_HFOV = math.radians(120)
WIDE_SIZE = (480, 100)       # px; 120 deg x ~40 deg with square pixels
WIDE_CLIP = ("0.05", "15")


def inject(sdf_text):
    """World SDF text with a wide depth camera (topic "radar/depth") and a wide segmentation camera ("radar/seg") next to the
    rig's rgbd camera (same link + pose, same update rate), plus the structure labels (floor / back wall / room shell)."""
    sdf_text = thermal.inject(sdf_text, {}, thermal=False)  # labels only
    root = ET.fromstring(sdf_text.encode())
    rig = next(m for m in root.find("world").findall("model") if m.get("name") == "rig")
    for link in rig.iter("link"):
        rgbd = next((s for s in link.findall("sensor") if s.get("type") == "rgbd_camera"), None)
        if rgbd is None:
            continue
        seg_ref = next(s for s in link.findall("sensor") if s.get("type") == "segmentation_camera")
        at = list(link).index(rgbd) + 1
        for name, typ, topic in (("radar_depth", "depth_camera", "radar/depth"), ("radar_seg", "segmentation", "radar/seg")):
            s = ET.Element("sensor", name=name, type=typ)
            if rgbd.find("pose") is not None:
                s.append(copy.deepcopy(rgbd.find("pose")))
            ET.SubElement(s, "topic").text = topic
            for tag in ("update_rate", "always_on"):
                ET.SubElement(s, tag).text = rgbd.find(tag).text
            c = ET.SubElement(s, "camera")
            if typ == "segmentation":
                c.append(copy.deepcopy(seg_ref.find("camera/segmentation_type")))
            ET.SubElement(c, "horizontal_fov").text = f"{WIDE_HFOV:.6f}"
            img = ET.SubElement(c, "image")
            ET.SubElement(img, "width").text, ET.SubElement(img, "height").text = str(WIDE_SIZE[0]), str(WIDE_SIZE[1])
            clip = ET.SubElement(c, "clip")
            ET.SubElement(clip, "near").text, ET.SubElement(clip, "far").text = WIDE_CLIP
            link.insert(at, s)
            at += 1
        break
    else:
        raise ValueError("rig has no rgbd_camera sensor")
    ET.indent(root)
    return ET.tostring(root, encoding="unicode", xml_declaration=True)
