"""What makes the full patrol runs slow? CPU, RAM and Gazebo speed per part of the run, for three short patrols.

    source /opt/ros/jazzy/setup.bash && python3 sim/smoke_test/cost_check.py --out runs/scratch/cost_check

Madelief 2026-11-14 ("start with figuring out 3"): a plain tour runs at ~0.95x real time, a full patrol (SLAM + detectors + dark
with headlight, 5-camera ring) at 0.46x. Three patrols on bld_train_005 (wheeled_ring), each with the same wall-clock budget
(--time-limit, patrol stops itself), all rendering on the RTX 4060:
    full     SLAM + drifting odometry, dark 2 % + headlight 0.7   (= cond_005_dark2)
    no_dark  the same in normal light, no headlight
    no_slam  dark + headlight on the true floor map (no SLAM, true odometry)
The detectors can't be switched off (patrol finds doors with them); their cost shows in the CPU / RAM per part.
Every 10 s: CPU (cores) and RAM (GB) per part (gazebo / bridge / slam / nav2 / detector / door_classifier / door_tour / other;
all processes carrying this tour's GZ_PARTITION), whole-machine CPU busy %, RAM available, 4060 use, and Gazebo speed
(sim s per real s). Output: OUT/result.json, OUT/report.md. Informational: changes nothing."""
import argparse, json, os, re, signal, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from gpu_ram_check import NV_ENV, rtf, sim_real   # noqa: E402

CLK = os.sysconf("SC_CLK_TCK")
PAGE = os.sysconf("SC_PAGE_SIZE")
BASE = ["bash", "sim/building/tour_test.sh", "sim/building/out/bld_train_005", "sim/configs/robots/wheeled_ring.yaml", "1.0", "4.915",
        "--hinge-look", "runs/dfine_n960_v7_hw8", "--door-step", "runs/props_door_v8_dha_448", "--patrol", "--ring", "5"]
DARK = {"LIGHT": "0.02", "HEADLIGHT": "0.7"}
SLAM = {"SLAM": "1", "ODOM_NOISE": "1"}
VARIANTS = {"full": ({**SLAM, **DARK}, ["--slam"]), "no_dark": (SLAM, ["--slam"]), "no_slam": (DARK, [])}
NAV2 = ("controller_server", "planner_server", "bt_navigator", "behavior_server", "map_server", "velocity_smoother",
        "lifecycle_manager", "smoother_server", "collision_monitor", "waypoint_follower", "amcl", "route_server", "docking_server")
GROUPS = ("gazebo", "bridge", "slam", "nav2", "detector", "door_classifier", "door_tour", "other")


def group(cmd):
    if "detect_server.py" in cmd:
        return "detector"
    if "props_server.py" in cmd:
        return "door_classifier"
    if "door_tour.py" in cmd:
        return "door_tour"
    if re.search(r"\bgz sim\b", cmd):
        return "gazebo"
    if "ros_gz_bridge" in cmd or "image_bridge" in cmd or "parameter_bridge" in cmd:
        return "bridge"
    if "slam_toolbox" in cmd:
        return "slam"
    if any(n in cmd for n in NAV2):
        return "nav2"
    return "other"


def usage(prev, now, dt):
    out = {}
    for pid, (g, t) in now.items():
        if pid in prev and dt > 0:
            out[g] = out.get(g, 0.0) + max(0, t - prev[pid][1]) / CLK / dt
    return out


def tour_procs(part):
    """{pid: (group, cpu ticks, rss bytes)} of every process in this Gazebo / ROS partition."""
    key, out = f"GZ_PARTITION={part}".encode(), {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            if key not in open(f"/proc/{p}/environ", "rb").read().split(b"\0"):
                continue
            cmd = open(f"/proc/{p}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
            st = open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()
            rss = int(open(f"/proc/{p}/statm").read().split()[1]) * PAGE
        except (OSError, IndexError, ValueError):
            continue
        out[int(p)] = (group(cmd), int(st[11]) + int(st[12]), rss)
    return out


def cpu_busy():
    f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(f) - f[3] - f[4], sum(f)     # busy, total (idle + iowait out)


def avail_gb():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable"):
            return int(ln.split()[1]) / 1e6


def gpu():
    try:
        s = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.split(",")
        return int(s[0]), int(s[1])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def run(name, out, minutes, limit):
    env_add, extra = VARIANTS[name]
    env = {k: v for k, v in os.environ.items() if k not in ("LIGHT", "SMOKE", "HEADLIGHT", "SLAM", "ODOM_NOISE")}
    env.update(NV_ENV); env.update(env_add)
    before = set((REPO / "runs/tours").iterdir())
    p = subprocess.Popen(BASE + extra + ["--time-limit", str(minutes)], cwd=REPO, env=env, stdout=open(out / f"{name}.log", "w"),
                         stderr=subprocess.STDOUT, start_new_session=True)
    part = f"tour_{p.pid}"
    t0, st, cpu, ram, busy, av, gp = time.time(), [], {g: [] for g in GROUPS}, {g: [] for g in GROUPS}, [], [], []
    prev, prev_t, prev_b = {}, None, cpu_busy()
    while p.poll() is None and time.time() - t0 < limit:
        time.sleep(10)
        now_p, now_t, now_b = tour_procs(part), time.time(), cpu_busy()
        if prev_t and any(g == "detector" for g, _, _ in now_p.values()):    # measure once the patrol itself runs
            u = usage({k: v[:2] for k, v in prev.items()}, {k: v[:2] for k, v in now_p.items()}, now_t - prev_t)
            for g in GROUPS:
                cpu[g].append(u.get(g, 0.0))
                ram[g].append(sum(r for gg, _, r in now_p.values() if gg == g) / 1e9)
            busy.append(100 * (now_b[0] - prev_b[0]) / max(1, now_b[1] - prev_b[1]))
            av.append(avail_gb()); gp.append(gpu())
            x = sim_real(part)
            if x:
                st.append(x)
        prev, prev_t, prev_b = now_p, now_t, now_b
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGINT); time.sleep(10)
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    new = [d for d in (REPO / "runs/tours").iterdir() if d not in before and d.is_dir()]
    tdir = max(new, key=lambda d: d.stat().st_mtime) if new else None
    stops = len([s for s in (tdir / "stops").iterdir() if (s / "meta.json").exists()]) if tdir and (tdir / "stops").is_dir() else 0
    return {"variant": name, "speed": rtf(st), "speed_samples": len(st), "minutes": round((time.time() - t0) / 60, 1), "stops": stops,
            "samples": len(busy), "cpu_cores": {g: mean(v) for g, v in cpu.items()},
            "ram_gb_max": {g: round(max(v), 2) if v else None for g, v in ram.items()},
            "machine_cpu_busy_pct": mean(busy), "machine_cpu_busy_max": round(max(busy), 1) if busy else None,
            "ram_available_min_gb": round(min(av), 2) if av else None,
            "gpu_util_pct": mean([g[0] for g in gp if g]), "gpu_mem_max_mb": max([g[1] for g in gp if g], default=None),
            "exit": p.returncode, "tour_dir": str(tdir) if tdir else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=8)
    ap.add_argument("--only", default="", help="comma list of variants (default all: full,no_dark,no_slam)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {"started": time.strftime("%Y-%m-%d %H:%M"), "cpus": os.cpu_count(), "runs": []}
    for name in (a.only.split(",") if a.only else VARIANTS):
        r = run(name, out, a.minutes, limit=(a.minutes + 6) * 60)
        res["runs"].append(r); print(json.dumps(r), flush=True)
        (out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    rows = ["| | " + " | ".join(r["variant"] for r in res["runs"]) + " |", "|---" * (len(res["runs"]) + 1) + "|"]
    def row(label, f):
        rows.append(f"| {label} | " + " | ".join(str(f(r)) for r in res["runs"]) + " |")
    row("Gazebo speed (sim s per real s)", lambda r: r["speed"])
    row(f"stops in {a.minutes:g} min", lambda r: r["stops"])
    row(f"machine CPU busy % (mean / max, {res['cpus']} threads)", lambda r: f"{r['machine_cpu_busy_pct']} / {r['machine_cpu_busy_max']}")
    for g in GROUPS:
        row(f"CPU cores: {g}", lambda r, g=g: r["cpu_cores"][g])
    for g in GROUPS:
        row(f"RAM GB max: {g}", lambda r, g=g: r["ram_gb_max"][g])
    row("RAM available, lowest (GB)", lambda r: r["ram_available_min_gb"])
    row("4060 busy % / memory MB", lambda r: f"{r['gpu_util_pct']} / {r['gpu_mem_max_mb']}")
    (out / "report.md").write_text(f"# Patrol cost check {res['started']}\n\n" + "\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
