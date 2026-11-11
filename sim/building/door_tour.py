"""Drive the robot past every reachable door side of a building (phase 2e; Madelief 2026-11-07: Nav2 + waypoints over the
approach poses, at a door stop ~2.5 m and look, then ~1 m, report, move on). Needs robot.launch.py + nav.launch.py running.

    source /opt/ros/jazzy/setup.bash
    python3 sim/building/door_tour.py sim/building/out/bld_train_000 --start 1.0 4.04 [--look 2.0] [--bag runs/tours/bld_train_000]
                                      [--max-visits N] [--out runs/tours/bld_train_000.tour.json] [--frames DIR]

Order = doorsim.robot.door_tour (greedy nearest next door side, far stop then near stop). Per stop it logs whether Nav2 reached it,
the time it took and the final position / heading error (from tf map -> base_link). Perception at the stops comes later; for now the
robot only stops for --look seconds (sim time) so a recording has frames of every door from both distances.
--frames saves what the robot sees at every stop (Madelief 2026-11-09): DIR/NNN_door<id>_<side>_<dist>/ with rgb / depth / seg png,
the true 2D + 3D boxes (door id per box) and meta.json (stop, reached, robot + camera pose in the map). doorsim.stop_frames.
--bag records the run with rosbag2 (camera, depth, boxes, lidar, odom, tf), for "record first, score after".
--door-step PROPS_RUN (with --hinge-look): the reactive door step instead of the fixed hinge look (DoD C, Madelief 2026-11-11):
doorsim.door_step.visit picks the extra looks itself until sure (at most --step-n), reports hinged + hinge side or sliding (P(slide)
from the door-crop classifier, train/props_server.py, mean over the stop frames, sliding at >= --p-slide-t); the sim's truth checks
it; right -> the door opens fully (/door_<id>_panel/cmd_pos), wrong -> a new look, then a new guess. The full report is logged.
"""
import argparse, json, math, pathlib, signal, subprocess, sys, time

import numpy as np
import rclpy
from PIL import Image
from geometry_msgs.msg import PoseStamped, Twist
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from tf2_ros import Buffer, TransformListener

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import building as B, door_step, doors as D, fog, hinge_look, patrol, robot  # noqa: E402

REPO = SIM.parent
IMG_W, HFOV = 640, 1.3          # robot camera = training rig (doorsim.robot)

BAG_TOPICS = ["/robot/rgbd/image", "/robot/rgbd/depth_image", "/robot/rgbd/camera_info", "/scan", "/odom", "/tf", "/tf_static", "/clock",
              "/cmd_vel", "/joint_states"]


def goal(nav, x, y, yaw):
    g = PoseStamped()
    g.header.frame_id = "map"
    g.header.stamp = nav.get_clock().now().to_msg()
    g.pose.position.x, g.pose.position.y = float(x), float(y)
    g.pose.orientation.z, g.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return g


def where(nav, tf, frame="base_link"):
    try:
        t = tf.lookup_transform("map", frame, rclpy.time.Time()).transform
    except Exception:
        return None
    q = t.rotation
    return t.translation.x, t.translation.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))


def wait_sim(nav, seconds):
    end = nav.get_clock().now().nanoseconds + seconds * 1e9
    while nav.get_clock().now().nanoseconds < end:
        rclpy.spin_once(nav, timeout_sec=0.05)


def wait_active(nav, nodes=("map_server", "bt_navigator"), per_call=5.0, total=300.0):
    """Like BasicNavigator.waitUntilNav2Active, but every get_state call has a timeout and is retried: a lost service response
    ("failed to send response to /map_server/get_state (timeout)", 2026-11-11) made the stock wait hang forever."""
    from lifecycle_msgs.srv import GetState
    end = time.time() + total
    for n in nodes:
        cli = nav.create_client(GetState, f"/{n}/get_state")
        while True:
            if time.time() > end:
                sys.exit(f"Nav2 node {n} not active after {total:.0f} s")
            if cli.wait_for_service(timeout_sec=per_call):
                fut = cli.call_async(GetState.Request())
                rclpy.spin_until_future_complete(nav, fut, timeout_sec=per_call)
                if fut.done() and fut.result() is not None and fut.result().current_state.label == "active":
                    break
                if not fut.done():
                    fut.cancel()
            time.sleep(1.0)
        nav.destroy_client(cli)
    print("Nav2 is ready (own check)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("building", help="path without extension, e.g. sim/building/out/bld_train_000")
    ap.add_argument("--start", nargs=2, type=float, required=True, help="spawn x y (same as robot.launch.py)")
    ap.add_argument("--look", type=float, default=2.0, help="seconds (sim time) standing still at each stop")
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds (wall) per stop before giving up")
    ap.add_argument("--max-visits", type=int, default=0)
    ap.add_argument("--bag", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--frames", default="", help="folder: save camera frames + true boxes at every stop")
    ap.add_argument("--name", default="robot", help="robot model name (camera topics live under it)")
    ap.add_argument("--hinge-look", default="", help="detector run (e.g. runs/dfine_n960_v7_hw8): after each door side, guess the "
                    "hinge edge from the handle and take extra looks aimed at it (option A, Madelief 2026-11-10); needs --frames")
    ap.add_argument("--look-dist", type=float, default=1.5, help="m from the hinge edge for the extra looks")
    ap.add_argument("--look-radius", type=float, default=0.33, help="m kept free around a look spot (footprint half-diagonal)")
    ap.add_argument("--door-step", default="", help="props classifier run (e.g. runs/props_door_v7_dha_448): reactive door step "
                    "+ open the door on a right report (needs --hinge-look)")
    ap.add_argument("--step-n", type=int, default=2, help="door step: extra looks before the first report (Claude 2026-11-11, replay)")
    ap.add_argument("--p-slide-t", type=float, default=0.4, help="door step: mean P(slide) >= this -> sliding (Claude 2026-11-11)")
    ap.add_argument("--no-open", action="store_true", help="door step: check reports but don't open doors")
    ap.add_argument("--rounds", type=int, default=1, help="2 = after the tour, load a map with the doors it opened and visit the "
                    "door sides that became reachable (Madelief 2026-11-11: two rounds; needs --door-step and --out)")
    ap.add_argument("--patrol", action="store_true", help="run 3, no markers (Madelief 2026-11-11): no door positions; patrol "
                    "viewpoints over the floor map, look around, door-step every door it finds (needs --door-step)")
    ap.add_argument("--vp-spacing", type=float, default=1.5, help="patrol: m between viewpoints")
    ap.add_argument("--max-door-depth", type=float, default=4.0, help="patrol: ignore door boxes further than this (m)")
    ap.add_argument("--fog-range", type=float, default=4.0, help="fog of war on the dashboard: camera range in m (0 = off)")
    ap.add_argument("--time-limit", type=float, default=80.0, help="patrol: minutes (tour_test.sh kills at 90)")
    ap.add_argument("--building-config", default=str(REPO / "sim/configs/buildings_v1.yaml"),
                    help="generator config (approach distances, robot_radius, min_door_gap) for the round-2 poses")
    a = ap.parse_args()

    b = json.loads(pathlib.Path(a.building + ".building.json").read_text())
    tour = robot.door_tour(b["poses"], a.start)
    if a.max_visits:
        tour = tour[:a.max_visits]
    print(f"{len(tour)} door sides to visit")

    rclpy.init()
    nav = BasicNavigator()
    nav.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)])
    tf = Buffer()
    TransformListener(tf, nav)
    stop_pub = nav.create_publisher(Twist, "/cmd_vel", 1)
    wait_active(nav)   # no AMCL: map -> odom comes from robot.launch.py; own wait with retries (stock one can hang)

    grab = None
    if a.frames:
        from doorsim import stop_frames
        grab = stop_frames.Grabber(a.name)
    n_frame = 0

    bag = None
    if a.bag:
        bag = subprocess.Popen(["ros2", "bag", "record", "-o", a.bag, "--use-sim-time", *BAG_TOPICS])
        time.sleep(2)

    det = None
    if a.hinge_look:
        if not a.frames:
            sys.exit("--hinge-look needs --frames (the detector reads the saved rgb.png)")
        det = subprocess.Popen([str(REPO / "train" / ".venv" / "bin" / "python"), str(REPO / "train" / "detect_server.py"), a.hinge_look],
                               cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        while (line := det.stdout.readline()) and line.strip() != "READY":
            pass
        doors = {d["id"]: {"pos": d["pos"], "yaw": d["yaw"], "width": d["spec"]["width"]} for d in b["doors"]}
        occ = [hinge_look.load_map(a.building + ".map.yaml")]    # list: round 2 swaps in the map with the opened doors
        free = lambda x, y: hinge_look.free(occ[0], x, y, a.look_radius)
        fx, cx = (IMG_W / 2) / math.tan(HFOV / 2), IMG_W / 2

    props, gz_node, door_pubs, known_hinged = None, None, {}, set()
    if a.door_step:
        if det is None:
            sys.exit("--door-step needs --hinge-look (detector) and --frames")
        props = subprocess.Popen([str(REPO / "train" / ".venv" / "bin" / "python"), str(REPO / "train" / "props_server.py"), a.door_step],
                                 cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        while (line := props.stdout.readline()) and line.strip() != "READY":
            pass
        bdoors = {d["id"]: d for d in b["doors"]}
        K = [fx, 0, cx, 0, fx, (IMG_W * 3 / 4) / 2, 0, 0, 1]   # 640 x 480, square pixels
        if not a.no_open:
            from gz.transport13 import Node
            from gz.msgs10.double_pb2 import Double
            gz_node = Node()

    live_path = pathlib.Path(a.out).parent / "live.json" if a.out else None
    live = {"pose": None, "now": None, "opened": [], "reports": []}
    live_t = [0.0]
    fog_map = (occ[0] if det is not None else hinge_look.load_map(a.building + ".map.yaml")) if a.fog_range > 0 else None
    fog_f = fog.new(fog_map) if fog_map is not None else None

    def write_live(force=False):
        """OUT/live.json for the run dashboard (Madelief 2026-11-11: show it in real time): robot pose, current door side,
        opened doors, reports. Written at most every 0.5 s while driving, and on every report / opening."""
        if live_path is None or (not force and time.time() - live_t[0] < 0.5):
            return
        live_t[0] = time.time()
        pos = where(nav, tf)
        if pos:
            live["pose"] = [round(v, 3) for v in pos]
            if fog_f is not None:                              # fog of war: what the camera (0.20 m ahead of the base) has had in view
                x, y, yaw = pos
                fog.look(fog_f, occ[0] if det is not None else fog_map, (x + 0.20 * math.cos(yaw), y + 0.20 * math.sin(yaw)), yaw,
                         HFOV, a.fog_range)
                live["fog"] = fog.encode(fog_f)
        tmp = live_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({**live, "time": time.time()}))
        tmp.replace(live_path)

    def detect(frame_dir):
        """Detector output for a saved frame (also written next to it as dets.json); empty when the detector is off / no frame."""
        if det is None or frame_dir is None:
            return {"boxes": [], "scores": [], "labels": []}
        p = pathlib.Path(a.frames) / frame_dir
        det.stdin.write(str((p / "rgb.png").resolve()) + "\n")
        out = json.loads(det.stdout.readline())
        (p / "dets.json").write_text(json.dumps(out) + "\n")
        return out

    def drive(v, s, tag):
        """Nav2 to stop s, stand still, measure, save the frame (folder NNN_door<id>_<side>_<tag>) -> record dict."""
        nonlocal n_frame
        t0 = time.time()
        nav.goToPose(goal(nav, s["x"], s["y"], s["yaw"]))
        while not nav.isTaskComplete():
            if time.time() - t0 > a.timeout:
                nav.cancelTask()
                break
            rclpy.spin_once(nav, timeout_sec=0.1)
            write_live()
        res = nav.getResult()
        stop_pub.publish(Twist())                    # Gazebo's DiffDrive keeps the last command
        wait_sim(nav, a.look)
        pos = where(nav, tf)
        r = {"dist": s["dist"], "result": res.name if isinstance(res, TaskResult) else str(res),
             "seconds": round(time.time() - t0, 1)}
        if pos:
            dyaw = (pos[2] - s["yaw"] + math.pi) % (2 * math.pi) - math.pi
            r.update(pos_err=round(math.hypot(pos[0] - s["x"], pos[1] - s["y"]), 3), yaw_err=round(abs(dyaw), 3))
        if grab:
            shot = grab.grab(nav.get_clock().now().nanoseconds, timeout=10)
            if shot is None:
                r["frame"] = None
            else:
                cam = where(nav, tf, "camera_optical")
                d = pathlib.Path(a.frames) / f"{n_frame:03d}_door{v['door']}_{v['side']}_{tag}"
                meta = {"building": a.building, "door": v["door"], "side": v["side"], "space": v["space"],
                        "dist": s["dist"], "target": {"x": s["x"], "y": s["y"], "yaw": s["yaw"]}, **r,
                        "robot_pose": pos, "camera_xy_yaw": cam, "stamp_ns": shot["stamp_ns"]}
                stop_frames.save_stop(d, shot["rgb"], shot["depth"], shot["seg"], shot["boxes"], shot["boxes3d"], meta)
                r["frame"] = d.name
                n_frame += 1
        return r, pos, where(nav, tf, "camera_link")   # camera_link: x forward, y left (for the edge column)

    def seen(cam, dets, edges):
        """Hinge near any of the edges in this look (edge column from the camera pose, give or take 0.15 m at that depth)."""
        if not cam:
            return False
        return any(hinge_look.hinge_seen(dets, hinge_look.edge_column(cam[0], cam[1], cam[2], e, fx, cx),
                                         max(0.3, math.hypot(e[0] - cam[0], e[1] - cam[1])), fx) for e in edges)

    def frame_depth(frame):
        f = pathlib.Path(a.frames) / frame / "depth.png" if frame else None
        return np.asarray(Image.open(f), np.float32) / 1000.0 if f and f.exists() else None

    def seen_sides(cam, dets, edge_pts, depth):
        """Image sides whose edge shows a hinge in this look (column + depth + size checks, as score_hinge_look)."""
        out = set()
        if not cam:
            return out
        for sd, e in edge_pts.items():
            c, fwd = hinge_look.edge_column(cam[0], cam[1], cam[2], e, fx, cx), hinge_look.edge_forward(cam[0], cam[1], cam[2], e)
            if c is not None and fwd is not None and hinge_look.hinge_seen(dets, c, fwd, fx, depth_img=depth):
                out.add(sd)
        return out

    def p_slide_of(frame, cam, dets, edge_pts):
        """Door-crop classifier P(slide) for one saved frame (door box picked by the door's edges); None when no usable box."""
        if not frame or not cam:
            return None
        cols = [hinge_look.edge_column(cam[0], cam[1], cam[2], edge_pts[sd], fx, cx) for sd in ("left", "right")]
        box = hinge_look.pick_door_span(dets, cols) if None not in cols else hinge_look.pick_door(dets, IMG_W)
        if box is None:
            return None
        f = (pathlib.Path(a.frames) / frame).resolve()
        props.stdin.write(json.dumps({"rgb": str(f / "rgb.png"), "depth": str(f / "depth.png"), "box": box, "K": K,
                                      "camera": {"world_position": [cam[0], cam[1], 0.40], "world_rpy": [0.0, 0.0, cam[2]]}}) + "\n")
        out = json.loads(props.stdout.readline())
        return out["opening"]["slide"] if "opening" in out else None

    def open_door(v, d):
        """Open door d fully. Hinged, robot on the swing side within reach of the panel -> back off to the far stop first."""
        if d["spec"]["door_type"] == "hinged":
            g = D._layout(d["spec"], B.SLOT)
            hx, hy = B.hinge_xy(d)
            th, sy = B.open_position(d), d["spec"]["hinge_y"]   # panel tip when fully open (same geometry as occupancy())
            tip = B._world(d, g["hinge"][0] + sy * g["pw"] * math.sin(th), g["hinge"][1] - sy * g["pw"] * math.cos(th))
            pos = where(nav, tf)
            nx, ny = math.cos(d["yaw"]), math.sin(d["yaw"])
            same = pos and ((pos[0] - d["pos"][0]) * nx + (pos[1] - d["pos"][1]) * ny) * ((tip[0] - d["pos"][0]) * nx + (tip[1] - d["pos"][1]) * ny) > 0
            if same and math.hypot(pos[0] - hx, pos[1] - hy) < g["pw"] + 0.35:
                drive(v, v["stops"][0], "backoff")
        topic = B.door_topic(d)
        if topic not in door_pubs:
            door_pubs[topic] = gz_node.advertise(topic, Double)
        msg = Double()
        msg.data = float(B.open_position(d))
        for _ in range(10):                                    # the controller's subscription can take a moment to connect
            door_pubs[topic].publish(msg)
            wait_sim(nav, 0.2)
        wait_sim(nav, 2.0)
        live["opened"].append(d["id"])
        write_live(force=True)
        pos = where(nav, tf)
        if pos:                                                # proof frame: the door after opening, from where the robot stands
            drive(v, {"x": pos[0], "y": pos[1], "yaw": pos[2], "dist": None}, "opened")

    def door_step_visit(v, shots):
        """Reactive door step for one door side (see the module docstring) -> log dict for tour.json."""
        real = v["real"] if "real" in v else v["door"]          # patrol: the real door it matched (None = no real door there)
        d = bdoors.get(real) if real is not None else None
        door = v["est_door"] if "est_door" in v else doors[v["door"]]
        yaw_ref = v["stops"][-1]["yaw"]
        edge_pts = {sd: hinge_look.edge_point(door, sd, yaw_ref) for sd in ("left", "right")}
        guess, stop_looks, ps = None, [], []
        for r, pos, cam, dd in shots:
            if r["result"] != "SUCCEEDED" or not r.get("frame"):
                continue
            depth = frame_depth(r["frame"])
            if guess is None and cam:
                cols = [hinge_look.edge_column(cam[0], cam[1], cam[2], edge_pts[sd], fx, cx) for sd in ("left", "right")]
                box = hinge_look.pick_door_span(dd, cols) if None not in cols else hinge_look.pick_door(dd, IMG_W)
                expected = hinge_look.edge_forward(cam[0], cam[1], cam[2], tuple(d["pos"]))
                guess = hinge_look.hinge_side_image(box, dd, depth_img=depth, expected=expected, fx=fx) if box else None
            stop_looks.append({"edge": None, "angle": None, "seen": seen_sides(cam, dd, edge_pts, depth), "frame": r["frame"]})
            p = p_slide_of(r["frame"], cam, dd, edge_pts)
            if p is not None:
                ps.append(p)
        p_mean = sum(ps) / len(ps) if ps else None
        # door_step.answer uses 0.5: rescale so --p-slide-t maps to 0.5
        p_use = None if p_mean is None else (0.5 * p_mean / a.p_slide_t if p_mean < a.p_slide_t else 0.5 + 0.5 * (p_mean - a.p_slide_t) / (1 - a.p_slide_t))
        near = v["stops"][-1]
        cache = {}

        def pose_for(edge, angle):
            if (edge, angle) not in cache:
                ps_ = hinge_look.look_poses(door, edge_pts[edge], (near["x"], near["y"]), free, dist=a.look_dist, angles=(angle,))
                cache[edge, angle] = ps_[0] if ps_ else None
            return cache[edge, angle]

        def look_fn(edge, angle):
            p = pose_for(edge, angle)
            r, pos, cam = drive(v, p, f"step_{edge}_{angle}")
            print(f"  step look door {v['door']} {edge} {angle} deg: {r['result']}")
            if r["result"] != "SUCCEEDED" or not r.get("frame"):
                return None
            return seen_sides(cam, detect(r["frame"]), edge_pts, frame_depth(r["frame"]))

        tr = door_step.truth(d["spec"]["door_type"], edge_pts, B.hinge_xy(d)) if d else ("none", None)
        opened = []

        def check_fn(rep):
            ok = door_step.is_right(rep, tr)
            live["reports"].append([v["door"], v["side"], rep["type"], rep["hinge_side"], ok])
            write_live(force=True)
            print(f"  report door {v['door']} {v['side']}: {rep['type']} {rep['hinge_side'] or ''} (sure {rep['sure']}) -> {'right' if ok else 'wrong'}")
            if ok and gz_node is not None and d is not None:
                open_door(v, d)
                opened.append(True)
            return ok

        if not stop_looks:
            return None
        log_ = door_step.visit([{k: l[k] for k in ("edge", "angle", "seen")} for l in stop_looks], look_fn, check_fn, guess,
                               lambda e, ang: pose_for(e, ang) is not None, n_max=a.step_n, p_slide=p_use,
                               other_side_hinged=real is not None and real in known_hinged)
        if any(r["right"] and r["type"] == "hinged" for r in log_["reports"]) or \
                any(r["sure"] and r["type"] == "hinged" for r in log_["reports"]):
            if real is not None:
                known_hinged.add(real)
        for l in log_["looks"]:
            l["seen"] = sorted(l["seen"])
        return {**log_, "handle_guess": guess, "p_slide": p_mean, "p_slide_frames": ps, "truth": list(tr), "opened": bool(opened),
                "stop_frames": [l["frame"] for l in stop_looks]}

    def load_map_srv(yaml_path, tries=3, per_call=10.0):
        """Nav2 map_server load_map (with timeouts + retries, as wait_active) -> True when loaded."""
        from nav2_msgs.srv import LoadMap
        cli = nav.create_client(LoadMap, "/map_server/load_map")
        for _ in range(tries):
            if cli.wait_for_service(timeout_sec=per_call):
                req = LoadMap.Request()
                req.map_url = str(pathlib.Path(yaml_path).resolve())
                fut = cli.call_async(req)
                rclpy.spin_until_future_complete(nav, fut, timeout_sec=per_call)
                if fut.done() and fut.result() is not None and fut.result().result == LoadMap.Response.RESULT_SUCCESS:
                    nav.destroy_client(cli)
                    return True
        nav.destroy_client(cli)
        return False

    def map_with_opened():
        """Write + load (Nav2) the map with the doors opened so far; the look-spot check uses it too. -> (ok, new building)."""
        import yaml
        cfg = yaml.safe_load(pathlib.Path(a.building_config).read_text())
        nb = B.after_opening(b, live["opened"], cfg)
        stem = pathlib.Path(a.out).parent / f"opened_{len(live['opened'])}"
        B.write_map(nb, stem)
        ok = load_map_srv(f"{stem}.map.yaml")
        if ok:
            wait_sim(nav, 3.0)                                 # costmaps pick up the new /map
            occ[0] = hinge_look.load_map(f"{stem}.map.yaml")
        return ok, nb

    patrol_log = {"viewpoints": [], "candidates": []}

    def patrol_visits():
        """Run 3 (no markers): viewpoints over the reachable floor map, 5 views each; door boxes -> candidates on the map; every
        new candidate (nearest first) -> a door step visit. Out of viewpoints -> if doors were opened, load that map and go on."""
        t_end = time.time() + a.time_limit * 60
        cands, handled, done_vp, n, opened_at_map = patrol_log["candidates"], set(), set(), 0, 0
        while time.time() < t_end:
            pos = where(nav, tf)
            here = (pos[0], pos[1]) if pos else tuple(a.start)
            vps = [q for q in patrol.viewpoints(occ[0], here, a.look_radius, a.vp_spacing) if (round(q[0], 2), round(q[1], 2)) not in done_vp]
            live["vps"] = [[round(q[0], 2), round(q[1], 2)] for q in vps]
            write_live(force=True)
            if not vps:
                if len(live["opened"]) > opened_at_map:
                    opened_at_map = len(live["opened"])
                    ok, _ = map_with_opened()
                    print(f"patrol: map with {opened_at_map} opened doors loaded: {ok}")
                    if ok:
                        continue
                break
            q = vps[0]
            done_vp.add((round(q[0], 2), round(q[1], 2)))
            live["vp_done"] = live.get("vp_done", []) + [[round(q[0], 2), round(q[1], 2)]]
            pv, seen_n = {"door": "vp", "side": f"{q[0]:.1f}_{q[1]:.1f}"}, 0
            print(f"patrol: viewpoint {q[0]:.1f} {q[1]:.1f} ({len(vps) - 1} left), {len(cands)} candidates so far")
            for yaw in patrol.view_yaws(5, pos[2] if pos else 0.0):
                if time.time() > t_end:
                    break
                r, _, cam = drive(pv, {"x": q[0], "y": q[1], "yaw": yaw, "dist": None}, "look")
                if r["result"] != "SUCCEEDED":
                    break                                      # can't get to this viewpoint: next one
                if not r.get("frame") or not cam:
                    continue
                dd, depth = detect(r["frame"]), frame_depth(r["frame"])
                for box, sc, lab in zip(dd["boxes"], dd["scores"], dd["labels"]):
                    if lab != "door" or sc < 0.5 or depth is None:
                        continue
                    e = patrol.door_estimate(box, depth, cam, fx, cx)
                    if not e or e["depth"] > a.max_door_depth:
                        continue
                    sn = patrol.snap_to_wall(occ[0], e["x"], e["y"], cam[:2])
                    if sn:
                        patrol.add_candidate(cands, {"x": sn[0], "y": sn[1], "yaw": sn[2], "width": e["width"]})
                        seen_n += 1
            patrol_log["viewpoints"].append({"xy": list(q), "door_boxes_used": seen_n})
            for k in sorted((k for k in range(len(cands)) if k not in handled),
                            key=lambda k: math.hypot(cands[k]["x"] - here[0], cands[k]["y"] - here[1])):
                handled.add(k)
                c = cands[k]
                m_ = patrol.match(c, b["doors"])
                c["match"] = list(m_) if m_ else None
                stops = [st for st in patrol.approach(c) if free(st["x"], st["y"])]
                if not stops:
                    c["status"] = "no free stop"
                    continue
                n += 1
                est = {"pos": [c["x"], c["y"]], "yaw": c["yaw"], "width": max(0.6, min(1.2, c["width"]))}
                yield 1, n, n, {"door": m_[0] if m_ else f"c{k}", "side": m_[1] if m_ else "x", "space": None, "stops": stops,
                                "real": m_[0] if m_ else None, "est_door": est, "cand": k}
                last = log["visits"][-1] if log["visits"] else {}
                st = last.get("door_step") or {}
                c["status"] = ("false door" if not m_ else "right" if st.get("right") else "wrong" if st else "not reached")
                live["cands"] = [[round(x["x"], 2), round(x["y"], 2), round(x["yaw"], 2), x.get("status", "new")] for x in cands]
                write_live(force=True)
        print(f"patrol done: {len(patrol_log['viewpoints'])} viewpoints, {len(cands)} candidates")

    def round_two():
        """Doors opened in round 1 -> new map (loaded into Nav2) + the door sides that became reachable, in tour order."""
        import yaml
        if not live["opened"] or not a.out:
            return []
        cfg = yaml.safe_load(pathlib.Path(a.building_config).read_text())
        nb = B.after_opening(b, live["opened"], cfg)
        stem = pathlib.Path(a.out).parent / "round2"
        B.write_map(nb, stem)
        ok = load_map_srv(f"{stem}.map.yaml")
        print(f"round 2: {len(live['opened'])} doors opened, map with them open loaded: {ok}")
        if not ok:
            return []
        wait_sim(nav, 3.0)                                     # costmaps pick up the new /map
        occ[0] = hinge_look.load_map(f"{stem}.map.yaml")
        done = {(x["door"], x["side"]) for x in log["visits"]}
        pos = where(nav, tf)
        here = (pos[0], pos[1]) if pos else tuple(a.start)
        t2 = robot.door_tour([q for q in nb["poses"] if (q["door"], q["side"]) not in done], here)
        live["round"], live["plan2"] = 2, [[x["door"], x["side"], x["stops"][0]["x"], x["stops"][0]["y"]] for x in t2]
        write_live(force=True)
        print(f"round 2: {len(t2)} new door sides to visit")
        return t2

    def all_visits():
        """(round, i, n, visit): the tour, then (--rounds 2) round 2, planned only once round 1 is done. --patrol: run 3 instead."""
        if a.patrol:
            yield from patrol_visits()
            return
        yield from ((1, i, len(tour), v) for i, v in enumerate(tour, 1))
        if a.rounds > 1 and props is not None:
            t2 = round_two()
            yield from ((2, i, len(t2), v) for i, v in enumerate(t2, 1))

    log = {"building": b.get("seed"), "start": a.start, "visits": []}
    try:
        for rnd, i, n_, v in all_visits():
            if a.patrol:                                       # patrol: progress = viewpoints (dashboard reads live.json)
                print(f"candidate {v['cand']}: door {v['door']} {v['side']}")
            else:
                print(f"[{i}/{n_}] door {v['door']} {v['side']}" + (" (round 2)" if rnd > 1 else ""))   # progress counter for the run dashboard
            live["now"] = [v["door"], v["side"]]
            write_live(force=True)
            rec = {"door": v["door"], "side": v["side"], "space": v["space"], "round": rnd, "stops": []}
            if "cand" in v:
                rec["cand"] = v["cand"]
            shots = []
            for s in v["stops"]:
                r, pos, cam = drive(v, s, f"{s['dist']:.1f}")
                shots.append((r, pos, cam, detect(r.get("frame"))))
                rec["stops"].append(r)
                print(f"door {v['door']} {v['side']} @ {s['dist']:.1f} m: {r}")
            if props is not None and any(r["result"] == "SUCCEEDED" for r, *_ in shots):
                rec["door_step"] = door_step_visit(v, shots)
                ds_ = rec["door_step"]
                if ds_:
                    print(f"door {v['door']} {v['side']} door step: first {'right' if ds_['first_right'] else 'wrong'}, "
                          f"{ds_['n_reports']} report(s), opened {ds_['opened']}, P(slide) {ds_['p_slide']}")
            elif det is not None and any(r["result"] == "SUCCEEDED" for r, *_ in shots):   # never got there: nothing to look at
                # hinge look (option A, Madelief 2026-11-10): handle -> hinge edge (far stop first: from the 1 m stop the 0.4 m
                # camera sees only up to ~1 m, where handles sit), then extra looks aimed at that edge; no handle -> both edges
                door, guess, yaw = doors[v["door"]], None, None
                for r, pos, cam, dd in shots:
                    # door box by the door's known edges + handle at the door's depth (Claude 2026-11-10: wrong guesses 5 -> 2 offline)
                    box, depth, expected = None, None, None
                    if cam:
                        cols = [hinge_look.edge_column(cam[0], cam[1], cam[2], hinge_look.edge_point(door, sd, cam[2]), fx, cx)
                                for sd in ("left", "right")]
                        expected = hinge_look.edge_forward(cam[0], cam[1], cam[2], tuple(door["pos"]))
                        if None not in cols:
                            box = hinge_look.pick_door_span(dd, cols)
                    if box is None and not (cam and None not in cols):
                        box = hinge_look.pick_door(dd, IMG_W)
                    if r.get("frame") and (pathlib.Path(a.frames) / r["frame"] / "depth.png").exists():
                        depth = np.asarray(Image.open(pathlib.Path(a.frames) / r["frame"] / "depth.png"), np.float32) / 1000.0
                    # size checks are Claude 2026-11-10 decisions (real hinge boxes <= 0.106 m tall, handles >= 0.033 m in the 7-building reruns)
                    side = hinge_look.hinge_side_image(box, dd, depth_img=depth, expected=expected, fx=fx) if box else None
                    if side and pos:
                        guess, yaw = side, pos[2]
                        break
                if guess:
                    edges = [hinge_look.edge_point(door, guess, yaw)]
                else:
                    edges = [hinge_look.edge_point(door, sd, v["stops"][-1]["yaw"]) for sd in ("left", "right")]
                looks = [{"frame": r.get("frame"), "kind": "stop", "dist": r["dist"], "result": r["result"], "seen": seen(cam, dd, edges)}
                         for r, pos, cam, dd in shots]
                near = v["stops"][-1]
                for k, e in enumerate(edges):
                    for p in hinge_look.look_poses(door, e, (near["x"], near["y"]), free, dist=a.look_dist):
                        r, pos, cam = drive(v, p, f"look{k}_{p['angle']}")
                        dd = detect(r.get("frame"))
                        looks.append({"frame": r.get("frame"), "kind": "look", "angle": p["angle"], "dist": p["dist"],
                                      "toward": p["toward"], "result": r["result"], "seen": seen(cam, dd, [e])})
                        print(f"  look door {v['door']} edge {k} {p['angle']} deg: {r['result']}, hinge seen {looks[-1]['seen']}")
                rec["hinge_look"] = {"guess": guess, "edges": [list(e) for e in edges], "looks": looks,
                                     "found": any(l["seen"] for l in looks)}
                print(f"door {v['door']} {v['side']} hinge look: guess {guess}, found {rec['hinge_look']['found']}")
            log["visits"].append(rec)
    finally:
        stop_pub.publish(Twist())
        if det is not None:
            det.stdin.close()
            det.wait(timeout=30)
        if props is not None:
            props.stdin.close()
            props.wait(timeout=30)
        if bag:
            bag.send_signal(signal.SIGINT)
            bag.wait(timeout=30)
        if a.out:
            pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            if a.patrol:
                log["patrol"] = patrol_log
            pathlib.Path(a.out).write_text(json.dumps(log, indent=1))
        n = sum(r["result"] == "SUCCEEDED" for v in log["visits"] for r in v["stops"])
        print(f"reached {n} / {sum(len(v['stops']) for v in log['visits'])} stops")
        nav.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
