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
from doorsim import hinge_look, robot  # noqa: E402

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
    nav.waitUntilNav2Active(localizer="map_server")   # no AMCL: map -> odom comes from robot.launch.py

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
        occ = hinge_look.load_map(a.building + ".map.yaml")
        free = lambda x, y: hinge_look.free(occ, x, y, a.look_radius)
        fx, cx = (IMG_W / 2) / math.tan(HFOV / 2), IMG_W / 2

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

    log = {"building": b.get("seed"), "start": a.start, "visits": []}
    try:
        for v in tour:
            rec = {"door": v["door"], "side": v["side"], "space": v["space"], "stops": []}
            shots = []
            for s in v["stops"]:
                r, pos, cam = drive(v, s, f"{s['dist']:.1f}")
                shots.append((r, pos, cam, detect(r.get("frame"))))
                rec["stops"].append(r)
                print(f"door {v['door']} {v['side']} @ {s['dist']:.1f} m: {r}")
            if det is not None and any(r["result"] == "SUCCEEDED" for r, *_ in shots):   # never got there: nothing to look at
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
                    side = hinge_look.hinge_side_image(box, dd, depth_img=depth, expected=expected) if box else None
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
        if bag:
            bag.send_signal(signal.SIGINT)
            bag.wait(timeout=30)
        if a.out:
            pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(a.out).write_text(json.dumps(log, indent=1))
        n = sum(r["result"] == "SUCCEEDED" for v in log["visits"] for r in v["stops"])
        print(f"reached {n} / {sum(len(v['stops']) for v in log['visits'])} stops")
        nav.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
