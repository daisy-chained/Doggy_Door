"""Drive the robot past every reachable door side of a building (phase 2e; Madelief 2026-11-07: Nav2 + waypoints over the
approach poses, at a door stop ~2.5 m and look, then ~1 m, report, move on). Needs robot.launch.py + nav.launch.py running.

    source /opt/ros/jazzy/setup.bash
    python3 sim/building/door_tour.py sim/building/out/bld_train_000 --start 1.0 4.04 [--look 2.0] [--bag runs/tours/bld_train_000]
                                      [--max-visits N] [--out runs/tours/bld_train_000.tour.json]

Order = doorsim.robot.door_tour (greedy nearest next door side, far stop then near stop). Per stop it logs whether Nav2 reached it,
the time it took and the final position / heading error (from tf map -> base_link). Perception at the stops comes later; for now the
robot only stops for --look seconds (sim time) so a recording has frames of every door from both distances.
--bag records the run with rosbag2 (camera, depth, boxes, lidar, odom, tf), for "record first, score after".
"""
import argparse, json, math, pathlib, signal, subprocess, sys, time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from tf2_ros import Buffer, TransformListener

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import robot  # noqa: E402

BAG_TOPICS = ["/rgbd/image", "/rgbd/depth_image", "/rgbd/camera_info", "/scan", "/odom", "/tf", "/tf_static", "/clock",
              "/cmd_vel", "/joint_states"]


def goal(nav, x, y, yaw):
    g = PoseStamped()
    g.header.frame_id = "map"
    g.header.stamp = nav.get_clock().now().to_msg()
    g.pose.position.x, g.pose.position.y = float(x), float(y)
    g.pose.orientation.z, g.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return g


def where(nav, tf):
    try:
        t = tf.lookup_transform("map", "base_link", rclpy.time.Time()).transform
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

    bag = None
    if a.bag:
        bag = subprocess.Popen(["ros2", "bag", "record", "-o", a.bag, "--use-sim-time", *BAG_TOPICS])
        time.sleep(2)

    log = {"building": b.get("seed"), "start": a.start, "visits": []}
    try:
        for v in tour:
            rec = {"door": v["door"], "side": v["side"], "space": v["space"], "stops": []}
            for s in v["stops"]:
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
                rec["stops"].append(r)
                print(f"door {v['door']} {v['side']} @ {s['dist']:.1f} m: {r}")
            log["visits"].append(rec)
    finally:
        stop_pub.publish(Twist())
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
