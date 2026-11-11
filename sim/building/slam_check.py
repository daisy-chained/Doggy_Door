"""SLAM smoke test (run 4 groundwork, Madelief 2026-11-11 ~13:45: "start building slam rn, wait for 3 to work before we integrate slam").

    SLAM=1 TOUR_SCRIPT=sim/building/slam_check.py bash sim/building/tour_test.sh sim/building/out/bld_train_007 \\
        sim/configs/robots/wheeled_low_fast.yaml 2.5 5.333

Not the run-4 patrol: the robot drives the run-3 viewpoint route (taken from the TRUE floor map, for this check only) and turns in
place 5 times at each viewpoint, while slam_toolbox builds /map from the lidar. At the end the SLAM map is saved (OUT/slam_map.*) and
scored against the true floor map (doorsim.slam_eval.compare: coverage of the true free floor, walls found, false walls).
Odometry is still Gazebo's true pose, so this checks mapping, not drift. Same arguments as door_tour.py (extra ones are ignored).
Writes OUT/slam_check.json and OUT/live.json (robot pose for the run dashboard)."""
import argparse, json, math, pathlib, subprocess, sys, time

import rclpy
from geometry_msgs.msg import Twist
from nav2_simple_commander.robot_navigator import BasicNavigator
from tf2_ros import Buffer, TransformListener

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from door_tour import goal, where, wait_sim, wait_active  # noqa: E402
from doorsim import hinge_look, patrol, slam_eval  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("building")
    ap.add_argument("--start", type=float, nargs=2, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames")
    ap.add_argument("--spacing", type=float, default=1.5)
    ap.add_argument("--radius", type=float, default=0.33)
    ap.add_argument("--timeout", type=float, default=60.0, help="wall seconds per goal")
    ap.add_argument("--time-limit", type=float, default=40.0, help="minutes")
    a, _ = ap.parse_known_args()
    out = pathlib.Path(a.out).parent
    out.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    nav = BasicNavigator()
    tf = Buffer()
    TransformListener(tf, nav)
    stop_pub = nav.create_publisher(Twist, "/cmd_vel", 1)
    wait_active(nav, nodes=("bt_navigator",))          # no map_server with SLAM
    true_m = hinge_look.load_map(a.building + ".map.yaml")
    vps = patrol.viewpoints(true_m, tuple(a.start), a.radius, a.spacing)
    print(f"slam check: {len(vps)} viewpoints (from the true map, this check only)")
    log, t_end, t0 = {"viewpoints": []}, time.time() + a.time_limit * 60, time.time()
    live = out / "live.json"

    def go(x, y, yaw):
        t = time.time()
        nav.goToPose(goal(nav, x, y, yaw))
        while not nav.isTaskComplete():
            if time.time() - t > a.timeout:
                nav.cancelTask()
                break
            rclpy.spin_once(nav, timeout_sec=0.1)
            pos = where(nav, tf)
            if pos:
                live.write_text(json.dumps({"pose": [round(v, 3) for v in pos], "time": time.time()}))
        stop_pub.publish(Twist())
        return str(nav.getResult().name if hasattr(nav.getResult(), "name") else nav.getResult()), round(time.time() - t, 1)

    try:
        for k, (x, y) in enumerate(vps):
            if time.time() > t_end:
                print("slam check: time limit")
                break
            res = []
            for yaw in patrol.view_yaws(5, 0.0):
                r, s = go(x, y, yaw)
                res.append([r, s])
                if r != "SUCCEEDED":
                    break
            wait_sim(nav, 0.5)
            print(f"slam check: viewpoint {k + 1} / {len(vps)} ({x:.1f}, {y:.1f}): {[r for r, _ in res]}")
            log["viewpoints"].append({"xy": [x, y], "results": res})
    finally:
        stop_pub.publish(Twist())
        stem = out / "slam_map"
        subprocess.run(["ros2", "run", "nav2_map_server", "map_saver_cli", "-f", str(stem), "--ros-args", "-p", "use_sim_time:=true"],
                       timeout=60, capture_output=True)
        if pathlib.Path(f"{stem}.yaml").exists():
            score = slam_eval.compare(slam_eval.load(a.building + ".map.yaml"), slam_eval.load(f"{stem}.yaml"))
            log["score"] = score
            print("slam check score: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in score.items()))
        else:
            print("slam check: map not saved")
        log["minutes"] = round((time.time() - t0) / 60, 1)
        (out / "slam_check.json").write_text(json.dumps(log, indent=1))
        try:
            nav.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
