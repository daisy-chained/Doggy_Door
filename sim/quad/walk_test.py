"""Headless Quad-SDK Spot walk test: does Spot get to the goal (flat ground, or up a Dutch staircase)?

    source ~/git/quad_ws/env.sh
    python3 sim/quad/walk_test.py --world dutch_stairs_20.sdf --top 2.6 --out runs/quad/walk_stairs_20

Follows Quad-SDK's first-run tutorial (docs/tutorials/first-run.md): Gazebo + Spot (inverse_dynamics controller) -> stand
(control/mode 1) -> planning stack with the global body planner (default goal x=5, y=0, global_body_planner.yaml) -> walk
(control/mode 2). Logs Spot's ground-truth body pose at 10 Hz until it reaches the goal, falls or times out.
Result (result.json): reached (x >= goal_x - 0.5 and body z >= top + 0.25), fell (|roll| > 0.8 rad or body z < top_here + 0.15),
max x / z, time. Own GZ_PARTITION + ROS_DOMAIN_ID so it can't talk to another Gazebo."""
import argparse, json, math, os, pathlib, signal, subprocess, time

SPOT = '[{"name": "robot_1", "type": "spot", "controller": "inverse_dynamics", "init_pose": "-x 0.0 -y 0.0 -z 0.7"}]'
PLAN = '[{"name": "robot_1", "type": "spot", "controller_mode": "inverse_dynamics", "reference": "gbpl", "twist_input": "none"}]'


def rpy(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch


def ground_z(x, top, run_up=1.5, going=0.22, steps=13):
    """Floor height under body x for the stairs world (0 on flat)."""
    if top <= 0 or x < run_up:
        return 0.0
    return min(top, (math.floor((x - run_up) / going) + 1) * top / steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="flat.sdf")
    ap.add_argument("--top", type=float, default=0.0, help="landing height (0 = flat)")
    ap.add_argument("--goal-x", type=float, default=5.0)
    ap.add_argument("--timeout", type=float, default=300.0, help="wall seconds after the walk command")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GZ_PARTITION="quadtest", ROS_DOMAIN_ID="37")
    os.environ.update(GZ_PARTITION="quadtest", ROS_DOMAIN_ID="37")

    import rclpy
    from std_msgs.msg import UInt8
    from quad_msgs.msg import RobotState
    procs = []

    def launch(args, log):
        p = subprocess.Popen(["ros2", "launch", "quad_utils", *args], env=env, stdout=open(out / log, "w"),
                             stderr=subprocess.STDOUT, start_new_session=True)
        procs.append(p); return p

    rclpy.init()
    node = rclpy.create_node("doggy_walk_test")
    trace, last = [], {}
    def on_state(m):
        p = m.body.pose
        last.update(x=p.position.x, y=p.position.y, z=p.position.z, q=p.orientation)
    node.create_subscription(RobotState, "/robot_1/state/ground_truth", on_state, 10)
    mode = node.create_publisher(UInt8, "/robot_1/control/mode", 10)

    def spin_for(sec):
        t_end = time.time() + sec
        while time.time() < t_end:
            rclpy.spin_once(node, timeout_sec=0.05)

    result = {"world": a.world, "top": a.top, "goal_x": a.goal_x, "reached": False, "fell": False, "stage": "start"}
    try:
        launch(["quad_gazebo.py", f"world:={a.world}", "gui:=false", "rviz:=false", f"robot_configs:={SPOT}"], "gazebo.log")
        t0 = time.time()
        while not last and time.time() - t0 < 180:
            spin_for(0.5)
        if not last:
            result["stage"] = "no ground truth after 180 s"; return
        spin_for(5)
        mode.publish(UInt8(data=1)); result["stage"] = "stand"
        spin_for(10)
        result["stand_z"] = last["z"]
        launch(["quad_plan.py", f"robot_configs:={PLAN}"], "plan.log")
        spin_for(15)
        mode.publish(UInt8(data=2)); result["stage"] = "walk"
        t_walk = time.time()
        while time.time() - t_walk < a.timeout:
            spin_for(0.1)
            r, pch = rpy(last["q"])
            row = {"t": round(time.time() - t_walk, 2), "x": last["x"], "y": last["y"], "z": last["z"], "roll": r, "pitch": pch}
            trace.append(row)
            if abs(r) > 0.8 or last["z"] < ground_z(last["x"], a.top) + 0.15:
                result["fell"] = True; break
            if last["x"] >= a.goal_x - 0.5 and last["z"] >= a.top + 0.25:
                result["reached"] = True; break
        result["time_s"] = round(time.time() - t_walk, 1)
        result["stage"] = "done"
    finally:
        if trace:
            result["max_x"] = max(r["x"] for r in trace); result["max_z"] = max(r["z"] for r in trace)
            result["last"] = trace[-1]
        (out / "trace.json").write_text(json.dumps(trace) + "\n")
        (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result))
        for p in procs[::-1]:
            try:
                os.killpg(p.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        time.sleep(8)
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
