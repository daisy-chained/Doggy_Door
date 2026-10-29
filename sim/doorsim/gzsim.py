"""Run and drive a headless Gazebo world from Python over gz-transport (no ROS bridge).

The world starts paused. Every step request also says pause=True: a WorldControl without it un-pauses
the world. Sensors render on sim time, so after moving something, step at least one sensor period and
wait for messages stamped after the move."""
import math, os, signal, subprocess, threading, time
import numpy as np
from gz.transport13 import Node
from gz.msgs10.annotated_axis_aligned_2d_box_v_pb2 import AnnotatedAxisAligned2DBox_V
from gz.msgs10.annotated_oriented_3d_box_v_pb2 import AnnotatedOriented3DBox_V
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.camera_info_pb2 import CameraInfo
from gz.msgs10.clock_pb2 import Clock
from gz.msgs10.double_pb2 import Double
from gz.msgs10.empty_pb2 import Empty
from gz.msgs10.image_pb2 import Image
from gz.msgs10.light_pb2 import Light
from gz.msgs10.model_pb2 import Model
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.world_control_pb2 import WorldControl

SENSORS = {"rgb": ("/rgbd/image", Image), "depth": ("/rgbd/depth_image", Image), "seg": ("/seg/labels_map", Image),
           "boxes": ("/boxes", AnnotatedAxisAligned2DBox_V), "boxes3d": ("/boxes3d", AnnotatedOriented3DBox_V)}


def ns(stamp):
    return stamp.sec * 10**9 + stamp.nsec


def quat_from_rpy(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (f(a / 2) for a in (roll, pitch, yaw) for f in (math.cos, math.sin))
    return {"x": sr*cp*cy - cr*sp*sy, "y": cr*sp*cy + sr*cp*sy, "z": cr*cp*sy - sr*sp*cy, "w": cr*cp*cy + sr*sp*sy}


def image_array(msg, kind):
    if kind == "depth":
        return np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width).copy()
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
    return a[..., 2].copy() if kind == "seg" else a[..., :3].copy()  # seg labels_map: channel 2 = label (object id)


class Sim:
    def __init__(self, sdf_path, world_name, attrs, log_path):
        self.world, self.attrs = world_name, attrs
        self.proc = subprocess.Popen(["gz", "sim", "-s", "--headless-rendering", str(sdf_path)],
                                     stdout=open(log_path, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        self.node, self.latest, self.lock = Node(), {}, threading.Lock()
        self._sub(Clock, f"/world/{world_name}/clock", "clock")
        self._sub(Pose_V, f"/world/{world_name}/pose/info", "poses")
        self._sub(CameraInfo, "/rgbd/camera_info", "camera_info")
        for key, (topic, typ) in SENSORS.items():
            self._sub(typ, topic, key)
        self.doors = {n: o for n, o in attrs["objects"].items() if "joint" in o}
        for n in self.doors:
            self._sub(Model, f"/world/{world_name}/model/{n}/joint_state", "js_" + n)
        self.door_pub = {n: self.node.advertise(f"/{n}/cmd_pos", Double) for n in self.doors}
        self.panes = {n: o for n, o in attrs["objects"].items() if o.get("part") == "glass_pane"}
        self.pane_pub = {n: (self.node.advertise(o["detach_topic"], Empty), self.node.advertise(o["attach_topic"], Empty))
                         for n, o in self.panes.items()}
        self.light_pub = self.node.advertise(f"/world/{world_name}/light_config", Light)
        for _ in range(120):  # server start + rendering init
            if self._control():
                break
            time.sleep(0.5)
        else:
            self.stop(); raise RuntimeError("Gazebo server did not answer")
        time.sleep(1.0)  # let publishers get matched to subscribers
        self.step(200)
        self.wait_for(["rgb"], 0, timeout=60)  # first render can take a while

    def _sub(self, typ, topic, key):
        def cb(m):
            with self.lock:
                self.latest[key] = m
        self.node.subscribe(typ, topic, cb)

    def _control(self, steps=0):
        # Gazebo never answers a multi_step request (the steps still run), so don't wait long for it
        req = WorldControl(pause=True, multi_step=steps) if steps else WorldControl(pause=True)
        return self.node.request(f"/world/{self.world}/control", req, WorldControl, Boolean, 50 if steps else 5000)[0]

    def now(self):
        with self.lock:
            c = self.latest.get("clock")
        return ns(c.sim) if c else 0

    def step(self, steps):
        """Run `steps` physics steps (1 ms each) and wait until the clock has moved that far."""
        target = self.now() + steps * 10**6
        self._control(steps)  # no reply comes (see _control), so judge by the clock
        t = time.time()
        while self.now() < target:
            if time.time() - t > 60:
                raise TimeoutError("sim clock did not advance")
            time.sleep(0.01)

    def wait_for(self, keys, after_ns, timeout=20):
        """Latest messages for `keys`, all stamped after after_ns and with the same stamp (one render)."""
        t = time.time()
        while time.time() - t < timeout:
            with self.lock:
                msgs = {k: self.latest.get(k) for k in keys}
            if all(m is not None for m in msgs.values()):
                stamps = {ns(m.header.stamp) for m in msgs.values()}
                if len(stamps) == 1 and min(stamps) > after_ns:
                    return msgs
            time.sleep(0.02)
        raise TimeoutError(f"no matching sensor messages for {keys}")

    def render(self, keys, steps=100):
        """Step one sensor period (10 Hz sensors = 100 steps) and return fresh messages for keys."""
        t0 = self.now()
        self.step(steps)
        return self.wait_for(keys, t0)

    def set_pose(self, name, position, rpy=None, orientation=None):
        q = orientation if orientation is not None else quat_from_rpy(*(rpy or (0, 0, 0)))
        req = Pose(name=name, position={"x": position[0], "y": position[1], "z": position[2]}, orientation=q)
        # while paused the reply only comes with the next step, so don't wait for it: the next step applies the pose
        self.node.request(f"/world/{self.world}/set_pose", req, Pose, Boolean, 50)

    def model_pose(self, name):
        with self.lock:
            poses = self.latest["poses"]
        p = next(q for q in poses.pose if q.name == name)
        return [p.position.x, p.position.y, p.position.z], {"x": p.orientation.x, "y": p.orientation.y, "z": p.orientation.z, "w": p.orientation.w}

    def joint(self, door):
        """(position, velocity) of the door's joint from its joint_state topic."""
        with self.lock:
            m = self.latest.get("js_" + door)
        j = next(j for j in m.joint if j.name == self.doors[door]["joint"]["name"])
        return j.axis1.position, j.axis1.velocity

    def set_doors(self, targets, max_steps=6000, chunk=500, still=0.01):
        """Command door joint positions and step until every door is near its target and still."""
        for n, v in targets.items():
            self.door_pub[n].publish(Double(data=v))
        done = 0
        while done < max_steps:
            self.step(chunk); done += chunk
            js = {n: self.joint(n) for n in targets}
            if all(abs(p - targets[n]) < 0.02 and abs(v) < still for n, (p, v) in js.items()):
                break
        return {n: self.joint(n)[0] for n in targets}

    def set_light(self, name, cfg, diffuse):
        kind = {"point": Light.POINT, "directional": Light.DIRECTIONAL, "spot": Light.SPOT}[cfg["type"]]
        msg = Light(name=name, type=kind, diffuse={"r": diffuse[0], "g": diffuse[1], "b": diffuse[2], "a": 1},
                    specular={"r": 0.1, "g": 0.1, "b": 0.1, "a": 1}, cast_shadows=cfg.get("cast_shadows", True), intensity=1.0)
        if "range" in cfg:
            msg.range = cfg["range"]
        if "attenuation" in cfg:
            msg.attenuation_constant, msg.attenuation_linear, msg.attenuation_quadratic = cfg["attenuation"]
        if "direction" in cfg:
            msg.direction.x, msg.direction.y, msg.direction.z = cfg["direction"]
        self.light_pub.publish(msg)

    def depth_without_panes(self, panes):
        """Depth image with the given glass panes moved out of the way (see-through ground truth).
        Each pane is detached, moved away, and put back + re-attached at exactly its old pose."""
        saved = {n: self.model_pose(n) for n in panes}
        for n in panes:
            self.pane_pub[n][0].publish(Empty())
        self.step(2)
        for n in panes:
            self.set_pose(n, [0, 0, -50])
        depth = image_array(self.render(["depth"])["depth"], "depth")
        for n, (pos, q) in saved.items():
            self.set_pose(n, pos, orientation=q)
        self.step(2)
        for n in panes:
            self.pane_pub[n][1].publish(Empty())
        self.step(2)
        return depth

    def camera_info(self):
        with self.lock:
            c = self.latest.get("camera_info")
        return None if c is None else {"width": c.width, "height": c.height, "K": list(c.intrinsics.k)}

    def stop(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
