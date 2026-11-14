"""Open a model line server (detector / door-crop classifier) for door_tour.

door_tour uses this for both models. The socket paths come from the env vars DOORSIM_DETECT_SOCK and
DOORSIM_PROPS_SOCK. If a shared service (sim/building/model_service.py) is running for the same argv, we talk to it;
otherwise we start our own process, which is the old behaviour.
"""
import json, os, socket, subprocess


class _Shared:
    shared = True

    def __init__(self, s):
        self._sock = s
        self.stdin = s.makefile("w", buffering=1)
        self.stdout = s.makefile("r")
        # closing stdin must end the write side so the service sees EOF
        raw_close = self.stdin.close

        def close():
            try:
                raw_close()
            finally:
                try:
                    s.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
        self.stdin.close = close

    def wait(self, timeout=None):
        for f in (self.stdin, self.stdout):
            try:
                f.close()
            except OSError:
                pass
        self._sock.close()
        return 0


class _Own:
    shared = False

    def __init__(self, proc):
        self._proc = proc
        self.stdin = proc.stdin
        self.stdout = proc.stdout

    def wait(self, timeout=None):
        return self._proc.wait(timeout=timeout)


def _try_shared(argv, sock):
    try:
        with open(sock + ".json") as f:
            info = json.load(f)
        if not info.get("ready") or [str(a) for a in info.get("argv", [])] != [str(a) for a in argv]:
            return None
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.connect(sock)
        except OSError:
            s.close()
            return None
        return _Shared(s)
    except (OSError, ValueError):
        return None


def open_model(argv, sock=None, cwd=None):
    if sock:
        m = _try_shared(argv, sock)
        if m:
            return m
    proc = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    while True:
        line = proc.stdout.readline()
        if not line:
            proc.wait()
            raise RuntimeError(f"model process ended before READY: {argv}")
        if line.strip() == "READY":
            return _Own(proc)
