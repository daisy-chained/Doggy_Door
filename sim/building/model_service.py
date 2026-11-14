"""Shared model service.

One detector (train/detect_server.py) or door-crop classifier (train/props_server.py) is shared by several robot
tours at once, so two tours fit in RAM side by side (Madelief 2026-11-14).

    python3 sim/building/model_service.py --sock PATH -- CMD...

CMD is a line server (one answer line per request line, prints READY when loaded). This service starts it, writes
PATH.json, listens on the Unix socket PATH and passes the lines of every client to CMD, one request at a time.
"""
import argparse, json, os, signal, socket, subprocess, sys, threading

ERROR_LINE = json.dumps({"error": "model service: model process ended"}) + "\n"


def _remove(*paths):
    for p in paths:
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass


def serve(sock_path, argv):
    info_path = sock_path + ".json"
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    # wait for the model to say READY
    while True:
        line = proc.stdout.readline()
        if not line:
            print("model service: model ended before READY", file=sys.stderr)
            return 1
        if line.strip() == "READY":
            break

    lock = threading.Lock()
    stop = threading.Event()
    exit_code = [0]

    def ask(line):
        """Send one line to the model, return its answer ('' if the model is gone)."""
        with lock:
            try:
                proc.stdin.write(line if line.endswith("\n") else line + "\n")
                proc.stdin.flush()
                return proc.stdout.readline()
            except (BrokenPipeError, ValueError, OSError):
                return ""

    def handle(conn):
        with conn, conn.makefile("r") as rf, conn.makefile("w") as wf:
            try:
                for line in rf:
                    answer = ask(line)
                    if not answer:
                        wf.write(ERROR_LINE); wf.flush()
                        exit_code[0] = 1
                        stop.set()
                        return
                    wf.write(answer); wf.flush()
            except OSError:
                pass  # client went away

    def on_signal(signum, frame):
        stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    _remove(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(16)
    srv.settimeout(0.2)
    with open(info_path, "w") as f:
        json.dump({"argv": argv, "pid": os.getpid(), "ready": True}, f)
    print("READY", flush=True)

    try:
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            threading.Thread(target=handle, args=(conn,), daemon=True).start()
    finally:
        srv.close()
        _remove(sock_path, info_path)
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return exit_code[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("missing CMD after --")
    sys.exit(serve(a.sock, cmd))


if __name__ == "__main__":
    main()
