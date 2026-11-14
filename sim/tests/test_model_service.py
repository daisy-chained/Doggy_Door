# Shared model service: one detector / door-crop classifier for several tours at once (Madelief 2026-11-14: "build it", so two tours
# fit side by side: each tour now starts its own detect_server.py (1.3 GB) + props_server.py (0.8 GB) over stdin / stdout).
# No torch / GPU needed: a fake line server stands in for the model. python3 -m pytest sim/tests/test_model_service.py
#
# sim/building/model_service.py (stdlib only):
#   python3 sim/building/model_service.py --sock PATH -- CMD...
#   Starts CMD (a line server: prints "READY" once loaded, then one line out per line in, in order), waits for READY, writes
#   PATH + ".json" = {"argv": CMD, "pid": <service pid>, "ready": true}, then listens on the Unix socket PATH. Every client
#   connection may send many lines; each line goes to CMD and CMD's answer line goes back to that client. Requests from different
#   clients are passed to CMD one at a time (a lock), so answers never mix. Prints "READY" itself once listening.
#   When CMD dies: clients get the line {"error": "model service: model process ended"} and the service exits (non-zero).
#   SIGTERM / SIGINT: closes CMD's stdin, waits for it, removes PATH and PATH.json, exits 0.
#   Function serve(sock_path, argv) runs the above (blocking) — the CLI calls it.
#
# sim/doorsim/model_link.py (stdlib only):
#   open_model(argv, sock=None, cwd=None) -> object with .stdin (write lines), .stdout (readline), .wait(timeout), .shared (bool).
#     sock given, PATH.json says ready with the same argv, and PATH accepts a connection -> a client of the shared service
#     (shared=True; .stdin.close() / .wait() only close this client's connection, the service keeps running);
#     anything else (sock None, missing, other argv, connect refused) -> starts argv itself as before (subprocess.Popen, text,
#     line-buffered, reads up to "READY"; shared=False). door_tour uses this for both models, the socket path from
#     env DOORSIM_DETECT_SOCK / DOORSIM_PROPS_SOCK.
import json, os, pathlib, signal, subprocess, sys, threading, time
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import model_link

SERVICE = SIM / "building" / "model_service.py"

FAKE = r'''
import sys, time, json
time.sleep(0.3)
print("READY", flush=True)
for line in sys.stdin:
    line = line.strip()
    if line == "die":
        sys.exit(3)
    time.sleep(0.02)
    print(json.dumps({"echo": line}), flush=True)
'''


@pytest.fixture
def fake(tmp_path):
    f = tmp_path / "fake_server.py"; f.write_text(FAKE)
    return [sys.executable, str(f)]


def start(sock, argv):
    p = subprocess.Popen([sys.executable, str(SERVICE), "--sock", str(sock), "--", *argv], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "READY"
    return p


def stop(p):
    if p.poll() is None:
        p.send_signal(signal.SIGTERM)
        p.wait(timeout=10)


def test_service_answers_and_writes_info(tmp_path, fake):
    sock = tmp_path / "m.sock"
    p = start(sock, fake)
    try:
        info = json.loads((tmp_path / "m.sock.json").read_text())
        assert info["argv"] == fake and info["ready"] is True and info["pid"] == p.pid
        m = model_link.open_model(fake, sock=str(sock))
        assert m.shared
        for k in range(3):
            m.stdin.write(f"img{k}.png\n"); m.stdin.flush()
            assert json.loads(m.stdout.readline()) == {"echo": f"img{k}.png"}
        m.stdin.close(); m.wait(timeout=5)
    finally:
        stop(p)
    assert p.returncode == 0
    assert not sock.exists() and not (tmp_path / "m.sock.json").exists()


def test_two_clients_at_once_never_mix(tmp_path, fake):
    sock = tmp_path / "m.sock"
    p = start(sock, fake)
    errors = []

    def client(tag):
        m = model_link.open_model(fake, sock=str(sock))
        for k in range(25):
            m.stdin.write(f"{tag}-{k}\n"); m.stdin.flush()
            got = json.loads(m.stdout.readline())
            if got != {"echo": f"{tag}-{k}"}:
                errors.append((tag, k, got))
        m.stdin.close(); m.wait(timeout=5)

    try:
        ts = [threading.Thread(target=client, args=(t,)) for t in ("a", "b", "c")]
        [t.start() for t in ts]; [t.join(timeout=60) for t in ts]
        assert not errors
        # the service keeps running after clients leave
        assert p.poll() is None
    finally:
        stop(p)


def test_falls_back_to_own_process(tmp_path, fake):
    # no socket at all
    m = model_link.open_model(fake, sock=str(tmp_path / "missing.sock"))
    assert not m.shared
    m.stdin.write("x\n"); m.stdin.flush()
    assert json.loads(m.stdout.readline()) == {"echo": "x"}
    m.stdin.close(); m.wait(timeout=10)
    # sock=None
    m = model_link.open_model(fake)
    assert not m.shared
    m.stdin.close(); m.wait(timeout=10)


def test_other_model_is_not_shared(tmp_path, fake):
    sock = tmp_path / "m.sock"
    p = start(sock, fake)
    try:
        other = fake + ["--familiar", "somewhere"]       # different argv (e.g. props_server with --familiar) -> own process
        m = model_link.open_model(other, sock=str(sock))
        assert not m.shared
        m.stdin.close(); m.wait(timeout=10)
    finally:
        stop(p)


def test_model_death_tells_the_client_and_ends_the_service(tmp_path, fake):
    sock = tmp_path / "m.sock"
    p = start(sock, fake)
    try:
        m = model_link.open_model(fake, sock=str(sock))
        m.stdin.write("die\n"); m.stdin.flush()
        assert "error" in json.loads(m.stdout.readline())
        p.wait(timeout=10)
        assert p.returncode != 0
    finally:
        stop(p)
