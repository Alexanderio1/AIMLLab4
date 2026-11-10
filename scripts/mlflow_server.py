"""Manage only the MLflow server started by this project."""
import argparse
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "mlflow-data"
PIDFILE = STATE / "server.pid"


def alive(pid):
    try:
        os.kill(pid, 0)
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        return b"mlflow" in command and str(STATE).encode() in command
    except (ProcessLookupError, FileNotFoundError, PermissionError):
        return False


def stop():
    if not PIDFILE.exists():
        print("No project-owned MLflow PID file; nothing stopped")
        return
    pid = int(PIDFILE.read_text())
    if alive(pid):
        os.killpg(pid, signal.SIGTERM)
        for _ in range(40):
            if not alive(pid):
                break
            time.sleep(.25)
        else:
            raise RuntimeError("MLflow did not stop; backup would not be consistent")
    PIDFILE.unlink(missing_ok=True)
    print("Project MLflow server stopped")


def start():
    STATE.mkdir(exist_ok=True)
    (STATE / "artifacts").mkdir(exist_ok=True)
    pid = int(PIDFILE.read_text()) if PIDFILE.exists() else None
    if pid is None or not alive(pid):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", 5000)) == 0:
                raise RuntimeError("Port 5000 is already used by another process. Do not stop an unrelated service.")
        with (STATE / "server.log").open("ab") as out:
            process = subprocess.Popen([
                str(ROOT / ".venv/bin/mlflow"), "server",
                "--host", "127.0.0.1", "--port", "5000", "--workers", "1",
                "--backend-store-uri", f"sqlite:///{STATE / 'mlflow.db'}",
                "--serve-artifacts", "--artifacts-destination", str(STATE / "artifacts"),
            ], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        PIDFILE.write_text(str(process.pid))
    for _ in range(90):
        try:
            r = requests.get("http://127.0.0.1:5000/health", timeout=2)
            if r.status_code == 200:
                print("MLflow ready: http://127.0.0.1:5000")
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError("MLflow did not become ready; see mlflow-data/server.log")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["start", "stop"])
    args = p.parse_args()
    globals()[args.action]()
