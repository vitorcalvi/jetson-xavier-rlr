"""24-hour soak supervisor.

Drives the RLR loop from the deployed checkout under continuous monitoring
and a thermal ceiling the failed-fan device actually respects. Mirrors
`autonomy_supervisor.py` but is independent because:

- The duration is 24 h, not the supervised 5.5 min that script targets.
- The watchdog pauses at 76 C and aborts at 80 C.
- A `RUNNING` sentinel refuses to start twice from the same directory, so a
  re-run never silently overwrites a half-written previous attempt.
- The sink is JSONL with a status field the KAPRO report generator can read.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("SOAK_DIR", "/models/rlr/soak-24h"))
CODE = Path(os.environ.get("SOAK_CODE", "/home/vitor/jetson-xavier-rlr/rlr"))
DURATION = float(os.environ.get("SOAK_DURATION", "86400"))
TICK = float(os.environ.get("SOAK_TICK", "10"))
LOAD_TARGET = int(os.environ.get("SOAK_LOAD_TARGET", "50"))
SELF_REPORT_EVERY = int(os.environ.get("SOAK_SELF_REPORT_EVERY", "20"))
REFLECT_EVERY = float(os.environ.get("SOAK_REFLECT_EVERY", "30"))
HEALTH_URL = os.environ.get("SOAK_HEALTH_URL", "http://127.0.0.1:8080/health")
LOG_INTERVAL = int(os.environ.get("SOAK_LOG_INTERVAL", "30"))

START = time.monotonic()
ROOT.mkdir(parents=True, exist_ok=True)
RUNNING = ROOT / "RUNNING"
if RUNNING.exists():
    print("ABORT: %s already exists" % RUNNING)
    sys.exit(2)
RUNNING.write_text("pid=%d ts=%s\n" % (os.getpid(),
                                       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
LOG = open(ROOT / "soak.jsonl", "w", buffering=1)
SUMMARY = ROOT / "summary.json"


def log(event, **fields):
    LOG.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         "elapsed_s": round(time.monotonic() - START, 1),
                         "event": event, **fields}, ensure_ascii=False) + "\n")


def zones():
    out = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            if "PMIC" in (path.parent / "type").read_text():
                continue
            out.append(int(path.read_text().strip()) / 1000.0)
        except Exception:
            pass
    return out


def ram_used():
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
        total = int(next(x.split()[1] for x in lines if x.startswith("MemTotal:")))
        avail = int(next(x.split()[1] for x in lines if x.startswith("MemAvailable:")))
        return round((total - avail) / 1024.0, 1)
    except Exception:
        return -1.0


def db_size():
    db = ROOT / "brain.db"
    return db.stat().st_size if db.exists() else 0


def stop(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def preflight():
    mode = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True,
                         timeout=5).stdout
    if "MODE_10W_DESKTOP" not in mode:
        raise RuntimeError("nvpmodel is not MODE_10W_DESKTOP")
    guard = subprocess.run(["systemctl", "is-active", "thermal-guard.service"],
                           capture_output=True, text=True).stdout.strip()
    if guard != "active":
        raise RuntimeError("thermal-guard is not active")
    import urllib.request
    urllib.request.urlopen(HEALTH_URL, timeout=5).close()
    log("preflight_ok", mode="MODE_10W_DESKTOP", guard=guard,
        load_target=LOAD_TARGET, tick=TICK, duration=DURATION)


DRIVER = ROOT / "driver.py"
DRIVER.write_text('''\
import os, sys, signal, threading
sys.path.insert(0, "%s")
from rlr.llm import LlamaServer
from rlr.runtime import Config, ReflectiveLoop

config = Config(data_dir=os.environ["SOAK_DIR"],
                db_path=os.environ["SOAK_DB"],
                tick_seconds=float(os.environ["SOAK_TICK"]),
                reflection_minutes=float(os.environ["SOAK_REFLECT_EVERY"]),
                self_report_every=int(os.environ["SOAK_SELF_REPORT_EVERY"]))
loop = ReflectiveLoop(LlamaServer("http://127.0.0.1:8080", model="Qwen2.5-1.5B"),
                      config=config)
loop.settings["load_target"] = int(os.environ["SOAK_LOAD_TARGET"])
stop_event = threading.Event()
for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, lambda *_: stop_event.set())
try:
    loop.run_forever(stop_event)
finally:
    loop.close()
''' % CODE)

status = "complete"
paused = False
try:
    preflight()
    env = dict(os.environ, PYTHONPATH=str(CODE), SOAK_DIR=str(ROOT),
               SOAK_DB=str(ROOT / "brain.db"), SOAK_TICK=str(TICK),
               SOAK_LOAD_TARGET=str(LOAD_TARGET),
               SOAK_SELF_REPORT_EVERY=str(SELF_REPORT_EVERY),
               SOAK_REFLECT_EVERY=str(REFLECT_EVERY))
    rlr_log = open(ROOT / "rlr.log", "w", buffering=1)
    rlr = subprocess.Popen([sys.executable, str(DRIVER)], cwd=CODE, env=env,
                           stdout=rlr_log, stderr=subprocess.STDOUT,
                           start_new_session=True)
    log("start", pid=rlr.pid, tick=TICK, duration=DURATION,
        load_target=LOAD_TARGET)

    next_sample = START + LOG_INTERVAL
    while time.monotonic() - START < DURATION:
        now = time.monotonic()
        ts = zones()
        if now >= next_sample:
            log("sample", max_temp_c=round(max(ts), 1) if ts else -1.0,
                ram_used_mib=ram_used(), db_bytes=db_size(),
                paused=int(paused))
            next_sample = now + LOG_INTERVAL
        if ts and max(ts) >= 80:
            status = "aborted_temperature_ge_80"
            log("abort_temperature", max_temp_c=max(ts))
            raise RuntimeError(status)
        if (not paused) and ts and max(ts) >= 76:
            if rlr.poll() is None:
                os.killpg(os.getpgid(rlr.pid), signal.SIGSTOP)
            paused = True
            log("thermal_pause", max_temp_c=max(ts))
        elif paused and ts and all(t < 70 for t in ts):
            if rlr.poll() is None:
                os.killpg(os.getpgid(rlr.pid), signal.SIGCONT)
            paused = False
            log("thermal_resume", max_temp_c=max(ts))
        if rlr.poll() is not None:
            status = "loop_exited"
            log("loop_exit", returncode=rlr.returncode)
            raise RuntimeError(status)
        time.sleep(5)
except Exception as error:
    if status == "complete":
        status = "error"
    log("supervisor_error", kind=type(error).__name__, message=str(error))
    try:
        if paused and 'rlr' in locals() and rlr.poll() is None:
            os.killpg(os.getpgid(rlr.pid), signal.SIGCONT)
    except Exception:
        pass
    if 'rlr' in locals() and rlr.poll() is None:
        stop(rlr)

SUMMARY.write_text(json.dumps({
    "status": status,
    "elapsed_s": round(time.monotonic() - START, 1),
    "duration_s": DURATION,
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(START)),
    "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "tick_seconds": TICK,
    "load_target": LOAD_TARGET,
}, indent=2))
LOG.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "elapsed_s": round(time.monotonic() - START, 1),
                     "event": "end", "status": status}) + "\n")
LOG.close()
try:
    RUNNING.unlink()
except OSError:
    pass
sys.exit(0 if status == "complete" else 1)