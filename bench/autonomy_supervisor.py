"""Supervised autonomy run for RLR with an independent thermal watchdog.

Mirrors the pattern of the project's existing soak supervisor: verify the fixed
operating policy before starting, drive one RLR loop from the deployed
checkout, sample thermal/RAM independently of RLR's own telemetry, pause the
loop preemptively near the thermal ceiling, and stop cleanly on any fault.

Environment:
  AUTONOMY_DIR              run directory (default /models/rlr/autonomy-run)
  AUTONOMY_TICK             loop tick seconds (default 30)
  AUTONOMY_DURATION         wall-clock seconds (default 330)
  AUTONOMY_SELF_REPORT_EVERY  ticks between self-reports (default 5)
  AUTONOMY_REFLECT_EVERY    reflection interval in minutes (default 15, 0 disables)
  AUTONOMY_PROBE            append a forced reflection/self-report probe at stop
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("AUTONOMY_DIR", "/models/rlr/autonomy-run"))
CODE = os.environ.get("AUTONOMY_CODE", "/home/vitor/jetson-xavier-rlr/rlr")
DURATION = float(os.environ.get("AUTONOMY_DURATION", "330"))
TICK = float(os.environ.get("AUTONOMY_TICK", "30"))
SELF_REPORT_EVERY = int(os.environ.get("AUTONOMY_SELF_REPORT_EVERY", "5"))
REFLECT_MINUTES = float(os.environ.get("AUTONOMY_REFLECT_EVERY", "15"))
PROBE = os.environ.get("AUTONOMY_PROBE", "0") not in ("0", "", "no", "false")

ROOT.mkdir(parents=True, exist_ok=True)
OUT = open(ROOT / "supervisor.log", "w", buffering=1)
THERM = open(ROOT / "thermal.csv", "w", buffering=1)
THERM.write("elapsed_s,max_temp_c,ram_used_mib,paused\n")
START = time.monotonic()
STATUS = "complete"
PROCS = []


def zones():
    """Non-PMIC thermal zones, matching the project's exclusion rule."""
    values = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            if "PMIC" in (path.parent / "type").read_text():
                continue
            values.append(int(path.read_text().strip()) / 1000.0)
        except Exception:
            pass
    return values


def ram_used():
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
        total = int(next(x.split()[1] for x in lines if x.startswith("MemTotal:")))
        avail = int(next(x.split()[1] for x in lines if x.startswith("MemAvailable:")))
        return round((total - avail) / 1024.0, 1)
    except Exception:
        return -1.0


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


DRIVER = ROOT / "driver.py"
DRIVER.write_text('''\
import os, sys, signal, threading
sys.path.insert(0, "%s")
from rlr.llm import LlamaServer
from rlr.runtime import Config, ReflectiveLoop

config = Config(data_dir=os.environ["AUTONOMY_DIR"], db_path=os.environ["AUTONOMY_DB"],
                tick_seconds=float(os.environ["AUTONOMY_TICK"]),
                reflection_minutes=float(os.environ["AUTONOMY_REFLECT_EVERY"]),
                self_report_every=int(os.environ["AUTONOMY_SELF_REPORT_EVERY"]))
loop = ReflectiveLoop(LlamaServer("http://127.0.0.1:8080", model="Qwen2.5-1.5B"),
                      config=config)
stop_event = threading.Event()
for received in (signal.SIGTERM, signal.SIGINT):
    signal.signal(received, lambda *_: stop_event.set())
try:
    loop.run_forever(stop_event)
finally:
    loop.close()
''' % CODE)

PROBE_SCRIPT = ROOT / "probe.py"
PROBE_SCRIPT.write_text('''\
"""Forced reflection and self-report probe against the run's own database."""
import json, os, sys
sys.path.insert(0, "%s")
from rlr.llm import LlamaServer
from rlr.runtime import Config, ReflectiveLoop

config = Config(data_dir=os.environ["AUTONOMY_DIR"], db_path=os.environ["AUTONOMY_DB"],
                self_report_every=0)
loop = ReflectiveLoop(LlamaServer("http://127.0.0.1:8080", model="Qwen2.5-1.5B"),
                      config=config)
result = {}
try:
    result["reflection_id"] = loop.reflect()
    result["reflection_produced"] = result["reflection_id"] is not None
    result["latest"] = loop.store.latest_reflection()
    result["self_report"] = loop.self_report()
    result["discrimination"] = loop.self_report_discrimination()
finally:
    loop.close()
open(os.path.join(os.environ["AUTONOMY_DIR"], "probe.json"), "w").write(
    json.dumps(result, ensure_ascii=False, indent=2))
''' % CODE)

try:
    mode = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5).stdout
    if "MODE_10W_DESKTOP" not in mode:
        raise RuntimeError("nvpmodel is not ID 5 / MODE_10W_DESKTOP")
    guard = subprocess.run(["systemctl", "is-active", "thermal-guard.service"],
                           capture_output=True, text=True).stdout.strip()
    if guard != "active":
        raise RuntimeError("thermal-guard is not active")
    import urllib.request
    urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=5).close()
    OUT.write("PREFLIGHT ok mode=MODE_10W_DESKTOP guard=%s server=healthy\n" % guard)

    env = dict(os.environ, PYTHONPATH=CODE, AUTONOMY_DIR=str(ROOT),
               AUTONOMY_DB=str(ROOT / "brain.db"), AUTONOMY_TICK=str(TICK),
               AUTONOMY_SELF_REPORT_EVERY=str(SELF_REPORT_EVERY))
    rlr_log = open(ROOT / "rlr.log", "w", buffering=1)
    rlr = subprocess.Popen([sys.executable, str(DRIVER)], cwd=CODE, env=env,
                           stdout=rlr_log, stderr=subprocess.STDOUT, start_new_session=True)
    PROCS.append(rlr)
    OUT.write("START tick=%.0fs duration=%.0fs self_report_every=%d reflect_every=%.1fm pid=%d\n"
              % (TICK, DURATION, SELF_REPORT_EVERY, REFLECT_MINUTES, rlr.pid))

    paused = False
    while time.monotonic() - START < DURATION:
        temperatures = zones()
        temperature = max(temperatures) if temperatures else -1.0
        THERM.write("%.1f,%.1f,%.1f,%d\n"
                    % (time.monotonic() - START, temperature, ram_used(), int(paused)))
        if temperature >= 80:
            STATUS = "aborted_temperature_ge_80"
            raise RuntimeError(STATUS)
        if paused and temperatures and all(value < 70 for value in temperatures):
            if rlr.poll() is None:
                os.killpg(os.getpgid(rlr.pid), signal.SIGCONT)
            paused = False
            OUT.write("THERMAL_RESUME zones=%s\n" % temperatures)
        elif not paused and temperature >= 76:
            if rlr.poll() is None:
                os.killpg(os.getpgid(rlr.pid), signal.SIGSTOP)
            paused = True
            OUT.write("THERMAL_PAUSE zones=%s\n" % temperatures)
        if rlr.poll() is not None:
            raise RuntimeError("RLR driver exited with code %s" % rlr.returncode)
        time.sleep(5)

    if paused and rlr.poll() is None:
        # The loop must not stay SIGSTOPped while the probe runs.
        os.killpg(os.getpgid(rlr.pid), signal.SIGCONT)
        paused = False
        OUT.write("THERMAL_RESUME_FOR_PROBE\n")
    OUT.write("LOOP_COMPLETE elapsed_s=%.1f\n" % (time.monotonic() - START))
except Exception as error:
    if STATUS == "complete":
        STATUS = "error"
    OUT.write("SUPERVISOR_ERROR %s %s\n" % (type(error).__name__, error))
finally:
    for process in reversed(PROCS):
        stop(process)
    THERM.close()
    OUT.write("SUPERVISOR_STATUS %s elapsed_s %.1f\n"
              % (STATUS, time.monotonic() - START))

if PROBE and STATUS == "complete":
    # The probe is read-mostly and the watchdog loop has already ended, so it
    # cannot be frozen by a thermal SIGSTOP. It writes its own JSON artifact.
    probe_env = dict(os.environ, PYTHONPATH=CODE, AUTONOMY_DIR=str(ROOT),
                     AUTONOMY_DB=str(ROOT / "brain.db"))
    try:
        probe = subprocess.run([sys.executable, str(PROBE_SCRIPT)], cwd=CODE, env=probe_env,
                               capture_output=True, text=True, timeout=300)
        OUT.write("PROBE rc=%d\n" % probe.returncode)
        if probe.returncode != 0:
            OUT.write("PROBE_STDERR %s\n" % probe.stderr.strip()[-1500:])
        else:
            OUT.write("PROBE_OK\n")
    except Exception as probe_error:
        OUT.write("PROBE_ERROR %s %s\n" % (type(probe_error).__name__, probe_error))
OUT.close()
sys.exit(0 if STATUS == "complete" else 1)
