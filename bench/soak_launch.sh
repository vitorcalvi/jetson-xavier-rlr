#!/bin/bash
# Start the 24-hour soak.
#
# The supervisor (soak_supervisor.py) does its own preflight and thermal
# pause/abort. This launcher is the entrypoint the human (or scheduler) calls;
# it waits for the device to cool, then starts the llama-server, the
# supervisor, and an independent watchdog that kills everything at 72 C.
# Both the watchdog and the supervisor are detached so closing the SSH
# session does not take them down.
set -u
ROOT=/models/rlr/soak-24h
mkdir -p "$ROOT"
echo "$(date -u +%FT%TZ) launcher started" >> "$ROOT/launcher.log"

# Block start if a previous soak is still RUNNING.
if [ -e "$ROOT/RUNNING" ]; then
  echo "ABORT: $ROOT/RUNNING exists; previous soak not finished" >> "$ROOT/launcher.log"
  exit 1
fi

for i in $(seq 1 180); do
  T=$(cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)
  T=$((T/1000))
  [ "$T" -le 54 ] && break
  sleep 10
done
echo "$(date -u +%FT%TZ) starting at ${T}C" >> "$ROOT/launcher.log"

pkill -9 -f llama-server 2>/dev/null; sleep 3

nohup /models/build/llama/bin/llama-server \
  -m /models/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
  -ngl 20 -c 4096 --load-mode mmap -b 512 -ub 128 \
  --host 127.0.0.1 --port 8080 --alias Qwen2.5-1.5B \
  > "$ROOT/llama-server.log" 2>&1 &
echo "$(date -u +%FT%TZ) llama-server pid=$!" >> "$ROOT/launcher.log"

for j in $(seq 1 60); do
  sleep 3
  if python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3)" 2>/dev/null; then
    echo "$(date -u +%FT%TZ) llama healthy after ${j}x3s" >> "$ROOT/launcher.log"
    break
  fi
done

cd /home/vitor/jetson-xavier-rlr
nohup bash bench/soak_watchdog.sh 72 30 > "$ROOT/watchdog-stdouterr.log" 2>&1 &
echo "$(date -u +%FT%TZ) watchdog pid=$!" >> "$ROOT/launcher.log"

SOAK_DIR="$ROOT" \
  SOAK_CODE=/home/vitor/jetson-xavier-rlr/rlr \
  SOAK_DURATION=86400 \
  SOAK_TICK=10 \
  SOAK_LOAD_TARGET=50 \
  SOAK_SELF_REPORT_EVERY=20 \
  SOAK_REFLECT_EVERY=30 \
  SOAK_LOG_INTERVAL=30 \
  nohup python3 bench/soak_supervisor.py > "$ROOT/supervisor.log" 2>&1 &
echo "$(date -u +%FT%TZ) supervisor pid=$!" >> "$ROOT/launcher.log"
echo "$(date -u +%FT%TZ) launcher exiting; soak supervisor detached" >> "$ROOT/launcher.log"