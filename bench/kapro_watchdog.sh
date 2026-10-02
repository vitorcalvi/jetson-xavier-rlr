#!/bin/bash
# Thermal watchdog for a KAPRO run.
#
# The Xavier fan is documented as failed, so nothing actively cools the device
# during a run. A Qwen2.5-1.5B server under full GPU offload was observed to
# reach 76.5 C, past the runtime's 75 C degraded threshold and close to its
# 78 C cooldown. This aborts the run before that, and records what it saw.
#
# Usage: bash kapro_watchdog.sh <max_temp_c> <interval_s> <pattern-to-kill>
set -u
MAX_T="${1:-76}"
INTERVAL="${2:-10}"
PATTERN="${3:-kapro_rlr}"
LOG="/models/bench/kapro_watchdog.csv"
echo "ts_utc,max_temp_c,ram_used_mib,action" > "$LOG"
MAXSEEN=0
while true; do
  RAW=$(cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)
  [ -z "$RAW" ] && continue
  T=$((RAW / 1000))
  RAM=$(free -m | awk '/Mem:/{print $3}')
  ACTION=ok
  if [ "$T" -gt "$MAXSEEN" ]; then MAXSEEN=$T; fi
  if [ "$T" -ge 76 ]; then ACTION=near_limit; fi
  echo "$(date -u +%FT%TZ),$T,$RAM,$ACTION" >> "$LOG"
  if [ "$T" -ge "$MAX_T" ]; then
    echo "$(date -u +%FT%TZ),$T,$RAM,ABORT" >> "$LOG"
    pkill -f "$PATTERN" 2>/dev/null
    echo "ABORTED at ${T}C (threshold ${MAX_T}C); peak ${MAXSEEN}C"
    exit 1
  fi
  sleep "$INTERVAL"
done
