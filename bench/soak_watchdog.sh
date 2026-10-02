#!/bin/bash
# Independent thermal watchdog for a 24-hour soak.
#
# The supervisor has its own 80 C abort, but 80 C is past the runtime's
# cooldown threshold (78 C), so the device is already in trouble by the
# time the supervisor fires. This watchdog kills the supervisor AND the
# model server at 72 C and leaves a record. It covers every process that
# could heat the device, so a stray benchmark that joined the device
# later would still be caught.
set -u
MAX_T="${1:-72}"
INTERVAL="${2:-30}"
LOG="/models/rlr/soak-24h/watchdog.csv"
echo "ts_utc,max_temp_c,ram_used_mib,action" > "$LOG"
MAXSEEN=0
while true; do
  RAW=$(cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)
  [ -z "$RAW" ] && continue
  T=$((RAW / 1000))
  RAM=$(free -m | awk '/Mem:/{print $3}')
  ACTION=ok
  if [ "$T" -gt "$MAXSEEN" ]; then MAXSEEN=$T; fi
  if [ "$T" -ge 70 ] && [ "$T" -lt "$MAX_T" ]; then ACTION=warn; fi
  echo "$(date -u +%FT%TZ),$T,$RAM,$ACTION" >> "$LOG"
  if [ "$T" -ge "$MAX_T" ]; then
    echo "$(date -u +%FT%TZ),$T,$RAM,ABORT" >> "$LOG"
    pkill -9 -f soak_supervisor 2>/dev/null
    pkill -9 -f llama-server 2>/dev/null
    pkill -9 -f autonomy_supervisor 2>/dev/null
    pkill -9 -f kapro_rlr 2>/dev/null
    echo "ABORTED at ${T}C (threshold ${MAX_T}C); peak ${MAXSEEN}C"
    exit 1
  fi
  sleep "$INTERVAL"
done