#!/bin/sh
set -eu
WARN=${WARN:-80}
KILL=${KILL:-85}
SHUTDOWN=${SHUTDOWN:-90}
INTERVAL=${INTERVAL:-5}
DRYRUN=${DRYRUN:-0}
LOG_STDERR=${LOG_STDERR:-0}
NVP_MODEL=${NVP_MODEL:-5}
MANAGED_UNITS=${MANAGED_UNITS:-"rlr-inference.service"}
ALLOW_PKILL_FALLBACK=${ALLOW_PKILL_FALLBACK:-1}
hot_kill=0
hot_shutdown=0
log() { logger -t thermal-guard -- "$*"; [ "$LOG_STDERR" = 1 ] && printf "%s\n" "$*" >&2 || :; }

max_temp() {
  max=0; max_name=unknown
  for zone in /sys/class/thermal/thermal_zone*; do
    [ -r "$zone/temp" ] || continue
    type=$(cat "$zone/type" 2>/dev/null) || continue
    case "$type" in *PMIC*) continue;; esac
    raw=$(cat "$zone/temp" 2>/dev/null) || continue
    case "$raw" in ""|*[!0-9-]*) continue;; esac
    value=$((raw / 1000))
    [ "$value" -le 0 ] && continue
    [ "$value" -ge 150 ] && continue
    if [ "$value" -gt "$max" ]; then max=$value; max_name=$type; fi
  done
}

terminate_workloads() {
  acted=0
  for unit in $MANAGED_UNITS; do
    if systemctl is-active --quiet "$unit" 2>/dev/null; then
      acted=1
      if [ "$DRYRUN" = 1 ]; then
        log "DRYRUN: would TERM managed unit $unit at ${max}C"
      else
        log "temperature ${max}C: TERM managed unit $unit"
        systemctl kill --kill-who=all --signal=TERM "$unit" 2>/dev/null || \
          log "failed to TERM managed unit $unit"
      fi
    fi
  done
  if [ "$acted" -eq 0 ] && [ "$ALLOW_PKILL_FALLBACK" = 1 ]; then
    pattern="llama-server|whisper|trtexec|cc1plus|nvcc|cicc|ptxas"
    if [ "$DRYRUN" = 1 ]; then
      log "DRYRUN: no active managed unit; fallback would TERM pattern $pattern at ${max}C"
    else
      log "temperature ${max}C: no active managed unit; conservative pkill fallback"
      pkill -TERM -f "$pattern" 2>/dev/null || :
    fi
  fi
}

while :; do
  max_temp
  log "maximum temperature: ${max}C (${max_name})"
  if [ "$max" -ge "$WARN" ]; then
    if [ "$DRYRUN" = 1 ]; then
      log "DRYRUN: set nvpmodel -m $NVP_MODEL (10W)"
    else
      current=$(nvpmodel -q 2>/dev/null || :)
      case "$current" in
        *MODE_10W*) ;;
        *) log "temperature ${max}C: switching to 10W model $NVP_MODEL"
           nvpmodel -m "$NVP_MODEL" >/dev/null 2>&1 || log "nvpmodel action failed";;
      esac
    fi
  fi
  if [ "$max" -ge "$KILL" ]; then hot_kill=$((hot_kill + 1)); else hot_kill=0; fi
  if [ "$max" -ge "$SHUTDOWN" ]; then hot_shutdown=$((hot_shutdown + 1)); else hot_shutdown=0; fi
  if [ "$hot_kill" -ge 2 ]; then
    terminate_workloads
    hot_kill=0
  fi
  if [ "$hot_shutdown" -ge 2 ]; then
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN: would poweroff at ${max}C"
    else log "temperature ${max}C for $hot_shutdown consecutive reads: poweroff"; systemctl poweroff; fi
    hot_shutdown=0
  fi
  sleep "$INTERVAL"
done
