#!/bin/bash
# Run the KAPRO-RLR conditions sequentially on the device, under a thermal
# watchdog, cooling between conditions.
#
# Sequential rather than parallel on purpose: one Xavier with a failed fan
# cannot hold two 1.5B inference sessions at a safe temperature, and a run that
# is thermally throttled mid-condition would make that condition's numbers
# incomparable with the others.
#
# Usage: bash kapro_run_all.sh <out-root> [per-seed]
set -u
ROOT="${1:-/models/bench/kapro-conditions}"
PER_SEED="${2:-60}"
SEEDS="11 23 42 71 101"
REPO=/home/vitor/jetson-xavier-rlr
mkdir -p "$ROOT"

# full context, then each self-model ablation, then the distractor condition.
CONDITIONS="full_self_model no_recent_actions no_capabilities no_goals no_reflections no_episodes no_memories no_self_model stale_telemetry corrupt_capabilities shuffled_action_history"
DISTRACTOR_CONDITIONS="distractors"

run_one () {
  local cond="$1" extra="${2:-}"
  local dir="$ROOT/$cond"
  echo "=== $cond $(date -u +%FT%TZ) ==="
  rm -rf "$dir"
  # shellcheck disable=SC2086
  nohup python3 "$REPO/bench/kapro_rlr.py" --seeds $SEEDS --per-seed "$PER_SEED" \
    --condition "$cond" $extra --out "$dir" > "$ROOT/$cond.log" 2>&1 &
  local pid=$!
  bash "$REPO/bench/kapro_watchdog.sh" 76 10 kapro_rlr >> "$ROOT/watchdog.log" 2>&1 &
  local wd=$!
  wait $pid
  local rc=$?
  kill $wd 2>/dev/null
  echo "$cond finished rc=$rc"
  return $rc
}

for cond in $CONDITIONS; do
  run_one "$cond" || echo "WARNING: $cond aborted thermally; results partial"
  # Cool to a safe margin before the next condition.
  for i in $(seq 1 60); do
    T=$(cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null|sort -n|tail -1)
    [ $((T/1000)) -le 62 ] && break
    sleep 10
  done
  echo "cooled to $((T/1000))C"
done

for cond in $DISTRACTOR_CONDITIONS; do
  run_one "$cond" "--distractors" || echo "WARNING: $cond aborted thermally"
done

echo "ALL CONDITIONS DONE $(date -u +%FT%TZ)"
