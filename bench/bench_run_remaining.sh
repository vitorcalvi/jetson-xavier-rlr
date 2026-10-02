#!/bin/bash
# Run all remaining KAPRO-RLR conditions sequentially, with a thermal
# cool-down between conditions and a single independent watchdog that covers
# any process that could heat the device.
#
# Each condition reuses its own output directory so --resume is per-condition.
# The watcher updates the canonical local mirror every cool-down so the
# report can be rebuilt from a single source while the device runs.
set -u
DEST="vitor@192.168.1.49"
REPO=/home/vitor/jetson-xavier-rlr
OUT_ROOT=/models/bench/kapro-conditions
LOG=/models/bench/bench_run_remaining.log

# Priority order. full_self_model already ran 13 cases; we resume from there.
# Priority order. full_self_model resumes from the 13 cases already scored
# on the device. no_self_model is the H1-ablation, no_recent_actions is the
# H2-ablation, no_capabilities/no_goals/no_memories test which field of the
# self-model matters most. stale_telemetry is a different branch of the
# KAPRO apparatus (governor branch). distractors is a different branch of
# the model branch (selection). The no_*/corrupt_*/shuffled_ variants are
# left for a longer time budget.
CONDITIONS="full_self_model no_self_model no_recent_actions no_capabilities no_goals no_memories stale_telemetry distractors"
SEEDS="11 23 42 71 101"
PER_SEED=60
KAPRO_TIMEOUT=240   # per-attempt wall clock, in seconds

# Conservative launch: device is 67 C; cool to 55 C before anything starts.
for i in $(seq 1 60); do
  T=$(cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null|sort -n|tail -1)
  [ $((T/1000)) -le 55 ] && break
  sleep 10
done
echo "$(date -u +%FT%TZ) starting at $((T/1000))C" >> "$LOG"

start_llama () {
  pkill -9 -f llama-server 2>/dev/null; sleep 3
  cd /models
  nohup /models/build/llama/bin/llama-server \
    -m /models/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
    -ngl 20 -c 4096 --load-mode mmap -b 512 -ub 128 \
    --host 127.0.0.1 --port 8080 --alias Qwen2.5-1.5B \
    > /models/bench/kapro_llama.log 2>&1 &
  for j in $(seq 1 40); do
    sleep 3
    if python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3)" 2>/dev/null; then
      echo "$(date -u +%FT%TZ) llama healthy after ${j}x3s" >> "$LOG"
      return 0
    fi
  done
  return 1
}

cool () {
  for i in $(seq 1 90); do
    T=$(cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null|sort -n|tail -1)
    [ $((T/1000)) -le 55 ] && return 0
    sleep 10
  done
}

run_condition () {
  local cond="$1" extra="${2:-}"
  local dir="$OUT_ROOT/$cond"
  local pid
  # The model server can die between conditions (observed during no_self_model:
  # all 300 attempts returned Connection refused). Restart it before each
  # condition to start from a known-good endpoint.
  start_llama
  echo "$(date -u +%FT%TZ) === $cond ===" >> "$LOG"
  # Resume from the canonical full_self_model run if this is the first
  # condition and the existing scored.jsonl is present. Avoids re-running
  # the 13 cases that already exist on the device.
  if [ "$cond" = "full_self_model" ] && [ -f /models/bench/kapro-full/scored.jsonl ] && [ ! -f "$dir/scored.jsonl" ]; then
    mkdir -p "$dir"
    cp /models/bench/kapro-full/scored.jsonl "$dir/scored.jsonl"
    cp /models/bench/kapro-full/cases.jsonl "$dir/cases.jsonl" 2>/dev/null
    cp /models/bench/kapro-full/knowing.jsonl "$dir/knowing.jsonl" 2>/dev/null
    cp /models/bench/kapro-full/acting.jsonl "$dir/acting.jsonl" 2>/dev/null
    echo "$(date -u +%FT%TZ) $cond seeded with $(wc -l < $dir/scored.jsonl) prior cases" >> "$LOG"
  else
    rm -rf "$dir"
  fi
  cd "$REPO"
  nohup python3 bench/kapro_rlr.py --seeds $SEEDS --per-seed $PER_SEED \
    --condition "$cond" $extra --out "$dir" --resume > "$OUT_ROOT/$cond.log" 2>&1 &
  pid=$!
  bash "$REPO/bench/kapro_watchdog.sh" 72 8 "kapro_rlr|llama-server" >> "$OUT_ROOT/$cond.wd.log" 2>&1 &
  local wd=$!
  local start=$(date +%s)
  while kill -0 $pid 2>/dev/null; do
    local now=$(date +%s)
    local elapsed=$((now - start))
    if [ $elapsed -gt $KAPRO_TIMEOUT ]; then
      echo "$(date -u +%FT%TZ) $cond timed out at ${elapsed}s, killing" >> "$LOG"
      kill -TERM $pid 2>/dev/null; sleep 2; kill -9 $pid 2>/dev/null
      break
    fi
    sleep 10
  done
  kill -9 $wd 2>/dev/null
  local scored=$(wc -l < "$dir/scored.jsonl" 2>/dev/null || echo 0)
  echo "$(date -u +%FT%TZ) $cond done scored=$scored" >> "$LOG"
}

start_llama
for cond in $CONDITIONS; do
  case "$cond" in
    distractors) run_condition "$cond" "--distractors" ;;
    *)            run_condition "$cond" "" ;;
  esac
  # Mirror artifacts locally so the report can be rebuilt offline.
  rsync -az "$OUT_ROOT/$cond/" "/tmp/kapro/$cond/" 2>/dev/null
  cool
done

# KAware preview continuation: best-effort, single attempt, 3-min budget.
echo "$(date -u +%FT%TZ) starting KAware preview continuation" >> "$LOG"
cd "$REPO"
nohup python3 bench/kapro_official.py \
  --dataset /models/bench/KAware \
  --out /models/bench/kapro-official/preview2.json \
  > /models/bench/kapro-official.continue.log 2>&1 &
pid=$!
bash "$REPO/bench/kapro_watchdog.sh" 69 8 "kapro_official|llama-server" >> "$LOG" 2>&1 &
wd=$!
wait $pid
kill -9 $wd 2>/dev/null
echo "$(date -u +%FT%TZ) KAware preview continuation done" >> "$LOG"

pkill -9 -f llama-server 2>/dev/null
echo "$(date -u +%FT%TZ) all conditions complete" >> "$LOG"