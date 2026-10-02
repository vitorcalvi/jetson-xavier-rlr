#!/bin/bash
# Multi-model KAPRO-RLR run.
#
# Phase 6 of the v2.2 spec: at least 2 independent model configurations for
# L3 robustness; 3+ for H6 correlation analysis. We run 0.5B Qwen as the
# second model after 1.5B Qwen (the canonical first model).
#
# The runner is structurally identical to bench_run_remaining.sh, but
# rotates which model llama-server loads between conditions. We use the
# existing KAPRO --condition parameter to flag the model in the manifest.
set -u
REPO=/home/vitor/jetson-xavier-rlr
OUT_ROOT=/models/bench/kapro-multimodel
LOG=/models/bench/kapro_multimodel.log
mkdir -p "$OUT_ROOT"
CONDITIONS="multimodel_qwen05"
SEEDS="11 23"
PER_SEED=30
KAPRO_TIMEOUT=240

start_llama_qwen05 () {
  pkill -9 -f llama-server 2>/dev/null; sleep 3
  cd /models
  nohup /models/build/llama/bin/llama-server \
    -m /models/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    -ngl 20 -c 4096 --load-mode mmap -b 512 -ub 128 \
    --host 127.0.0.1 --port 8080 --alias Qwen2.5-0.5B \
    > /models/bench/kapro_llama05b.log 2>&1 &
  for j in $(seq 1 40); do
    sleep 3
    if python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3)" 2>/dev/null; then
      echo "$(date -u +%FT%TZ) llama05b healthy" >> "$LOG"
      return 0
    fi
  done
  return 1
}

start_llama_qwen15 () {
  pkill -9 -f llama-server 2>/dev/null; sleep 3
  cd /models
  nohup /models/build/llama/bin/llama-server \
    -m /models/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
    -ngl 20 -c 4096 --load-mode mmap -b 512 -ub 128 \
    --host 127.0.0.1 --port 8080 --alias Qwen2.5-1.5B \
    > /models/bench/kapro_llama15b.log 2>&1 &
  for j in $(seq 1 40); do
    sleep 3
    if python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3)" 2>/dev/null; then
      echo "$(date -u +%FT%TZ) llama15b healthy" >> "$LOG"
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

# Model pair: name -> (model_file, llama_alias, env_alias_for_kapro)
# The KAPRO runner uses --condition full_self_model, so the "condition" is
# a sibling axis. The model name goes in the manifest via the run dir name.
declare -A MODELS=(
  [qwen05]=/models/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
  [qwen15]=/models/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
)

# Run qwen05 first (the new configuration); qwen15 is the existing canonical
# 1.5B run we already have.
for MODEL in qwen05 qwen15; do
  MODEL_FILE=${MODELS[$MODEL]}
  ALIAS=$(basename "$MODEL_FILE" .gguf | sed 's/-Instruct-Q4_K_M//; s/-instruct-q4_k_m//')
  ALIAS=$(echo "$ALIAS" | awk '{print toupper($0)}')
  DIR=$OUT_ROOT/$MODEL
  echo "$(date -u +%FT%TZ) === $MODEL ===" >> "$LOG"
  start_llama_$MODEL
  rm -rf "$DIR"
  cd "$REPO"
  nohup python3 bench/kapro_rlr.py --seeds $SEEDS --per-seed $PER_SEED \
    --condition full_self_model --out "$DIR" > "$OUT_ROOT/$MODEL.log" 2>&1 &
  PID=$!
  bash "$REPO/bench/kapro_watchdog.sh" 72 8 "kapro_rlr|llama-server" >> "$OUT_ROOT/$MODEL.wd.log" 2>&1 &
  WD=$!
  START=$(date +%s)
  while kill -0 $PID 2>/dev/null; do
    if [ $(( $(date +%s) - START )) -gt $KAPRO_TIMEOUT ]; then
      kill -TERM $PID 2>/dev/null; sleep 2; kill -9 $PID 2>/dev/null
      break
    fi
    sleep 10
  done
  kill -9 $WD 2>/dev/null
  N=$(wc -l < "$DIR/scored.jsonl" 2>/dev/null || echo 0)
  echo "$(date -u +%FT%TZ) $MODEL done scored=$N" >> "$LOG"
  pkill -9 -f llama-server 2>/dev/null
  cool
done
pkill -9 -f llama-server 2>/dev/null
echo "$(date -u +%FT%TZ) all conditions complete" >> "$LOG"
