#!/bin/bash
# Run all 8 KAware preview cases with 16K context and preflight admission.
#
# The v2.2 spec's Phase 1/2 fix: the 4K launch parameter rejects most preview
# tool-calling prompts as "context exceeded". This wraps kapro_official with
# a 16K llama-server, persists per-case raw generations and tool traces, and
# returns 0..8 case statuses by index.
set -u
REPO=/home/vitor/jetson-xavier-rlr
OUT_DIR=/models/bench/kapro-official/preview_16k
mkdir -p "$OUT_DIR"
LOG=/models/bench/kapro_official_16k.log

for i in $(seq 1 90); do
  T=$(cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null|sort -n|tail -1)
  [ $((T/1000)) -le 54 ] && break
  sleep 10
done
echo "$(date -u +%FT%TZ) starting at $((T/1000))C" >> "$LOG"

pkill -9 -f llama-server 2>/dev/null; sleep 3
cd /models
nohup /models/build/llama/bin/llama-server \
  -m /models/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
  -ngl 20 -c 16384 --load-mode mmap -b 512 -ub 128 \
  --host 127.0.0.1 --port 8080 --alias Qwen2.5-1.5B \
  > /models/bench/kapro_llama16k.log 2>&1 &
for j in $(seq 1 40); do sleep 3; python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3)" 2>/dev/null && { echo HEALTHY; break; }; done

cd "$REPO"
nohup bash bench/kapro_watchdog.sh 70 8 "kapro_official|llama-server" > "$OUT_DIR/wd.log" 2>&1 &
nohup python3 bench/kapro_official.py --dataset /models/bench/KAware \
  --out "$OUT_DIR/preview.json" > "$LOG" 2>&1 &
PID=$!
START=$(date +%s)
while kill -0 $PID 2>/dev/null; do
  if [ $(( $(date +%s) - START )) -gt 540 ]; then
    kill -TERM $PID 2>/dev/null; sleep 2; kill -9 $PID 2>/dev/null
    break
  fi
  sleep 10
done
N=$(wc -l < "$OUT_DIR/preview.partial.jsonl" 2>/dev/null || echo 0)
echo "$(date -u +%FT%TZ) preview16k done scored=$N" >> "$LOG"
pkill -9 -f llama-server 2>/dev/null
