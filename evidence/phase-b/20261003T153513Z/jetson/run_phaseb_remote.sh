#!/usr/bin/env bash
set -u
ROOT=/models/rlr/phaseb-accelerated-20261003T153513Z
CAND=/models/rlr/governor-candidates/hardening-1cdb094
LLAMA=/models/build/llama/bin/llama-server
MODEL=/models/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
TRACE=/models/rlr/governor-runs/smoke300-1cdb094-20261003-151218/thermal.csv
EVENTS="$ROOT/events.jsonl"
START_EPOCH=$(date +%s)
GLOBAL_SECONDS=2700
DEADLINE=$((START_EPOCH+GLOBAL_SECONDS))
THERMAL_ABORT=72
COOL_START=57

jlog(){ printf '%s\n' "$1" | tee -a "$EVENTS"; }
max_temp(){ python3 - <<'PY'
from pathlib import Path
v=[]
for p in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
    try:
        t=(p.parent/'type').read_text().strip()
        if 'PMIC' in t: continue
        v.append(int(p.read_text())/1000.0)
    except: pass
print(max(v) if v else 999.0)
PY
}
now_iso(){ date -u +%Y-%m-%dT%H:%M:%SZ; }
remaining(){ echo $((DEADLINE-$(date +%s))); }
cleanup_model(){
  if [ -n "${MODEL_PID:-}" ] && kill -0 "$MODEL_PID" 2>/dev/null; then kill -TERM "$MODEL_PID" 2>/dev/null || true; sleep 1; kill -KILL "$MODEL_PID" 2>/dev/null || true; wait "$MODEL_PID" 2>/dev/null || true; fi
  MODEL_PID=""
  # Only clean the exact benchmark model executable if one escaped ownership.
  pids=$(pgrep -f '^/models/build/llama/bin/llama-server -m /models/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf ' || true)
  [ -z "$pids" ] || kill -TERM $pids 2>/dev/null || true
}
trap 'cleanup_model' EXIT INT TERM
wait_cool(){
  local label="$1" t
  while :; do
    [ "$(remaining)" -gt 120 ] || return 90
    t=$(max_temp)
    echo "COOL label=$label temp_c=$t remaining_s=$(remaining)"
    awk -v t="$t" -v c="$COOL_START" 'BEGIN{exit !(t<=c)}' && return 0
    sleep 5
  done
}
start_model(){
  local label="$1"
  cleanup_model
  "$LLAMA" -m "$MODEL" -ngl 20 -c 4096 --load-mode mmap -b 512 -ub 128 --host 127.0.0.1 --port 8080 --alias Qwen2.5-1.5B >"$ROOT/llama-$label.log" 2>&1 &
  MODEL_PID=$!
  for i in $(seq 1 60); do
    kill -0 "$MODEL_PID" 2>/dev/null || return 91
    python3 - <<'PY' >/dev/null 2>&1 && return 0
import urllib.request
urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=2).read()
PY
    sleep 2
  done
  return 92
}
run_block(){
  local label="$1" mode="$2" condition="$3" extra="$4" attempt=1 rc t peak thermal
  while [ "$attempt" -le 2 ]; do
    [ "$(remaining)" -gt 180 ] || { echo "GLOBAL_CUTOFF before $label"; return 99; }
    wait_cool "$label-a$attempt" || return $?
    t=$(max_temp)
    jlog "{\"ts\":\"$(now_iso)\",\"event\":\"START\",\"label\":\"$label\",\"attempt\":$attempt,\"temp_c\":$t,\"remaining_s\":$(remaining)}"
    start_model "$label-a$attempt" || { rc=$?; jlog "{\"ts\":\"$(now_iso)\",\"event\":\"MODEL_START_FAIL\",\"label\":\"$label\",\"attempt\":$attempt,\"rc\":$rc}"; cleanup_model; attempt=$((attempt+1)); continue; }
    out="$ROOT/${label}.jsonl"
    [ "$attempt" -eq 1 ] || out="$ROOT/${label}.retry${attempt}.jsonl"
    cmd=(python3 "$ROOT/phaseb_infer.py" --mode "$mode" --out "$out")
    [ -z "$condition" ] || cmd+=(--condition "$condition")
    if [ "$mode" = action ]; then cmd+=(--cases "$ROOT/public_cases.jsonl"); fi
    if [ "$mode" = forecast ]; then cmd+=(--trace "$TRACE"); fi
    timeout --signal=TERM --kill-after=5 300 env PHASEB_CANDIDATE="$CAND" "${cmd[@]}" >"$ROOT/${label}.stdout.a${attempt}" 2>"$ROOT/${label}.stderr.a${attempt}" &
    RUN_PID=$!
    peak=$(max_temp); thermal=0
    while kill -0 "$RUN_PID" 2>/dev/null; do
      t=$(max_temp)
      peak=$(awk -v a="$peak" -v b="$t" 'BEGIN{print (b>a?b:a)}')
      echo "RUN label=$label attempt=$attempt temp_c=$t peak_c=$peak remaining_s=$(remaining)"
      if awk -v t="$t" -v a="$THERMAL_ABORT" 'BEGIN{exit !(t>=a)}'; then
        thermal=1; kill -TERM "$RUN_PID" 2>/dev/null || true; sleep 2; kill -KILL "$RUN_PID" 2>/dev/null || true; break
      fi
      if [ "$(remaining)" -le 90 ]; then kill -TERM "$RUN_PID" 2>/dev/null || true; break; fi
      sleep 3
    done
    wait "$RUN_PID"; rc=$?
    cleanup_model
    t=$(max_temp)
    jlog "{\"ts\":\"$(now_iso)\",\"event\":\"END\",\"label\":\"$label\",\"attempt\":$attempt,\"rc\":$rc,\"thermal_abort\":$thermal,\"peak_temp_c\":$peak,\"end_temp_c\":$t,\"remaining_s\":$(remaining)}"
    if [ "$rc" -eq 0 ] && [ "$thermal" -eq 0 ]; then return 0; fi
    [ "$thermal" -eq 0 ] || return "$rc"
    attempt=$((attempt+1))
  done
  return "$rc"
}

echo "PHASEB_REMOTE_START=$(now_iso) deadline_epoch=$DEADLINE"
echo "GUARD=$(systemctl is-active thermal-guard.service 2>/dev/null || true) ALLOW24=$(test -e /home/vitor/.rlr_allow_24h && echo true || echo false)"
nvpmodel -q 2>/dev/null | head -2 || true
sha256sum "$CAND/bench/round3_config.json" "$ROOT/prereg.json" "$ROOT/public_cases.jsonl" "$ROOT/phaseb_infer.py" > "$ROOT/start_hashes.txt"

# Frozen randomized action order.
for cond in B2 NO_SELF_MODEL B2n B0 B1 B0F; do
  run_block "action-$cond" action "$cond" "" || echo "BLOCK_FAIL action-$cond rc=$?"
done
# L2 self-state track: the B2n manipulation is active here.
for cond in B2 B2n NO_SELF_MODEL; do
  run_block "selfstate-$cond" selfstate "$cond" "" || echo "BLOCK_FAIL selfstate-$cond rc=$?"
done
# F1.3 forecast diagnostics vs persistence and linear baselines.
if [ -f "$TRACE" ]; then
  run_block "forecast" forecast "" "" || echo "BLOCK_FAIL forecast rc=$?"
else
  jlog "{\"ts\":\"$(now_iso)\",\"event\":\"FORECAST_BLOCKED\",\"reason\":\"trace_missing\"}"
fi
cleanup_model
echo "PHASEB_REMOTE_END=$(now_iso) remaining_s=$(remaining) temp_c=$(max_temp)"
echo "GUARD_END=$(systemctl is-active thermal-guard.service 2>/dev/null || true) ALLOW24_END=$(test -e /home/vitor/.rlr_allow_24h && echo true || echo false)"
