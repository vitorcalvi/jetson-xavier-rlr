#!/usr/bin/env bash
set -euo pipefail

BENCH_BIN="${BENCH_BIN:-/models/build/llama/bin/llama-bench}"
MODEL_PATH="${MODEL_PATH:-/models/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
GEN_TOKENS="${GEN_TOKENS:-128}"
GPU_LAYERS="${GPU_LAYERS:-99}"
REPETITIONS="${REPETITIONS:-3}"

if [ ! -x "$BENCH_BIN" ]; then
  echo "llama-bench not found or not executable: $BENCH_BIN" >&2
  exit 2
fi
if [ ! -f "$MODEL_PATH" ]; then
  echo "model not found: $MODEL_PATH" >&2
  exit 2
fi

exec "$BENCH_BIN" \
  -m "$MODEL_PATH" \
  -ngl "$GPU_LAYERS" \
  -p "$PROMPT_TOKENS" \
  -n "$GEN_TOKENS" \
  -r "$REPETITIONS"
