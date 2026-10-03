#!/usr/bin/env bash
set -euo pipefail

LLAMA_BIN="${LLAMA_BIN:-/models/build/llama/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-/models/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CTX="${CTX:-2048}"
BATCH="${BATCH:-512}"
UBATCH="${UBATCH:-128}"
GPU_LAYERS="${GPU_LAYERS:-99}"
LOG_DIR="${LOG_DIR:-/models/rlr}"

if [ ! -x "$LLAMA_BIN" ]; then
  echo "llama-server not found or not executable: $LLAMA_BIN" >&2
  echo "Run scripts/build_llama_xavier.sh first." >&2
  exit 2
fi
if [ ! -f "$MODEL_PATH" ]; then
  echo "model not found: $MODEL_PATH" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

exec "$LLAMA_BIN" \
  -m "$MODEL_PATH" \
  -ngl "$GPU_LAYERS" \
  -c "$CTX" \
  -b "$BATCH" \
  -ub "$UBATCH" \
  --host "$HOST" \
  --port "$PORT"
