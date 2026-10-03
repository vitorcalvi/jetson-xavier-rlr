#!/bin/sh
set -eu

: "${RLR_LLAMA_SERVER:?RLR_LLAMA_SERVER is required}"
: "${RLR_MODEL:?RLR_MODEL is required}"
RLR_HOST=${RLR_HOST:-127.0.0.1}
RLR_PORT=${RLR_PORT:-8080}
RLR_GPU_LAYERS=${RLR_GPU_LAYERS:-20}
RLR_CONTEXT=${RLR_CONTEXT:-4096}
RLR_BATCH=${RLR_BATCH:-512}
RLR_UBATCH=${RLR_UBATCH:-128}

case "$RLR_LLAMA_SERVER" in /*) ;; *) echo "RLR_LLAMA_SERVER must be absolute" >&2; exit 64;; esac
case "$RLR_MODEL" in /*) ;; *) echo "RLR_MODEL must be absolute" >&2; exit 64;; esac
[ -x "$RLR_LLAMA_SERVER" ] || { echo "llama-server is not executable" >&2; exit 66; }
[ -r "$RLR_MODEL" ] || { echo "model is not readable" >&2; exit 66; }

exec "$RLR_LLAMA_SERVER" -m "$RLR_MODEL" -ngl "$RLR_GPU_LAYERS" -c "$RLR_CONTEXT" \
  --load-mode mmap -b "$RLR_BATCH" -ub "$RLR_UBATCH" --host "$RLR_HOST" --port "$RLR_PORT"
