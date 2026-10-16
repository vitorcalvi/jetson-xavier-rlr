#!/usr/bin/env bash
set -euo pipefail
# Research artifact compiler only. No power/clock mutations or service restart.
if [[ $# -ne 2 ]]; then echo 'usage: export_quant_fp16.sh policy.onnx policy_fp16.engine' >&2; exit 2; fi
INPUT=$1
OUTPUT=$2
[[ -f "$INPUT" && -s "$INPUT" ]] || { echo 'ONNX input absent/empty' >&2; exit 2; }
command -v trtexec >/dev/null 2>&1 || { echo 'trtexec unavailable: install compatible JetPack/TensorRT first' >&2; exit 3; }
[[ ! -e "$OUTPUT" ]] || { echo 'Refuse to overwrite existing engine' >&2; exit 4; }
trtexec --onnx="$INPUT" --fp16 --saveEngine="$OUTPUT" --skipInference
[[ -s "$OUTPUT" ]] || { echo 'TensorRT output missing' >&2; exit 5; }
echo "Engine created: $OUTPUT (NOT accuracy/latency validated)"
