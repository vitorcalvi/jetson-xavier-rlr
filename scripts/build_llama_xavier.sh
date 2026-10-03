#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${LLAMA_SRC_DIR:-/models/src/llama.cpp}"
BUILD_DIR="${LLAMA_BUILD_DIR:-/models/build/llama}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
JOBS="${JOBS:-2}"
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"

if ! command -v "$CMAKE_BIN" >/dev/null 2>&1; then
  echo "cmake not found: $CMAKE_BIN" >&2
  exit 2
fi

if [ ! -d "$SRC_DIR/.git" ]; then
  mkdir -p "$(dirname "$SRC_DIR")"
  git clone --depth 1 "$LLAMA_REPO" "$SRC_DIR"
fi

"$CMAKE_BIN" -S "$SRC_DIR" -B "$BUILD_DIR" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=72 \
  -DCMAKE_BUILD_TYPE=Release

"$CMAKE_BIN" --build "$BUILD_DIR" --target llama-server llama-bench -j"$JOBS"

"$BUILD_DIR/bin/llama-server" --version || true
"$BUILD_DIR/bin/llama-bench" --help >/dev/null

echo "Built Xavier NX llama.cpp CUDA binaries in $BUILD_DIR/bin"
