#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/workspace"
L4T_DIR="${WORKSPACE}/Linux_for_Tegra"
BOARD="${BOARD:-jetson-xavier-nx-devkit}"
IMAGE_NAME="${IMAGE_NAME:-jetson_xavier_nx_lean_r35}"

echo "=== Step 4: Generate SD card image (skip strip) ==="
cd "${L4T_DIR}"

if [[ ! -x "./tools/jetson-disk-image-creator.sh" ]]; then
  echo "ERROR: tools/jetson-disk-image-creator.sh not found"
  exit 1
fi

echo "Board: ${BOARD}"
echo "Rootfs size: $(du -sh rootfs | cut -f1)"

sudo ./tools/jetson-disk-image-creator.sh -d SD \
  -o "${IMAGE_NAME}.img" \
  -b "${BOARD}" 2>&1 | tail -80

IMG_PATH="${L4T_DIR}/${IMAGE_NAME}.img"
if [[ ! -f "${IMG_PATH}" ]]; then
  IMG_PATH="${WORKSPACE}/${IMAGE_NAME}.img"
fi

if [[ -f "${IMG_PATH}" ]]; then
  echo "=== Image created: ${IMG_PATH} ==="
  ls -lh "${IMG_PATH}"
else
  echo "ERROR: Image not created"
  exit 1
fi
