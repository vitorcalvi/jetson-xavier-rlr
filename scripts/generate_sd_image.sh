#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/workspace"
L4T_DIR="${WORKSPACE}/Linux_for_Tegra"
BOARD="${BOARD:-jetson-xavier-nx-devkit}"
IMAGE_NAME="${IMAGE_NAME:-jetson_xavier_nx_lean_r35}"

echo "=== Step 4: Generate and flash SD card image ==="
cd "${L4T_DIR}"

# Verify jetson-disk-image-creator.sh exists
if [[ ! -x "./tools/jetson-disk-image-creator.sh" ]]; then
  echo "ERROR: tools/jetson-disk-image-creator.sh not found or not executable"
  exit 1
fi

# 4.2 Generate SD image
echo "Generating SD image with jetson-disk-image-creator.sh..."
echo "Board: ${BOARD}"
sudo ./tools/jetson-disk-image-creator.sh -d SD \
  -o "${IMAGE_NAME}.img" \
  -b "${BOARD}"

IMG_PATH="${L4T_DIR}/${IMAGE_NAME}.img"
if [[ ! -f "${IMG_PATH}" ]]; then
  # Try workspace fallback
  IMG_PATH="${WORKSPACE}/${IMAGE_NAME}.img"
  if [[ ! -f "${IMG_PATH}" ]]; then
    echo "ERROR: Image not created at ${L4T_DIR}/${IMAGE_NAME}.img or ${WORKSPACE}/${IMAGE_NAME}.img"
    exit 1
  fi
fi

echo "Image created: ${IMG_PATH}"
ls -lh "${IMG_PATH}"

# 4.3 Flash to microSD card (macOS host step — container can't access host disk)
if [[ -z "${SD_DEVICE:-}" ]]; then
  echo ""
  echo "SD_DEVICE not set. Available block devices inside container:"
  lsblk 2>/dev/null || echo "lsblk not available"
  echo ""
  echo "NOTE: Flash from macOS host after exiting container:"
  echo "  diskutil list"
  echo "  diskutil unmountDisk /dev/diskN"
  echo "  sudo dd if=${IMG_PATH} of=/dev/rdiskN bs=4m status=progress"
  echo "  sync"
  exit 0
fi

echo "Flashing ${IMG_PATH} to ${SD_DEVICE}..."
echo "WARNING: ALL DATA ON ${SD_DEVICE} WILL BE ERASED"
read -p "Press Enter to confirm, or Ctrl+C to abort..."

sudo dd if="${IMG_PATH}" of="${SD_DEVICE}" bs=4M status=progress conv=fsync
sync
echo "Flash complete."

sudo eject "${SD_DEVICE}" 2>/dev/null || true
echo "SD card ejected. Insert into Xavier NX module and power on."
