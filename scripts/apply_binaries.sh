#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/workspace"
L4T_DIR="${WORKSPACE}/Linux_for_Tegra"
ROOTFS_DIR="${L4T_DIR}/rootfs"

echo "=== Step 3.1: Apply NVIDIA binaries ==="
cd "${L4T_DIR}"

# Verify rootfs exists
if [[ ! -d "${ROOTFS_DIR}" ]] || [[ -z "$(ls -A "${ROOTFS_DIR}" 2>/dev/null)" ]]; then
  echo "ERROR: rootfs/ directory is missing or empty in ${L4T_DIR}"
  exit 1
fi

# Ensure qemu-aarch64-static is available in rootfs for chroot
if [[ -f "${ROOTFS_DIR}/usr/bin/qemu-aarch64-static" ]]; then
  echo "qemu-aarch64-static already present in rootfs"
else
  echo "Copying qemu-aarch64-static into rootfs..."
  mkdir -p "${ROOTFS_DIR}/usr/bin"
  cp /usr/bin/qemu-aarch64-static "${ROOTFS_DIR}/usr/bin/"
fi

# Apply NVIDIA binaries
echo "Running apply_binaries.sh..."
sudo ./apply_binaries.sh

echo "=== NVIDIA binaries applied ==="
