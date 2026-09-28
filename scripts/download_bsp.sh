#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="/workspace"
L4T_DIR="${WORKSPACE}/Linux_for_Tegra"
ROOTFS_DIR="${L4T_DIR}/rootfs"
BSP_URL="${BSP_URL:-}"
ROOTFS_URL="${ROOTFS_URL:-}"

echo "=== Download BSP and Sample Rootfs ==="
echo "Workspace: ${WORKSPACE}"

mkdir -p "${L4T_DIR}"
cd "${WORKSPACE}"

# Download Driver Package (BSP)
if [[ -n "${BSP_URL}" ]]; then
  echo "Downloading Driver Package (BSP)..."
  wget -O Linux_for_Tegra.tar.bz2 "${BSP_URL}"
  mkdir -p Linux_for_Tegra_extract
  tar -xjf Linux_for_Tegra.tar.bz2 -C Linux_for_Tegra_extract --strip-components=1
  # Move contents up if nested
  if [[ -d "Linux_for_Tegra_extract/Linux_for_Tegra" ]]; then
    mv Linux_for_Tegra_extract/Linux_for_Tegra/* "${L4T_DIR}/"
  else
    mv Linux_for_Tegra_extract/* "${L4T_DIR}/"
  fi
  rm -rf Linux_for_Tegra_extract
else
  echo "BSP_URL not set. Place Linux_for_Tegra/ in ${WORKSPACE}/"
  exit 1
fi

# Download Sample Root Filesystem
if [[ -n "${ROOTFS_URL}" ]]; then
  echo "Downloading Sample Root Filesystem..."
  wget -O rootfs.tar.bz2 "${ROOTFS_URL}"
  mkdir -p "${ROOTFS_DIR}"
  tar -xjf rootfs.tar.bz2 -C "${ROOTFS_DIR}" --strip-components=1
else
  echo "ROOTFS_URL not set. Place rootfs/ in ${L4T_DIR}/"
  exit 1
fi

echo "=== BSP and rootfs extracted successfully ==="
echo "Next: bash /workspace/scripts/apply_binaries.sh"
