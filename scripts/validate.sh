#!/usr/bin/env bash
set -euo pipefail

# Validation script — run ON THE JETSON after first boot via SSH or serial.
# Copy this script to the Jetson and execute it there.

echo "=== Step 5: First-boot validation ==="

echo "--- OS / Kernel ---"
uname -a

echo "--- L4T / JetPack version ---"
if command -v jetson_release &>/dev/null; then
  jetson_release -v
else
  echo "jetson_release not found; checking /etc/nv_tegra_release..."
  cat /etc/nv_tegra_release 2>/dev/null || echo "nv_tegra_release not found"
fi

echo "--- Root filesystem device ---"
df -h /
lsblk

echo "--- Memory footprint ---"
free -h

echo "--- Disk usage ---"
df -h

echo "--- Boot device check (expect mmcblk1 for SD) ---"
ROOT_DEV=$(findmnt -n -o SOURCE /)
echo "Root device: ${ROOT_DEV}"
if echo "${ROOT_DEV}" | grep -q "mmcblk1"; then
  echo "PASS: Root is on SD card (mmcblk1)"
elif echo "${ROOT_DEV}" | grep -q "mmcblk0"; then
  echo "FAIL: Root is on eMMC (mmcblk0) — SD boot did not succeed"
else
  echo "UNKNOWN: Root device is ${ROOT_DEV}"
fi

echo "--- GPU compute stack ---"
if command -v nvcc &>/dev/null; then
  nvcc --version
else
  echo "nvcc not found — CUDA toolkit may be missing"
fi

echo "--- SSH service ---"
if systemctl is-active --quiet ssh; then
  echo "PASS: SSH is active"
else
  echo "FAIL: SSH is not active"
fi

echo "--- Headless (no GDM/X11) ---"
if systemctl is-active --quiet gdm 2>/dev/null; then
  echo "WARN: GDM is active — display stack not stripped"
else
  echo "PASS: GDM is inactive — headless OK"
fi

echo "--- Boot time (since last boot) ---"
systemd-analyze 2>/dev/null || echo "systemd-analyze not available"

echo "--- Image size ---"
du -sh / 2>/dev/null || true

echo "=== Validation complete ==="
