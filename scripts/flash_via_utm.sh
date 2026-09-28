#!/usr/bin/env bash
# flash_via_utm.sh
# ----------------
# Runs inside the UTM-Linux guest to dry-run and then (after gate confirmation)
# flash the Jetson in recovery over USB. The host-side USB device is forwarded
# into the guest by UTM (USB controller enabled in the wizard).
#
# Usage on the GUEST:
#   bash flash_via_utm.sh --dry-run    # parse partition map, no writes
#   bash flash_via_utm.sh --flash      # actual flash.sh invocation
# Pre-req: BSP already extracted into ~/Linux_for_Tegra/ inside the guest.

set -euo pipefail

BSP_DIR="${BSP_DIR:-$HOME/Linux_for_Tegra}"
BOARD="${BOARD:-jetson-xavier-nx-devkit}"
TARGET="${TARGET:-mmcblk1p1}"

if [[ ! -x "${BSP_DIR}/flash.sh" ]]; then
  echo "FATAL: ${BSP_DIR}/flash.sh not found or not executable." >&2
  exit 1
fi
if [[ ! -d "${BSP_DIR}/rootfs" ]]; then
  echo "FATAL: ${BSP_DIR}/rootfs missing — re-extract the BSP tarball." >&2
  exit 1
fi

cd "${BSP_DIR}"

check_usb_device() {
  echo "=== USB bus probe ==="
  lsusb 2>/dev/null || true
  echo "---"
  ls /dev/bus/usb 2>&1 || true
  echo "---"
  if ! lsusb 2>/dev/null | grep -qi "NVIDIA\|single serial"; then
    echo "WARN: Jetson in RCM not visible to guest. Confirm host-side:"
    echo "  - macOS connected to DBUG Type-C"
    echo "  - UTM VM's USB controller = Apple-USB-Host (passthrough)"
    echo "  - USB Single Serial dialog accepted on Mac"
    exit 2
  fi
  echo "OK: NVIDIA Jetson visible."
}

verify_layout() {
  echo "=== Layout dry-run for ${BOARD} on ${TARGET} ==="
  sudo ./flash.sh --no-flash --no-systemimg -t "${BOARD}" "${TARGET}"
}

run_flash() {
  echo "=== Real flash for ${BOARD} on ${TARGET} ==="
  sudo ./flash.sh "${BOARD}" "${TARGET}"
}

case "${1:-}" in
  --dry-run)
    check_usb_device
    verify_layout
    echo "Dry-run OK — partition map printed above."
    ;;
  --flash)
    check_usb_device
    echo "=== About to write to the Jetson (${TARGET}) — this is destructive. ==="
    echo "Press Ctrl-C within 5 seconds to abort..."
    sleep 5
    run_flash
    ;;
  *)
    echo "Usage: $0 {--dry-run|--flash}" >&2
    exit 1
    ;;
esac
