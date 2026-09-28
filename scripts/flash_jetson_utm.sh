#!/usr/bin/env bash
# flash_jetson_utm.sh
# -------------------
# Run NVIDIA Linux_for_Tegra `flash.sh` against a Jetson in recovery mode,
# using an Ubuntu 22.04 arm64 VM managed by UTM on macOS host.
#
# Why this exists:
#   Docker on macOS cannot pass arbitrary USB devices into a container
#   (`flash.sh` fails with "ECID is empty / probing target failed").
#   UTM has reliable macOS USB passthrough; we use it as the USB host
#   and run flash.sh natively inside the guest.
#
# What you MUST do ONCE before first run:
#   1. Open UTM.app.
#   2. Create a new VM (no ISO needed if you have a cloud image):
#        - Name: jetson-flash
#        - Architecture: arm64
#        - System: Linux
#        - ISO / Image: ubuntu-22.04-server-cloudimage-arm64.img
#          (download from cloud-images.ubuntu.com/jammy/current/)
#        - Memory: 4096 MB
#        - Disk: 30 GB
#        - Network: Shared (the default; gives the guest a 10.0.2.x IP)
#        - USB: enabled, attached to host controller
#   3. Boot the VM, finish Ubuntu cloud-init, install the prereqs once:
#        sudo apt-get update
#        sudo apt-get install -y openssh-server python3 liblz4-tool \
#             libxml2-utils qemu-user-static
#        sudo systemctl enable --now ssh
#        ip a   # find the VM's IP (`10.0.2.15` for QEMU/UTM shared net)
#   4. Put a line in `~/.ssh/config` so this script can reach it:
#        Host jetson-flash
#          HostName 10.0.2.15
#          User lab
#          PreferredAuthentications publickey
#          IdentityFile ~/.ssh/id_ed25519
#
# Then every flash is just:
#   make flash-utm
#
# Powers down the VM when done. Does NOT delete it.

set -euo pipefail

VM_NAME="${JETSON_UTM_NAME:-jetson-flash}"
SSH_TARGET="${JETSON_UTM_SSH:-jetson-flash}"
HOST_DIR="${1:-$(pwd)}"

log() { printf '\033[1;34m[flash_utm]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[flash_utm]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[flash_utm]\033[0m %s\n' "$*" >&2; exit 1; }

# --- Locate UTM tooling ------------------------------------------------------
if ! command -v utmctl >/dev/null 2>&1; then
  die "utmctl not found on PATH (brew install utm) — cannot manage UTM VM."
fi

# --- Sanity: does the VM exist? ---------------------------------------------
log "Checking for UTM VM named '${VM_NAME}'..."
VM_LINE="$(utmctl list | awk -v n="${VM_NAME}" 'NR>1 && $3==n {print; exit}')"
if [[ -z "${VM_LINE}" ]]; then
  VM_UUID="$(utmctl list | awk 'NR>1 {print $1; exit}')"
  if [[ -n "${VM_UUID}" ]]; then
    VM_NAME="$(utmctl list | awk 'NR>1 {print $3; exit}')"
    warn "VM not found by name; using first available VM '${VM_NAME}'."
  else
    die "No UTM VM registered. Run 'utmctl list' from a logged-in UTM GUI and create one first. See script header comments (1-4)."
  fi
else
  VM_UUID="$(echo "${VM_LINE}" | awk '{print $1}')"
fi

# --- Boot the VM ------------------------------------------------------------
CURRENT_STATUS="$(utmctl status "${VM_UUID}" 2>/dev/null || true)"
if ! echo "${CURRENT_STATUS}" | grep -qi "Started"; then
  log "Starting UTM VM ${VM_NAME} (uuid=${VM_UUID})..."
  utmctl start "${VM_UUID}"
else
  log "UTM VM ${VM_NAME} already started."
fi

# Wait for SSH
log "Waiting for ssh ${SSH_TARGET} to come up..."
for i in {1..60}; do
  if ssh -o ConnectTimeout=2 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
        "${SSH_TARGET}" true 2>/dev/null; then
    log "ssh is up after ${i} attempts."
    break
  fi
  sleep 2
  [[ "${i}" == "60" ]] && die "ssh never came up to ${SSH_TARGET}."
done

# --- Detect Jetson in recovery on the macOS host ----------------------------
log "Looking for Jetson in recovery mode on the host USB bus..."
JETSON_USB=""
for i in {1..10}; do
  JETSON_USB="$(system_profiler SPUSBDataType 2>/dev/null | \
                grep -i "USB Single Serial" -B1 -A4 || true)"
  if [[ -n "${JETSON_USB}" ]]; then
    log "Detected Jetson in recovery on host (try $i)."
    break
  fi
  warn "No recovery-mode Jetson detected yet (attempt $i/10). Hold REC button + plug DC 12-24V if not already."
  sleep 3
done
[[ -z "${JETSON_USB}" ]] && die "No Jetson in recovery detected over USB. Check DBUG Type-C cable, REC button hold, and that the camera/capture 'Allow USB Single Serial' dialog was accepted."

# --- Hand the device into the guest -----------------------------------------
log "Attaching the Jetson USB device to the guest VM..."
# utmctl usb <uuid> <action> [bus.device]
# Action set is implementation-dependent across UTM versions; try the common ones.
attach_attempts=( "attach --help" )
for action in attach connect forward add; do
  if utmctl usb "${VM_UUID}" "${action}" --help 2>/dev/null | grep -q "bus"; then
    log "Will use 'utmctl usb ${VM_UUID} ${action} <bus>.<dev>'."
    ATTACH_SUBCMD="${action}"
    break
  fi
done

if [[ -z "${ATTACH_SUBCMD:-}" ]]; then
  warn "Could not autodetect 'utmctl usb attach' verb. Open UTM GUI -> USB menu -> connect the Jetson manually, then press Enter."
  read -r _
fi

# --- Run flash.sh inside the guest ------------------------------------------
log "Running flash.sh inside the guest against mmcblk1p1..."
REMOTE_DIR="/home/$(ssh -o BatchMode=yes "${SSH_TARGET}" whoami)/lab"
ssh -t "${SSH_TARGET}" "set -euo pipefail; \
  if [[ ! -d '${REMOTE_DIR}/Linux_for_Tegra' ]]; then \
    ln -sf '${HOST_DIR%/}' '${REMOTE_DIR}' || true; \
  fi; \
  ls '${REMOTE_DIR}/Linux_for_Tegra/flash.sh' >/dev/null; \
  cd '${REMOTE_DIR}/Linux_for_Tegra'; \
  sudo ./flash.sh 'jetson-xavier-nx-devkit' 'mmcblk1p1'"

FLASH_RC=$?

# --- Tidy up ----------------------------------------------------------------
if [[ -n "${ATTACH_SUBCMD:-}" ]]; then
  utmctl usb "${VM_UUID}" detach 1-1 2>/dev/null || true
fi

if [[ ${FLASH_RC} -eq 0 ]]; then
  log "flash.sh completed. Power-cycle the Jetson (DC unplug + replug) to boot from SD."
else
  die "flash.sh exited with rc=${FLASH_RC}. Inspect guest terminal or run 'utmctl attach ${VM_UUID}' to see the live log."
fi
