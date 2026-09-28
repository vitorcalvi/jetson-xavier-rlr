# Jetson Xavier NX Lean SD-Boot Image — Build Report

**Date:** 2026-09-14
**Target:** Jetson Xavier NX Devkit on DFRobot DFR1242 carrier
**L4T:** r35.6.1 / JetPack 5.1.2
**Host:** macOS arm64, Docker Desktop with `--platform linux/amd64` + QEMU emulation

---

## 1. Final Deliverable

| Item | Path | Size |
|------|------|------|
| SD image | `Linux_for_Tegra/jetson_xavier_nx_lean_r35.img` | 7.4 GB |
| MD5 | `9948f0df4a1691f3756b48582054b7c4` | — |

The image is a raw GPT-partitioned SD image, ready to flash to a microSD card.

---

## 2. Build Steps Completed

### Step 1 — Docker image
Built `jetson-build:r35.6.1` (Ubuntu 20.04 x86_64 with qemu-aarch64-static) to run the NVIDIA toolchain under QEMU emulation on Apple Silicon.

### Step 2 — BSP + rootfs
Downloaded and extracted JetPack 5.1.2 BSP and rootfs into `Linux_for_Tegra/`.

### Step 3 — Apply NVIDIA binaries
`apply_binaries.sh` succeeded after copying `qemu-aarch64-static` into the rootfs.

### Step 4 — Rootfs DNS fix
Fixed rootfs `resolv.conf` (127.0.0.53 → 8.8.8.8) so chroot networking works.

### Step 5 — Headless strip
Used `strip_rootfs_no_apt.sh` (manual deletion, no apt):
- Removed GNOME/GDM/X11/LightDM, CUDA sample apps, docs, locales, caches
- Created user `jetson` / password `<DEVICE_PASSWORD>`
- Enabled SSH, disabled unnecessary services, created `jetson-clocks.service`
- Final rootfs size: 6.1 GB

### Step 6 — SD image generation
Patched `jetson-disk-image-creator.sh` `write_partitions()` to use `dd seek` instead of `losetup -P`. The `losetup -P` path created wrong partition sizes for `kernel_b` (20 MB instead of 64 MB), causing "No space left on device". All partitions written successfully; final image 7.4 GB.

---

## 3. Critical Fix: `losetup -P` → `dd seek`

**File:** `Linux_for_Tegra/tools/jetson-disk-image-creator.sh`, `write_partitions()`

**Problem:** Under QEMU/Docker emulation, `losetup -P` produced incorrect partition sizes — `kernel_b` was created at 20 MB instead of the required 64 MB. This caused `mkfs.vfat` and the final `dd` of the rootfs to fail with "No space left on device".

**Fix:** Replaced the loop-device-based partition writer with direct `dd seek` writes, which bypasses loop device sizing entirely and works correctly under emulation.

---

## 4. Blocked / Not Used

- `strip_rootfs.sh` (apt-based variant): fails with DNS resolution errors inside the QEMU container. Docker container also crashes with "unexpected EOF" under QEMU when running heavy apt operations. Not used; the no-apt variant succeeded instead.

---

## 5. Next Steps: Flash & Boot

### 5a. Flash microSD from macOS
```bash
diskutil list                                   # find SD device
diskutil unmountDisk /dev/diskN
sudo dd if=jetson_xavier_nx_lean_r35.img of=/dev/rdiskN bs=4m status=progress
sync
```

### 5b. Boot
The Xavier NX devkit on the DFRobot DFR1242 **defaults to eMMC boot** if the SD
card isn't detected first by QSPI's boot-order table — and the QSPI baked by
`jetson-disk-image-creator.sh` keeps the factory order (`eMMC, SD`). Symptom:
power-on → board boots eMMC rescue image, not our SD image.

Fix: re-flash the bootloader to retarget `mmcblk1` first. This requires
running `Linux_for_Tegra/flash.sh` against the board in **recovery mode**.

### 5c. Recovery entry on the DFR1242
1. Power off (unplug DC 12–24V).
2. Set the boot-mode slide switch to **Recovery** (silk-screen `REC` near the DBUG Type-C).
3. **Hold the REC tactile button** while plugging DC power.
4. Release REC ~2 s after power stabilises.
5. Plug the **DBUG&Recovery** Type-C (on the carrier — *not* the USB-A ports) into the Mac.
6. macOS will pop a single dialog: “Allow USB Single Serial to connect?” → click **Allow**.

Verify on the Mac:
```bash
ioreg -l | grep -A2 -i "USB Single Serial"      # vendor 0x1A86, product "USB Single Serial"
system_profiler SPUSBDataType | grep -A4 nvidia # explicit NVIDIA enumeration
```

### 5d. Flash from a UTM-managed Ubuntu VM (recommended path on macOS)
**Why a VM and not the build container:** Docker Desktop on macOS does *not*
forward arbitrary USB devices into the container. The host sees the
recovery-mode Jetson (`ioreg`), but `docker run … flash.sh` only sees the root
hub (`ls /dev/bus/usb/001/001` inside the container) and `flash.sh` errors
out with `ECID is empty / probing the target board failed`. UTM has reliable
macOS USB passthrough and a CLI (`utmctl`); we drive `flash.sh` natively
inside an Ubuntu 22.04 arm64 guest.

**One-time guest setup** (~10 min):
1. In UTM.app → New VM → Linux → Name `jetson-flash` → Architecture arm64.
2. Use the Ubuntu 22.04 arm64 cloud image from
   `https://cloud-images.ubuntu.com/jammy/current/` as the boot disk.
3. Memory 4096 MB, disk 30 GB, **USB controller enabled**.
4. Boot, install openssh-server and the flash deps once:
   ```bash
   sudo apt-get update
   sudo apt-get install -y openssh-server python3 liblz4-tool libxml2-utils qemu-user-static
   sudo systemctl enable --now ssh
   ```
5. Find the guest IP (`10.0.2.15` for UTM shared network is typical).
6. Add `~/.ssh/config` entry `Host jetson-flash`.

**Flash** (one-liner every time):
```bash
make flash-utm       # uses scripts/flash_jetson_utm.sh; expects Jetson in RCM
```

What the script does:
- `utmctl list` → fetch the VM uuid.
- `utmctl start` if the VM isn't already up.
- wait for ssh.
- re-check that the Jetson is visible on the host (`system_profiler … | grep "USB Single Serial"`).
- attach the USB device into the guest.
- `ssh jetson-flash 'cd <workspace>/Linux_for_Tegra && sudo ./flash.sh jetson-xavier-nx-devkit mmcblk1p1'`.
- detach the device and report.

After it completes, **power-cycle the Jetson (unplug + replug DC)** and it
will boot from the SD card.

### 5c. Validate
Run `scripts/validate.sh` on the Jetson via SSH. Key checks:
- Root device is `mmcblk1` (SD), not `mmcblk0` (eMMC)
- CUDA toolkit present (`nvcc --version`)
- SSH active
- GDM inactive (headless OK)

---

## 6. Relevant Files

| File | Purpose |
|------|---------|
| `scripts/strip_rootfs_no_apt.sh` | Headless rootfs strip (used) |
| `scripts/apply_binaries.sh` | Apply NVIDIA binaries into rootfs |
| `scripts/generate_sd_image_only.sh` | SD image generation (skips strip) |
| `scripts/validate.sh` | Post-boot validation |
| `Linux_for_Tegra/tools/jetson-disk-image-creator.sh` | Patched `write_partitions()` |
| `Linux_for_Tegra/jetson_xavier_nx_lean_r35.img` | Final image |