.PHONY: help build shell download apply strip flash flash-utm validate clean

IMAGE_NAME := jetson-build:r35.6.1
UTM_VM_NAME ?= jetson-flash
UTM_SSH_TARGET ?= jetson-flash
CONTAINER_RUN := docker run --rm -it --privileged --platform linux/amd64 \
  -v "$(pwd)":/workspace \
  -e BSP_URL \
  -e ROOTFS_URL \
  -e L4T_VERSION \
  -e BOARD \
  -e IMAGE_NAME \
  -e USERNAME \
  -e PASSWORD \
  -e SD_DEVICE \
  $(IMAGE_NAME)

help:
	@echo "Jetson Xavier NX — Lean SD Image Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make build      Build the Docker image (x86_64 Ubuntu 20.04 via QEMU)"
	@echo "  make shell      Open interactive shell in the build container"
	@echo "  make download   Download BSP and rootfs (requires BSP_URL, ROOTFS_URL)"
	@echo "  make apply      Apply NVIDIA binaries to rootfs"
	@echo "  make strip      Strip rootfs (headless, keep CUDA, create user)"
	@echo "  make flash      Generate SD image (and optionally flash if SD_DEVICE is set)"
	@echo "  make flash-utm  Flash Jetson in recovery via UTM-managed Ubuntu VM (macOS)"
	@echo "  make validate   Run validation (copy validate.sh to Jetson and run there)"
	@echo "  make clean      Remove build artifacts"
	@echo ""
	@echo "Required environment variables:"
	@echo "  BSP_URL        URL to NVIDIA Driver Package (BSP) tarball"
	@echo "  ROOTFS_URL     URL to NVIDIA Sample Root Filesystem tarball"
	@echo "  SD_DEVICE      (optional) Block device for SD card, e.g. /dev/sdX"
	@echo ""
	@echo "Example:"
	@echo "  make build"
	@echo "  BSP_URL=... ROOTFS_URL=... make download"
	@echo "  make apply"
	@echo "  USERNAME=jetson PASSWORD=<DEVICE_PASSWORD> make strip"
	@echo "  BOARD=jetson-xavier-nx-devkit make flash"

build:
	docker build --platform linux/amd64 -t $(IMAGE_NAME) .

shell:
	$(CONTAINER_RUN) bash

download:
	$(CONTAINER_RUN) bash /workspace/scripts/download_bsp.sh

apply:
	$(CONTAINER_RUN) bash /workspace/scripts/apply_binaries.sh

strip:
	$(CONTAINER_RUN) bash /workspace/scripts/strip_rootfs.sh

flash:
	$(CONTAINER_RUN) bash /workspace/scripts/generate_sd_image.sh

# Flash to a Jetson in recovery mode via a UTM-managed Ubuntu VM on macOS.
# Why: Docker Desktop on macOS does not pass arbitrary USB devices into the
# container, so the in-container `flash.sh` always fails with "ECID is empty".
# UTM has reliable USB passthrough; we drive it through utmctl.
# Pre-req: see the top of scripts/flash_jetson_utm.sh for the one-time
# UTM VM creation (Ubuntu 22.04 arm64, USB enabled) and ssh setup.
flash-utm:
	@if [ ! -t 0 ]; then echo "NOTE: re-run with an interactive terminal; this target needs utmctl/listen." ; exit 1 ; fi
	JETSON_UTM_NAME="$(UTM_VM_NAME)" JETSON_UTM_SSH="$(UTM_SSH_TARGET)" \
	  bash scripts/flash_jetson_utm.sh "$(PWD)"

validate:
	@echo "Copy validate.sh to the Jetson and run: bash validate.sh"
	@echo "Or: scp scripts/validate.sh jetson@<ip>:/home/jetson/"

clean:
	rm -rf Linux_for_Tegra/*.img Linux_for_Tegra/rootfs Linux_for_Tegra/*.tar.bz2
	rm -rf Linux_for_Tegra_extract
	find . -name "*.img" -delete
