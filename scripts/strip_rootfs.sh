#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/workspace"
L4T_DIR="${WORKSPACE}/Linux_for_Tegra"
ROOTFS_DIR="${L4T_DIR}/rootfs"
USERNAME="${USERNAME:-jetson}"
PASSWORD="${PASSWORD:?Set PASSWORD to a strong device password}"
OVERLAY_DIR="${WORKSPACE}/overlay"

echo "=== Step 3: Build lean headless rootfs ==="
echo "Rootfs dir  : ${ROOTFS_DIR}"
echo "Default user: ${USERNAME}"

if [[ ! -d "${ROOTFS_DIR}" ]]; then
  echo "ERROR: rootfs directory not found at ${ROOTFS_DIR}"
  exit 1
fi

# Mount required filesystems for chroot
sudo mount -t proc  proc  "${ROOTFS_DIR}/proc"  2>/dev/null || sudo mount -t proc proc "${ROOTFS_DIR}/proc"
sudo mount -t sysfs sysfs "${ROOTFS_DIR}/sys"   2>/dev/null || sudo mount -t sysfs sysfs "${ROOTFS_DIR}/sys"
sudo mount -o bind /dev "${ROOTFS_DIR}/dev"
sudo mount -o bind /dev/pts "${ROOTFS_DIR}/dev/pts"

cleanup() {
  echo "Cleaning up mounts..."
  sudo umount "${ROOTFS_DIR}/dev/pts" 2>/dev/null || true
  sudo umount "${ROOTFS_DIR}/dev"     2>/dev/null || true
  sudo umount "${ROOTFS_DIR}/sys"     2>/dev/null || true
  sudo umount "${ROOTFS_DIR}/proc"    2>/dev/null || true
}
trap cleanup EXIT

# 3.2 Remove desktop/GUI packages
echo "Removing desktop/GUI packages..."
sudo chroot "${ROOTFS_DIR}" /bin/bash -lc '
  set -e
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get remove -y --purge \
    ubuntu-desktop \
    gnome-shell \
    gdm3 \
    lightdm \
    plymouth \
    xserver-xorg* \
    nvidia-l4t-graphics \
    "^libnvidia-egl*" \
    "^libnvidia-gl*" \
    "^nvidia-driver*" \
    2>/dev/null || true
  apt-get autoremove -y || true
  apt-get clean || true
'

# 3.3 Remove sample/demo CUDA apps (keep CUDA runtime)
echo "Removing CUDA sample apps..."
sudo chroot "${ROOTFS_DIR}" /bin/bash -lc '
  set -e
  export DEBIAN_FRONTEND=noninteractive
  apt-get remove -y --purge \
    cuda-samples-11-4 \
    cuda-visual-tools-11-4 \
    cuda-nvcc-11-4 \
    2>/dev/null || true
  apt-get autoremove -y || true
'

# 3.4 Purge docs, locales, caches
echo "Purging docs, locales, and caches..."
sudo chroot "${ROOTFS_DIR}" /bin/bash -lc '
  set -e
  rm -rf /usr/share/doc/* /usr/share/man/* /usr/share/info/*
  find /usr/share/locale -mindepth 1 -maxdepth 1 ! -name "en*" -exec rm -rf {} +
  rm -rf /var/lib/apt/lists/*
  rm -rf /var/cache/apt/archives/*.deb
'

# 3.5 Install SSH server and utilities
echo "Installing SSH server and utilities..."
sudo chroot "${ROOTFS_DIR}" /bin/bash -lc '
  set -e
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y \
    openssh-server \
    sudo \
    curl \
    wget \
    python3 \
    python3-pip \
    python3-venv \
    net-tools \
    iproute2 \
    htop \
    nano \
    jq
  pip3 install --no-cache-dir jetson-stats
'

# 3.6 Create default user non-interactively
echo "Creating default user: ${USERNAME}"
if [[ -x "${L4T_DIR}/tools/l4t_create_default_user.sh" ]]; then
  sudo "${L4T_DIR}/tools/l4t_create_default_user.sh" \
    --autologin \
    --name "${USERNAME}" \
    --password "${PASSWORD}" \
    --accept-license \
    "${ROOTFS_DIR}"
else
  echo "l4t_create_default_user.sh not found; creating user manually..."
  sudo chroot "${ROOTFS_DIR}" /bin/bash -lc "
    useradd -m -s /bin/bash -G sudo ${USERNAME} && \
    echo '${USERNAME}:${PASSWORD}' | chpasswd && \
    echo '${USERNAME} ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/${USERNAME}
  "
fi

# 3.7 Enable SSH and harden config
echo "Configuring SSH..."
sudo chroot "${ROOTFS_DIR}" /bin/bash -lc '
  systemctl enable ssh
  mkdir -p /etc/ssh/sshd_config.d
  cat > /etc/ssh/sshd_config.d/99-hardening.conf << "SSHEOF"
PermitRootLogin no
PasswordAuthentication yes
ChallengeResponseAuthentication no
UsePAM yes
SSHEOF
'

# 3.8 Disable unnecessary services
echo "Disabling unnecessary services..."
sudo chroot "${ROOTFS_DIR}" /bin/bash -lc '
  systemctl disable \
    ModemManager \
    NetworkManager-wait-online \
    systemd-udevd \
    gdm \
    lightdm \
    2>/dev/null || true
'

# 3.9 jetson_clocks service for consistent performance
echo "Creating jetson_clocks service..."
sudo chroot "${ROOTFS_DIR}" /bin/bash -lc '
  cat > /etc/systemd/system/jetson-clocks.service << "EOF"
[Unit]
Description=Jetson Clocks Max Performance
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks --store
StandardOutput=journal

[Install]
WantedBy=multi-user.target
EOF
  systemctl enable jetson-clocks.service
'

# 3.10 Apply overlay files if any
echo "Applying overlay files..."
if [[ -d "${OVERLAY_DIR}" ]] && [[ -n "$(ls -A "${OVERLAY_DIR}" 2>/dev/null)" ]]; then
  sudo rsync -av "${OVERLAY_DIR}/" "${ROOTFS_DIR}/"
fi

# 3.11 Record image size
echo "=== Rootfs size before flashing ==="
du -sh "${ROOTFS_DIR}"

echo "=== Lean rootfs build complete ==="
echo "Next: bash /workspace/scripts/generate_sd_image.sh"
