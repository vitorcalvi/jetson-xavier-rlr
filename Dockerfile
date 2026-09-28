# Docker-based lean SD image build for Jetson Xavier NX
# Target: Ubuntu 20.04 x86_64 via QEMU emulation on Apple Silicon macOS
# Build with: docker build --platform linux/amd64 -t jetson-build:r35.6.1 .
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV L4T_VERSION=R35.6.1
ENV BOARD=jetson-xavier-nx-devkit

# Install host build dependencies and QEMU for arm64 chroot support
RUN apt-get update && apt-get install -y \
    qemu-user-static \
    binfmt-support \
    build-essential \
    python3 \
    python3-pip \
    libxml2-utils \
    bzip2 \
    sshpass \
    abootimg \
    pv \
    curl \
    wget \
    rsync \
    kmod \
    dosfstools \
    git \
    sudo \
    gdisk \
    parted \
    xz-utils \
    lz4 \
    cpio \
    python3-yaml \
    systemd-container \
  && rm -rf /var/lib/apt/lists/*

# Enable QEMU static binaries for arm64/aarch64 so chroot works inside x86_64 container
RUN mkdir -p /usr/bin && \
    cp /usr/bin/qemu-aarch64-static /usr/bin/ 2>/dev/null || true && \
    cp /usr/bin/qemu-arm-static /usr/bin/ 2>/dev/null || true

# Workspace
RUN mkdir -p /workspace/overlay
WORKDIR /workspace

# Default: open bash
CMD ["/bin/bash"]
