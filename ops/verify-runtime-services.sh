#!/bin/sh
set -eu

systemd-analyze verify /etc/systemd/system/thermal-guard.service
systemctl is-active --quiet thermal-guard.service

if [ -r /sys/fs/cgroup/cgroup.controllers ]; then
  systemd-analyze verify /etc/systemd/system/rlr-inference.service
  systemctl show rlr-inference.service \
    -p CPUQuotaPerSecUSec -p MemoryHigh -p MemoryMax -p TasksMax -p OOMPolicy
else
  echo "BLOCKED: cgroups v2 unavailable; inference resource verification skipped" >&2
  exit 3
fi
