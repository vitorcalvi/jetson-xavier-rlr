#!/bin/sh
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
[ "$(id -u)" -eq 0 ] || { echo "run as root" >&2; exit 77; }

systemd-analyze verify "$HERE/thermal-guard.service" "$HERE/rlr-inference.service"
install -m755 "$HERE/thermal-guard.sh" /usr/local/bin/thermal-guard.sh
install -m644 "$HERE/thermal-guard.service" /etc/systemd/system/thermal-guard.service
systemctl daemon-reload
systemctl enable --now thermal-guard.service

if ! "$HERE/check-cgroup-v2.sh"; then
  echo "BLOCKED: inference cgroup-v2 service not installed; thermal guard updated" >&2
  exit 3
fi

install -d -m755 /etc/rlr
install -m755 "$HERE/rlr-inference-launch.sh" /usr/local/bin/rlr-inference-launch.sh
install -m644 "$HERE/rlr-inference.service" /etc/systemd/system/rlr-inference.service
systemctl daemon-reload
systemctl enable rlr-inference.service

echo "installed; configure /etc/rlr/inference.env before starting rlr-inference.service"
