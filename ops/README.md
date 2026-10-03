# RLR production safety operations

This directory contains the independent thermal guard and the managed inference
service boundary. The RLR Governor must remain outside the inference service so
resource pressure in llama.cpp cannot starve the safety controller.

## 1. OPA and telemetry

Install the pinned/checksummed OPA binary:

```sh
sudo ./install-opa.sh
opa version
```

`rlr.telemetry.AutoTelemetryProvider` uses jtop when its service is available and
falls back to sysfs otherwise. Missing sensors remain `None`; they are never
fabricated as zero. On Xavier, verify/install the jtop daemon with the jetson-stats
tooling provided by the installed package.

## 2. cgroups v2 gate

Run `./check-cgroup-v2.sh` before installing the resource-constrained inference
unit. A nonzero exit is a **BLOCKED hardware prerequisite**, not a passing check.
Do not enable cgroups v2 by changing boot parameters as part of this script.

## 3. Managed inference service

Copy `inference.env.example` to `/etc/rlr/inference.env` and set the absolute
llama-server/model paths for the device. Then, on a cgroups-v2-capable system:

```sh
sudo ./install-runtime-services.sh
sudo systemctl start rlr-inference.service
sudo ./verify-runtime-services.sh
```

The unit applies `CPUQuota`, `MemoryHigh`, `MemoryMax`, `TasksMax`, `OOMPolicy`,
and restart controls. A successful start is not considered enough: validation
reads back the effective systemd resource properties.

## 4. Thermal guard

The guard samples the highest non-PMIC thermal zone every five seconds. Defaults:
80°C requests NVIDIA 10W mode, 85°C terminates heavy workloads after two
consecutive reads, and 90°C powers off after two consecutive reads.

At the kill threshold it first targets exact units from `MANAGED_UNITS` using
`systemctl kill --kill-who=all`. If none are active, `ALLOW_PKILL_FALLBACK=1`
keeps the legacy process-pattern termination as a conservative compatibility
fallback. Set the fallback to `0` once every heavy workload is managed by a known
unit/cgroup.

For a non-destructive guard check:

```sh
sudo systemctl stop thermal-guard.service
sudo DRYRUN=1 MANAGED_UNITS='rlr-inference.service' /usr/local/bin/thermal-guard.sh
# Ctrl-C after observing logs, then:
sudo systemctl start thermal-guard.service
```

Never report cgroup or jtop hardware validation as PASS from amd64 CI. Record
Xavier checks separately, including the exact cgroup mode, jtop service state,
NVPModel readback, thermal-guard DRYRUN evidence, and the controlled
validation → policy → executor → postcondition → journal action.
