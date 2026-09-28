# Thermal guard

Install on the Jetson as root (copy files from this directory):

```sh
install -m755 thermal-guard.sh /usr/local/bin/thermal-guard.sh
install -m644 thermal-guard.service /etc/systemd/system/thermal-guard.service
systemctl daemon-reload
systemctl enable --now thermal-guard.service
```

The service reads the highest non-PMIC thermal-zone temperature every 5 seconds. Defaults: at 80°C it switches to NVIDIA 10W mode (mode 5); at 85°C it sends TERM to matching build/inference workloads after two consecutive readings; at 90°C it powers off after two consecutive readings. Values are configurable with `WARN`, `KILL`, `SHUTDOWN`, and `INTERVAL` in the script.

For a non-destructive dry run, stop the service and run it in a shell with `DRYRUN=1`; it logs the actions it would take without changing power mode, terminating workloads, or shutting down:

```sh
systemctl stop thermal-guard.service
DRYRUN=1 /usr/local/bin/thermal-guard.sh
```

Stop it with Ctrl-C, then restart the service with `systemctl start thermal-guard.service`.
