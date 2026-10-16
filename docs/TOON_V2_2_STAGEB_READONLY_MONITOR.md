# TOON v2.2 — 48h Stage 2 read-only health monitor

This monitor does not transmit or cancel broker orders and does not change the Stage 2 systemd service or the `HALT_ENTRIES` file. It reads the existing halted controller, independent safety watcher, runtime heartbeat, service PID, `/proc/<pid>/status`, `/proc/<pid>/fd`, and sysfs thermal zones. It terminates on missing/stale watcher data, unexpected broker exposure, missing halt, a Stage 2 PID change, or a sampled temperature at/above 80°C.

## Run on a Jetson with the operator's explicit safety hold

```sh
python scripts/watch_stage2_readonly.py --once --out /home/vitor/rlr/monitoring/stage2_health.toon
systemd-run --user --unit=toon22-stage2-health-watch --property=Restart=no --property=RuntimeMaxSec=176400 python scripts/watch_stage2_readonly.py --hours 48 --sample-seconds 300 --out /home/vitor/rlr/monitoring/stage2_health.toon
systemctl --user show toon22-stage2-health-watch.service -p ActiveState -p MainPID -p ExecMainStatus
systemctl --user status toon22-stage2-health-watch.service
systemctl --user stop toon22-stage2-health-watch.service  # stops monitor only
```

Logs are append-only, mode 0600. A successful start and individual samples do **not** prove 48-hour uptime or zero leaks. The final sample must be compared against the first; inspect sample counts, RSS, FD count, watcher lag, max temperature, and PID continuity. This is not a live market-feed shadow run, L2 directional IC validation, genuine 30-trade Alpaca certification, or a profitability proof. Do not remove or bypass the global halt to run this monitor.
