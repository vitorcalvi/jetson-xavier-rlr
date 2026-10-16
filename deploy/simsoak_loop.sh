#!/bin/bash
# Overnight simsoak loop for RLR Phase 1 step D2.
#
# rlr-simsoak drives a *simulated* fault-injected soak (a private clock, no
# wall-clock sleeping), so a single "--hours 6" pass finishes in minutes. To soak
# overnight this service re-runs the 6-hour soak until the RLR kill file
# (~/rlr/KILL) appears, writing each pass's TOON result under ~/rlr/soak/ and
# keeping simsoak.latest.toon pointed at the newest one. It never prints or
# persists secrets and never needs sudo.
set -u

VENV="$HOME/rlr/venv/bin"
SOAK="$HOME/rlr/soak"
KILL="$HOME/rlr/KILL"
HOURS="${RLR_SIMSOAK_HOURS:-6}"
SEED="${RLR_SIMSOAK_SEED:-17}"
REST="${RLR_SIMSOAK_REST_SECONDS:-120}"

mkdir -p "$SOAK"
while [ ! -e "$KILL" ]; do
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  out="$SOAK/simsoak-$ts.toon"
  "$VENV/rlr-simsoak" --hours "$HOURS" --seed "$SEED" > "$out" 2>>"$SOAK/simsoak.err"
  rc=$?
  cp "$out" "$SOAK/simsoak.latest.toon"
  printf '%s pass exit=%s hours=%s seed=%s\n' "$ts" "$rc" "$HOURS" "$SEED" \
    >> "$SOAK/simsoak.passes.log"
  sleep "$REST"
done
printf '%s simsoak loop stopped: kill file present\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >> "$SOAK/simsoak.passes.log"