#!/bin/sh
set -eu

ROOT=${CGROUP_ROOT:-/sys/fs/cgroup}
if [ ! -r "$ROOT/cgroup.controllers" ]; then
  echo "BLOCKED: unified cgroups v2 is not mounted at $ROOT" >&2
  exit 2
fi

controllers=$(cat "$ROOT/cgroup.controllers")
missing=""
for name in cpu memory pids; do
  case " $controllers " in *" $name "*) ;; *) missing="$missing $name";; esac
done
if [ -n "$missing" ]; then
  echo "BLOCKED: cgroups v2 missing required controllers:$missing" >&2
  exit 3
fi

echo "cgroups_v2=ready controllers=$controllers"
