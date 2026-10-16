#!/usr/bin/env bash
#
# check_protected.sh -- block diffs that touch the safety kernel.
#
# Usage:
#   scripts/check_protected.sh <base> <head>
#   scripts/check_protected.sh <base>..<head>
#
# Every file changed in <base>..<head> is matched against protected_paths.txt.
# If any changed file matches a protected glob the script exits 1, unless the
# environment variable KERNEL_CHANGE=1 is set to acknowledge an intentional,
# reviewed kernel change.
#
# Exit codes:
#   0  allowed (nothing protected changed, or KERNEL_CHANGE=1)
#   1  blocked (protected path changed without KERNEL_CHANGE=1)
#   2  usage error or git failure

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: check_protected.sh <base> <head>
   or: check_protected.sh <base>..<head>

Exits 1 if a file changed between <base> and <head> matches a glob in
protected_paths.txt, unless KERNEL_CHANGE=1 is set.
EOF
  exit 2
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
script_root="$(cd -- "$script_dir/.." && pwd)"

# Parse the revision range as either "<base> <head>" or "<base>..<head>".
if [ "$#" -eq 1 ]; then
  case "$1" in
    *...*) base="${1%%...*}"; head="${1##*...}" ;;
    *..*) base="${1%%..*}"; head="${1##*..}" ;;
    *) usage ;;
  esac
elif [ "$#" -eq 2 ]; then
  base="$1"
  head="$2"
else
  usage
fi

if [ -z "$base" ] || [ -z "$head" ]; then
  usage
fi

# Operate on the repository that contains the current working directory.
if ! repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  echo "check_protected: not inside a git repository" >&2
  exit 2
fi

# Locate the protection list: explicit override, then the target repository,
# then the repository that ships this script.
if [ -n "${KERNEL_PROTECTED_PATHS:-}" ]; then
  protected_file="$KERNEL_PROTECTED_PATHS"
elif [ -f "$repo_root/protected_paths.txt" ]; then
  protected_file="$repo_root/protected_paths.txt"
else
  protected_file="$script_root/protected_paths.txt"
fi

if [ ! -f "$protected_file" ]; then
  echo "check_protected: protection list not found: $protected_file" >&2
  exit 2
fi

is_protected() {
  local file="$1" pat
  while IFS= read -r pat || [ -n "$pat" ]; do
    pat="${pat%$'\r'}"
    if [ -z "$pat" ]; then
      continue
    fi
    if [[ "$pat" == '#'* ]]; then
      continue
    fi
    # A trailing slash protects every file beneath the directory.
    if [[ "$pat" == */ ]]; then
      pat="${pat}*"
    fi
    # An optional leading slash anchors the glob; strip it for matching.
    if [[ "$pat" == /* ]]; then
      pat="${pat#/}"
    fi
    if [[ "$file" == $pat ]]; then
      return 0
    fi
  done < "$protected_file"
  return 1
}

if ! changed_files="$(git diff --name-only "$base" "$head" 2>&1)"; then
  echo "check_protected: git diff ${base} ${head} failed:" >&2
  echo "$changed_files" >&2
  exit 2
fi

blocked=()
while IFS= read -r file; do
  if [ -z "$file" ]; then
    continue
  fi
  if is_protected "$file"; then
    blocked+=("$file")
  fi
done <<< "$changed_files"

if [ "${#blocked[@]}" -eq 0 ]; then
  echo "check_protected: ok — no protected paths changed in ${base}..${head}"
  exit 0
fi

if [ "${KERNEL_CHANGE:-0}" = "1" ]; then
  echo "check_protected: KERNEL_CHANGE=1 — allowing ${#blocked[@]} protected change(s) in ${base}..${head}:"
  printf '  %s\n' "${blocked[@]}"
  exit 0
fi

echo "check_protected: BLOCKED — protected safety-kernel path(s) changed in ${base}..${head}:" >&2
printf '  %s\n' "${blocked[@]}" >&2
echo "Set KERNEL_CHANGE=1 to allow this change after kernel review." >&2
exit 1