#!/bin/sh
set -eu

VERSION=1.21.1
DEST=${OPA_DEST:-/usr/local/bin/opa}
arch=$(uname -m)
case "$arch" in
  x86_64)
    asset=opa_linux_amd64_static
    digest=668506eb17a2eaa1fce6cc0d1f42ef85125d4ac5bda5fc74d1152d0c77145031
    ;;
  aarch64|arm64)
    asset=opa_linux_arm64_static
    digest=9a1f3625529c6f01240fe68286dde06aa0b23c5253da700ac48f4d943ff8a4de
    ;;
  *) echo "unsupported architecture: $arch" >&2; exit 65;;
esac

url="https://github.com/open-policy-agent/opa/releases/download/v${VERSION}/${asset}"
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
if command -v curl >/dev/null 2>&1; then
  curl -fsSL --retry 3 -o "$tmp" "$url"
elif command -v wget >/dev/null 2>&1; then
  wget -q --tries=3 -O "$tmp" "$url"
else
  echo "curl or wget is required to download OPA" >&2
  exit 69
fi
printf '%s  %s\n' "$digest" "$tmp" | sha256sum -c -
install -m755 "$tmp" "$DEST"
"$DEST" version
