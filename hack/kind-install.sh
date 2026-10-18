#!/usr/bin/env bash
# Download one kind release and check it against the release's published sha256.
# Usage: kind-install.sh <version> <dest>
set -euo pipefail

version=$1
dest=$2

case "$(uname -s)" in
  Linux) os=linux ;;
  Darwin) os=darwin ;;
  MINGW* | MSYS*) os=windows ;;
  *) echo "unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac
case "$(uname -m)" in
  x86_64 | amd64) arch=amd64 ;;
  arm64 | aarch64) arch=arm64 ;;
  *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

asset="kind-${os}-${arch}"
url="https://github.com/kubernetes-sigs/kind/releases/download/${version}/${asset}"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

curl -fsSL -o "$tmp/$asset" "$url"
curl -fsSL -o "$tmp/$asset.sha256sum" "$url.sha256sum"
if command -v sha256sum >/dev/null; then
  (cd "$tmp" && sha256sum -c "$asset.sha256sum")
else
  (cd "$tmp" && shasum -a 256 -c "$asset.sha256sum")
fi

mkdir -p "$(dirname "$dest")"
install -m 0755 "$tmp/$asset" "$dest"
