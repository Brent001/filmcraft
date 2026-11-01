#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

: "${VERSION:?VERSION must be set by the release workflow}"

case "$(uname -m)" in
  x86_64|amd64) arch="x86_64" ;;
  aarch64|arm64) arch="aarch64" ;;
  *)
    printf 'Unsupported server release architecture: %s\n' "$(uname -m)" >&2
    exit 1
    ;;
esac

cargo build --locked --release -p filmcraft-cli

binary="target/release/filmcraft-cli"
if [[ ! -x "$binary" ]]; then
  printf 'Expected CLI binary was not built: %s\n' "$binary" >&2
  exit 1
fi

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
package="$stage/filmcraft-server"
mkdir -p "$package"

install -m 0755 "$binary" "$package/filmcraft-cli"

for file in README.md LICENSE-MIT LICENSE-APACHE NOTICE ATTRIBUTION.md; do
  if [[ -f "$file" ]]; then
    install -m 0644 "$file" "$package/$file"
  fi
done

mkdir -p dist/release
archive="dist/release/filmcraft-server-${VERSION}-linux-${arch}.tar.gz"
tar -C "$stage" -czf "$archive" filmcraft-server
tar -tzf "$archive" >/dev/null
printf 'Created %s\n' "$archive"
