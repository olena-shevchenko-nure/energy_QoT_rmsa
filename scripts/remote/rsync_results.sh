#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/remote/rsync_results.sh user@host:/remote/path/to/runs/eon ./runs_remote/eon
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

SRC="$1"
DST="$2"
rsync -avz --progress "$SRC" "$DST"
