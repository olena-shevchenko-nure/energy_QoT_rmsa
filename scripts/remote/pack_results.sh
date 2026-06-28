#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/remote/pack_results.sh --run-dir runs/eon/eon_topn_baseline_smoke/<run_id>

Options:
  --run-dir PATH
  --include-checkpoints
  --help
USAGE
}

RUN_DIR=""
INCLUDE_CHECKPOINTS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --include-checkpoints) INCLUDE_CHECKPOINTS=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "--run-dir must point to an existing run directory." >&2
  exit 2
fi

RUN_ID="$(basename "$RUN_DIR")"
OUT="$(dirname "$RUN_DIR")/${RUN_ID}.tar.gz"
EXCLUDES=()
if [[ "$INCLUDE_CHECKPOINTS" -eq 0 ]]; then
  EXCLUDES+=(--exclude "*.pt" --exclude "*.pth" --exclude "*.ckpt")
fi

tar -czf "$OUT" "${EXCLUDES[@]}" -C "$(dirname "$RUN_DIR")" "$RUN_ID"
echo "$OUT"
