#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/remote/launch_nohup.sh --name eon_smoke --config configs/experiments/eon/remote_smoke.yaml

Options:
  --name NAME       Job name for logs and pid file.
  --config PATH     Experiment config.
  --dry-run         Print command without launching.
  --help            Show this help.
USAGE
}

NAME=""
CONFIG=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      NAME="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$NAME" || -z "$CONFIG" ]]; then
  usage >&2
  exit 2
fi

mkdir -p runs/launch_logs runs/pids
LOG="runs/launch_logs/${NAME}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="runs/pids/${NAME}.pid"
ACTIVATE='if [[ -z "${VIRTUAL_ENV:-}" && -f ".venv-cse2026/bin/activate" ]]; then source ".venv-cse2026/bin/activate"; fi'
PYTHON_SELECT='if command -v python >/dev/null 2>&1; then PYTHON_BIN=python; elif command -v python3 >/dev/null 2>&1; then PYTHON_BIN=python3; else echo "python not found" >&2; exit 127; fi'
CMD="${ACTIVATE}; ${PYTHON_SELECT}; \${PYTHON_BIN} scripts/experiments/run_eon_experiment.py --config \"${CONFIG}\""

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "command: nohup bash -lc '$CMD' > '$LOG' 2>&1 &"
  echo "pid file: $PID_FILE"
  exit 0
fi

nohup bash -lc "$CMD" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"
echo "Launched nohup job: $NAME pid=$PID"
echo "Tail: tail -f $LOG"
echo "PID file: $PID_FILE"
