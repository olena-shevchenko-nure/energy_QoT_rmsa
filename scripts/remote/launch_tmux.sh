#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/remote/launch_tmux.sh --name eon_smoke --config configs/experiments/eon/remote_smoke.yaml

Options:
  --name NAME       tmux session name.
  --config PATH     Experiment config.
  --dry-run         Print command without launching tmux.
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

mkdir -p runs/launch_logs
LOG="runs/launch_logs/${NAME}_$(date +%Y%m%d_%H%M%S).log"
ACTIVATE='if [[ -z "${VIRTUAL_ENV:-}" && -f ".venv-cse2026/bin/activate" ]]; then source ".venv-cse2026/bin/activate"; fi'
CMD="${ACTIVATE}; python scripts/experiments/run_eon_experiment.py --config \"${CONFIG}\" 2>&1 | tee -a \"${LOG}\""

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "tmux session: $NAME"
  echo "command: bash -lc '$CMD'"
  echo "launcher log: $LOG"
  exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed; use scripts/remote/launch_nohup.sh instead." >&2
  exit 1
fi

tmux new-session -d -s "$NAME" "bash -lc '$CMD'"
echo "Launched tmux session: $NAME"
echo "Attach: tmux attach -t $NAME"
echo "Tail: tail -f $LOG"
