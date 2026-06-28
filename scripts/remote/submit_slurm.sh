#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/remote/submit_slurm.sh --config configs/experiments/eon/remote_smoke.yaml --job-name eon_smoke

Options:
  --config PATH
  --job-name NAME
  --time HH:MM:SS        Default: 02:00:00
  --mem MEM              Default: 8G
  --gres VALUE           Optional, e.g. gpu:1
  --cpus-per-task N      Default: 4
  --help
USAGE
}

CONFIG=""
JOB_NAME="eon_job"
TIME="02:00:00"
MEM="8G"
GRES=""
CPUS="4"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --gres) GRES="$2"; shift 2 ;;
    --cpus-per-task) CPUS="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  usage >&2
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is not available on this machine." >&2
  exit 1
fi

mkdir -p runs/slurm_logs
SBATCH_ARGS=(--job-name "$JOB_NAME" --time "$TIME" --mem "$MEM" --cpus-per-task "$CPUS" --output "runs/slurm_logs/${JOB_NAME}_%j.out" --error "runs/slurm_logs/${JOB_NAME}_%j.err")
if [[ -n "$GRES" ]]; then
  SBATCH_ARGS+=(--gres "$GRES")
fi

sbatch "${SBATCH_ARGS[@]}" --wrap "if [[ -z \"\${VIRTUAL_ENV:-}\" && -f .venv-cse2026/bin/activate ]]; then source .venv-cse2026/bin/activate; fi; python scripts/experiments/run_eon_experiment.py --config '$CONFIG'"
