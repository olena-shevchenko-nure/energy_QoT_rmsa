#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/remote/bootstrap_remote.sh --venv .venv-cse2026 --device cpu
  bash scripts/remote/bootstrap_remote.sh --venv .venv-cse2026 --device cuda

Options:
  --venv PATH     Virtual environment path. Default: .venv-cse2026
  --device MODE   cpu, cuda, or auto. Default: cpu
  --help          Show this help.
USAGE
}

VENV_PATH=".venv-cse2026"
DEVICE="cpu"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
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

if [[ ! -f "requirements-data.txt" ]]; then
  echo "Run this script from the repository root." >&2
  exit 1
fi

python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-data.txt
python -m pip install -r requirements-experiments.txt

if [[ -n "${TORCH_INSTALL_CMD:-}" ]]; then
  echo "Running TORCH_INSTALL_CMD for device=${DEVICE}"
  bash -lc "$TORCH_INSTALL_CMD"
else
  echo "TORCH_INSTALL_CMD is not set; skipping optional PyTorch install."
fi

echo "Python:"
python --version
echo "Pip:"
python -m pip --version
echo "Machine:"
uname -a

python - <<'PY'
import importlib
import platform

print("platform:", platform.platform())
for name in ["numpy", "pandas", "pyarrow", "networkx", "yaml", "tqdm", "pytest"]:
    module = importlib.import_module(name)
    print(f"import {name}: ok {getattr(module, '__version__', '')}")
try:
    import torch
except ImportError:
    print("torch: not installed")
else:
    print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
PY

echo "Bootstrap complete. Activate with: source $VENV_PATH/bin/activate"
