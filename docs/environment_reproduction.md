# Pinned Environment

This repository includes a pinned reproduction environment for the MVP80 experiments.

## Files

- `requirements-repro.txt`: installable pinned Python dependencies for fresh reproduction.
- `requirements-repro-lock.txt`: full observed `pip freeze` snapshot from the experiment host.
- `environment-repro.yml`: conda/mamba environment wrapper using `requirements-repro.txt`.
- `third_party/optical-networking-gym.lock`: pinned external Optical Networking Gym commit used by automatic setup.

## Captured Experiment Host

The pinned package versions were captured from the server used for the final experiments:

```text
OS: Ubuntu 24.04.4 LTS
Kernel: Linux 6.8.0-107-generic x86_64 GNU/Linux
CPU: 2 x AMD EPYC 9354 32-Core Processor, 64 physical cores / 128 threads
Python: 3.12.3
pip: 26.1.1
GPU: NVIDIA L40
NVIDIA driver: 580.126.09
```

## Recommended Fresh Setup

```bash
python3.12 -m venv .venv-repro
source .venv-repro/bin/activate
python -m pip install --upgrade pip==26.1.1
python -m pip install -r requirements-repro.txt
python -m pip install -e .
```

The MVP80 wrapper installs the pinned Optical Networking Gym source checkout automatically on first run:

```bash
python scripts/reproduce_mvp80.py --dry-run
python scripts/reproduce_mvp80.py
```

Manual setup is still possible when network access is restricted or a shared checkout is preferred:

```bash
git clone https://github.com/carlosnatalino/optical-networking-gym.git external/optical-networking-gym
git -C external/optical-networking-gym checkout 622d0741ff75388161f7c468757ae880471d6d2b
```

If your local path differs from `external/optical-networking-gym`, pass it to the reproduction wrapper; the wrapper can clone the pinned commit to that path if it is missing:

```bash
python scripts/reproduce_mvp80.py --ong-source-path /path/to/optical-networking-gym
```

## Lock File Caveat

`requirements-repro-lock.txt` is the full observed host snapshot. It includes some Ubuntu/system packages reported by `pip freeze`; those entries may not be portable to a clean pip-only virtual environment. Use `requirements-repro.txt` for fresh installation and keep `requirements-repro-lock.txt` as provenance.

The pinned environment is intended to reproduce the paper evaluation/runtime pipeline. Re-training some GPU-specific branches, especially LightGBM GPU training, may require a GPU-enabled LightGBM build and matching local CUDA/OpenCL drivers.
