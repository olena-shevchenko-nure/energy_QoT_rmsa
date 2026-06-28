# Remote Experiment Setup

This guide sets up a remote Linux machine for CSE 2026 EON experiments without
hardcoded usernames, hosts, keys, passwords, or machine-specific paths.

## 1. Clone And Checkout

```bash
git clone https://github.com/shevchenko-oleksandr/CSE_2026.git
cd CSE_2026
git checkout infra/remote-experiment-env-v1
```

## 2. Configure Private Environment

```bash
cp configs/remote/remote.env.example .env
```

Edit `.env` for the remote machine:

```bash
export CSE2026_PROJECT_ROOT=/path/to/CSE_2026
export CSE2026_DATA_ROOT=/path/to/data
export CSE2026_RUNS_ROOT=/path/to/runs
export CSE2026_DEVICE=auto
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TORCH_INSTALL_CMD=
```

Then load it:

```bash
set -a
source .env
set +a
```

## 3. Bootstrap Python

CPU-safe default:

```bash
bash scripts/remote/bootstrap_remote.sh --venv .venv-cse2026 --device cpu
source .venv-cse2026/bin/activate
```

For GPU machines, set a site-approved PyTorch install command before bootstrap:

```bash
export TORCH_INSTALL_CMD="python -m pip install torch"
bash scripts/remote/bootstrap_remote.sh --venv .venv-cse2026 --device cuda
```

The script installs `requirements-data.txt` and `requirements-experiments.txt`,
prints Python/pip/machine info, checks core imports, and reports Torch/CUDA if
Torch is installed.

## 4. Check Remote Environment

```bash
python scripts/remote/check_remote_env.py --dataset data/eon/generated/nsfnet_smoke
```

To also run dataset validation:

```bash
python scripts/remote/check_remote_env.py \
  --dataset data/eon/generated/nsfnet_smoke \
  --run-validation
```

Reports are written under:

```text
runs/env_checks/<timestamp>/env_report.json
```

## 5. Validate Smoke Dataset

```bash
python scripts/experiments/run_eon_experiment.py \
  --config configs/experiments/eon/remote_smoke.yaml
```

Each run creates:

```text
runs/eon/<experiment_name>/<YYYYMMDD_HHMMSS>_<short_git_sha>/
```

with resolved configs, git info, env report, logs, metrics, and reports.

## 6. Generate MVP Dataset

```bash
python scripts/data/generate_eon_data.py \
  --config configs/data/nsfnet_mvp.yaml \
  --output "$CSE2026_DATA_ROOT/eon/generated/nsfnet_mvp"
```

Validate it:

```bash
python scripts/experiments/run_eon_experiment.py \
  --config configs/experiments/eon/remote_mvp_validate.yaml
```

## 7. Launch With tmux

Dry run:

```bash
bash scripts/remote/launch_tmux.sh \
  --name eon_smoke \
  --config configs/experiments/eon/remote_smoke.yaml \
  --dry-run
```

Launch:

```bash
bash scripts/remote/launch_tmux.sh \
  --name eon_smoke \
  --config configs/experiments/eon/remote_smoke.yaml
```

Monitor:

```bash
tmux attach -t eon_smoke
tail -f runs/launch_logs/eon_smoke_*.log
```

## 8. Launch With nohup

```bash
bash scripts/remote/launch_nohup.sh \
  --name eon_smoke \
  --config configs/experiments/eon/remote_smoke.yaml
```

Monitor:

```bash
cat runs/pids/eon_smoke.pid
tail -f runs/launch_logs/eon_smoke_*.log
```

## 9. Optional SLURM

```bash
bash scripts/remote/submit_slurm.sh \
  --config configs/experiments/eon/remote_smoke.yaml \
  --job-name eon_smoke \
  --time 02:00:00 \
  --mem 8G \
  --cpus-per-task 4
```

For GPU jobs, pass a cluster-appropriate `--gres`, for example:

```bash
bash scripts/remote/submit_slurm.sh \
  --config configs/experiments/eon/remote_dataloader_smoke.yaml \
  --job-name eon_dataloader \
  --gres gpu:1
```

No partition names are hardcoded.

## 10. Monitor Logs

Run logs are inside each run directory:

```bash
find runs/eon -name run.log -o -name stdout.log
tail -f runs/eon/<experiment>/<run_id>/run.log
tail -f runs/eon/<experiment>/<run_id>/stdout.log
```

Launcher logs are under:

```text
runs/launch_logs/
runs/slurm_logs/
```

## 11. Pack And Download Results

Pack a run:

```bash
bash scripts/remote/pack_results.sh \
  --run-dir runs/eon/eon_topn_baseline_smoke/<run_id>
```

Include checkpoints only when needed:

```bash
bash scripts/remote/pack_results.sh \
  --run-dir runs/eon/eon_topn_baseline_smoke/<run_id> \
  --include-checkpoints
```

Sync from remote to local:

```bash
bash scripts/remote/rsync_results.sh \
  user@host:/remote/path/to/runs/eon \
  ./runs_remote/eon
```

Replace `user@host` and paths manually; do not commit them.

## 12. Resume

Resume the latest run for a config:

```bash
python scripts/experiments/run_eon_experiment.py \
  --config configs/experiments/eon/remote_topn_baseline.yaml \
  --resume latest
```

Resume a specific run:

```bash
python scripts/experiments/run_eon_experiment.py \
  --config configs/experiments/eon/remote_topn_baseline.yaml \
  --resume runs/eon/eon_topn_baseline_smoke/<run_id>
```

New runs never overwrite an existing run directory. Resume intentionally writes
into the selected existing run directory.

## 13. Troubleshooting

`pyarrow` missing:

```bash
python -m pip install -r requirements-data.txt
```

`tmux` missing:

```bash
bash scripts/remote/launch_nohup.sh --name eon_smoke --config configs/experiments/eon/remote_smoke.yaml
```

No CUDA:

```bash
python scripts/remote/check_remote_env.py --dataset data/eon/generated/nsfnet_smoke
```

Check `torch.cuda_available` and `nvidia_smi` in `env_report.json`. Use CPU
stages until the cluster CUDA/PyTorch setup is fixed.

Insufficient disk:

```bash
python scripts/remote/check_remote_env.py --dataset data/eon/generated/nsfnet_smoke
```

Review `disk` in the report and move `CSE2026_DATA_ROOT` or
`CSE2026_RUNS_ROOT` to a larger filesystem.

Dataset path not found:

```bash
ls "$CSE2026_DATA_ROOT/eon/generated"
python scripts/experiments/run_eon_experiment.py --config configs/experiments/eon/remote_smoke.yaml --dry-run
```

Dirty git tree:

```bash
git status --short
```

The env report records branch, commit, and dirty status for reproducibility.

## 14. Security Notes

- Do not commit `.env` or `remote.env`.
- Do not commit SSH keys or passwords.
- Do not hardcode remote usernames or hostnames.
- Do not commit large run artifacts, checkpoints, or generated MVP/full datasets.
- Keep `data/eon/generated/nsfnet_smoke` committed and valid.
