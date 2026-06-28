#!/usr/bin/env bash
set -euo pipefail

echo START_DAGGER_XGBOOST_ENERGY_KSP
python3 scripts/experiments/run_eon_experiment.py \
  --config configs/experiments/eon/remote_train_dagger_tree_ranker_xgboost_energy_ksp_mvp16.yaml
XGB_RUN=$(ls -td runs/eon/eon_train_dagger_tree_ranker_xgboost_energy_ksp_mvp16/* | head -1)
export CSE2026_DAGGER_XGBOOST_RANKER_PATH="$XGB_RUN/tree_ranker.json"
echo "XGB_RUN=$XGB_RUN"
echo "CSE2026_DAGGER_XGBOOST_RANKER_PATH=$CSE2026_DAGGER_XGBOOST_RANKER_PATH"

echo START_DAGGER_LIGHTGBM_ENERGY_KSP
python3 scripts/experiments/run_eon_experiment.py \
  --config configs/experiments/eon/remote_train_dagger_tree_ranker_lightgbm_energy_ksp_mvp16.yaml
LGB_RUN=$(ls -td runs/eon/eon_train_dagger_tree_ranker_lightgbm_energy_ksp_mvp16/* | head -1)
export CSE2026_DAGGER_LIGHTGBM_RANKER_PATH="$LGB_RUN/tree_ranker.json"
echo "LGB_RUN=$LGB_RUN"
echo "CSE2026_DAGGER_LIGHTGBM_RANKER_PATH=$CSE2026_DAGGER_LIGHTGBM_RANKER_PATH"

echo START_DAGGER_TREE_MVP80_ROLLOUT
python3 scripts/experiments/run_eon_experiment.py \
  --config configs/experiments/eon/remote_ong_rollout_mvp80_dagger_tree_energy_ksp_guarded_compare.yaml
ROLLOUT_RUN=$(ls -td runs/eon/eon_ong_rollout_mvp80_dagger_tree_energy_ksp_guarded_compare/* | head -1)
export ROLLOUT_RUN
echo "ROLLOUT_RUN=$ROLLOUT_RUN"

python3 - <<'PY'
import json
import os
from pathlib import Path

for label, env_name in [
    ("XGBOOST", "CSE2026_DAGGER_XGBOOST_RANKER_PATH"),
    ("LIGHTGBM", "CSE2026_DAGGER_LIGHTGBM_RANKER_PATH"),
]:
    run = Path(os.environ[env_name]).parent
    with (run / "metrics.json").open(encoding="utf-8") as fh:
        metrics = json.load(fh)
    print(label + "_TRAIN", json.dumps({
        "ranker_path": metrics.get("ranker_path"),
        "train": metrics.get("train"),
        "eval": metrics.get("eval"),
    }, sort_keys=True))

rollout = Path(os.environ.get("ROLLOUT_RUN", ""))
if not rollout:
    candidates = sorted(Path("runs/eon/eon_ong_rollout_mvp80_dagger_tree_energy_ksp_guarded_compare").glob("*"))
    rollout = candidates[-1]
with (rollout / "metrics.json").open(encoding="utf-8") as fh:
    metrics = json.load(fh)
print("ROLLOUT_SUMMARY", json.dumps(metrics.get("per_policy"), sort_keys=True))
PY

echo DAGGER_TREE_ENERGY_KSP_FINISHED
