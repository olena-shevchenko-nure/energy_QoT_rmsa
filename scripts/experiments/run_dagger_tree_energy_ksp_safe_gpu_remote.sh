#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p runs/launch_logs
if [[ "${CSE2026_SAFE_GPU_REMOTE_LOG:-0}" != "1" ]]; then
  export CSE2026_SAFE_GPU_REMOTE_LOG=1
  LOG_PATH="runs/launch_logs/dagger_tree_energy_ksp_safe_gpu_$(date +%Y%m%d_%H%M%S).log"
  exec > >(tee -a "$LOG_PATH") 2>&1
fi
echo "LOG_PATH=${LOG_PATH:-already_redirected}"

echo START_GPU_PREFLIGHT
python3 - <<'PY'
import numpy as np

import xgboost as xgb
print("XGBOOST_VERSION", xgb.__version__)
x = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]], dtype=np.float32)
y = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
dtrain = xgb.DMatrix(x, label=y)
dtrain.set_group(np.asarray([2, 2], dtype=np.uint32))
version_parts = tuple(int(part) for part in xgb.__version__.split(".")[:2] if part.isdigit())
xgb_params = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@1",
    "eta": 0.1,
    "max_depth": 2,
    "seed": 42,
}
if version_parts and version_parts < (2, 0):
    xgb_params["tree_method"] = "gpu_hist"
else:
    xgb_params["tree_method"] = "hist"
    xgb_params["device"] = "cuda"
xgb.train(xgb_params, dtrain, num_boost_round=1, verbose_eval=False)
print("XGBOOST_GPU_PREFLIGHT_OK")

import lightgbm as lgb
print("LIGHTGBM_VERSION", lgb.__version__)
dtrain = lgb.Dataset(x, label=y, group=np.asarray([2, 2], dtype=np.int32), free_raw_data=False)
lgb.train(
    {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1],
        "learning_rate": 0.1,
        "num_leaves": 3,
        "min_data_in_leaf": 1,
        "device_type": "gpu",
        "max_bin": 63,
        "verbosity": -1,
        "seed": 42,
    },
    dtrain,
    num_boost_round=1,
    callbacks=[lgb.log_evaluation(period=0)],
)
print("LIGHTGBM_GPU_PREFLIGHT_OK")
PY

echo START_DAGGER_SAFE_XGBOOST_ENERGY_KSP
python3 scripts/experiments/run_eon_experiment.py \
  --config configs/experiments/eon/remote_train_dagger_tree_ranker_xgboost_energy_ksp_safe_mvp16_gpu.yaml
XGB_RUN=$(ls -td runs/eon/eon_train_dagger_tree_ranker_xgboost_energy_ksp_safe_mvp16_gpu/* | head -1)
export CSE2026_DAGGER_SAFE_XGBOOST_RANKER_PATH="$XGB_RUN/tree_ranker.json"
echo "XGB_RUN=$XGB_RUN"
echo "CSE2026_DAGGER_SAFE_XGBOOST_RANKER_PATH=$CSE2026_DAGGER_SAFE_XGBOOST_RANKER_PATH"

echo START_DAGGER_SAFE_LIGHTGBM_ENERGY_KSP
python3 scripts/experiments/run_eon_experiment.py \
  --config configs/experiments/eon/remote_train_dagger_tree_ranker_lightgbm_energy_ksp_safe_mvp16_gpu.yaml
LGB_RUN=$(ls -td runs/eon/eon_train_dagger_tree_ranker_lightgbm_energy_ksp_safe_mvp16_gpu/* | head -1)
export CSE2026_DAGGER_SAFE_LIGHTGBM_RANKER_PATH="$LGB_RUN/tree_ranker.json"
echo "LGB_RUN=$LGB_RUN"
echo "CSE2026_DAGGER_SAFE_LIGHTGBM_RANKER_PATH=$CSE2026_DAGGER_SAFE_LIGHTGBM_RANKER_PATH"

echo START_DAGGER_SAFE_TREE_MVP80_ROLLOUT
python3 scripts/experiments/run_eon_experiment.py \
  --config configs/experiments/eon/remote_ong_rollout_mvp80_dagger_tree_energy_ksp_safe_guarded_compare.yaml
ROLLOUT_RUN=$(ls -td runs/eon/eon_ong_rollout_mvp80_dagger_tree_energy_ksp_safe_guarded_compare/* | head -1)
export ROLLOUT_RUN
echo "ROLLOUT_RUN=$ROLLOUT_RUN"

python3 - <<'PY'
import json
import os
from pathlib import Path

for label, env_name in [
    ("SAFE_XGBOOST", "CSE2026_DAGGER_SAFE_XGBOOST_RANKER_PATH"),
    ("SAFE_LIGHTGBM", "CSE2026_DAGGER_SAFE_LIGHTGBM_RANKER_PATH"),
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
    candidates = sorted(Path("runs/eon/eon_ong_rollout_mvp80_dagger_tree_energy_ksp_safe_guarded_compare").glob("*"))
    rollout = candidates[-1]
with (rollout / "metrics.json").open(encoding="utf-8") as fh:
    metrics = json.load(fh)
print("SAFE_ROLLOUT_SUMMARY", json.dumps(metrics.get("per_policy"), sort_keys=True))
PY

echo DAGGER_TREE_ENERGY_KSP_SAFE_GPU_FINISHED
