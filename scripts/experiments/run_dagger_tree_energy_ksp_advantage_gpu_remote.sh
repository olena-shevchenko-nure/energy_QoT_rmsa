#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p runs/launch_logs
if [[ "${CSE2026_ADV_GPU_REMOTE_LOG:-0}" != "1" ]]; then
  export CSE2026_ADV_GPU_REMOTE_LOG=1
  LOG_PATH="runs/launch_logs/dagger_tree_energy_ksp_advantage_gpu_$(date +%Y%m%d_%H%M%S).log"
  exec > >(tee -a "$LOG_PATH") 2>&1
fi
echo "LOG_PATH=${LOG_PATH:-already_redirected}"

echo START_GPU_PREFLIGHT
python3 - <<'PY'
import numpy as np

import xgboost as xgb
print("XGBOOST_VERSION", xgb.__version__)
x = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]], dtype=np.float32)
y_binary = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
y_reg = np.asarray([-1.0, 1.0, -0.5, 0.5], dtype=np.float32)
version_parts = tuple(int(part) for part in xgb.__version__.split(".")[:2] if part.isdigit())
xgb_device = {"tree_method": "gpu_hist"} if version_parts and version_parts < (2, 0) else {"tree_method": "hist", "device": "cuda"}
for objective, label in [("binary:logistic", y_binary), ("reg:squarederror", y_reg)]:
    dtrain = xgb.DMatrix(x, label=label)
    params = {"objective": objective, "eta": 0.1, "max_depth": 2, "seed": 42, **xgb_device}
    xgb.train(params, dtrain, num_boost_round=1, verbose_eval=False)
print("XGBOOST_GPU_PREFLIGHT_OK")

import lightgbm as lgb
print("LIGHTGBM_VERSION", lgb.__version__)
for objective, label in [("binary", y_binary), ("regression", y_reg)]:
    dtrain = lgb.Dataset(x, label=label, free_raw_data=False)
    lgb.train(
        {
            "objective": objective,
            "metric": "binary_logloss" if objective == "binary" else "rmse",
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

if [[ "${CSE2026_DAGGER_ADV_SKIP_XGBOOST:-0}" == "1" ]]; then
  if [[ -z "${CSE2026_DAGGER_ADV_XGBOOST_RANKER_PATH:-}" ]]; then
    echo "CSE2026_DAGGER_ADV_SKIP_XGBOOST=1 requires CSE2026_DAGGER_ADV_XGBOOST_RANKER_PATH" >&2
    exit 2
  fi
  XGB_RUN="$(dirname "$CSE2026_DAGGER_ADV_XGBOOST_RANKER_PATH")"
  echo "SKIP_DAGGER_ADV_XGBOOST_ENERGY_KSP"
else
  echo START_DAGGER_ADV_XGBOOST_ENERGY_KSP
  python3 scripts/experiments/run_eon_experiment.py \
    --config configs/experiments/eon/remote_train_dagger_tree_ranker_xgboost_energy_ksp_advantage_mvp16_gpu.yaml
  XGB_RUN=$(ls -td runs/eon/eon_train_dagger_tree_ranker_xgboost_energy_ksp_advantage_mvp16_gpu/* | head -1)
  export CSE2026_DAGGER_ADV_XGBOOST_RANKER_PATH="$XGB_RUN/tree_ranker.json"
fi
echo "XGB_RUN=$XGB_RUN"
echo "CSE2026_DAGGER_ADV_XGBOOST_RANKER_PATH=$CSE2026_DAGGER_ADV_XGBOOST_RANKER_PATH"

echo START_DAGGER_ADV_LIGHTGBM_ENERGY_KSP
python3 scripts/experiments/run_eon_experiment.py \
  --config configs/experiments/eon/remote_train_dagger_tree_ranker_lightgbm_energy_ksp_advantage_mvp16_gpu.yaml
LGB_RUN=$(ls -td runs/eon/eon_train_dagger_tree_ranker_lightgbm_energy_ksp_advantage_mvp16_gpu/* | head -1)
export CSE2026_DAGGER_ADV_LIGHTGBM_RANKER_PATH="$LGB_RUN/tree_ranker.json"
echo "LGB_RUN=$LGB_RUN"
echo "CSE2026_DAGGER_ADV_LIGHTGBM_RANKER_PATH=$CSE2026_DAGGER_ADV_LIGHTGBM_RANKER_PATH"

echo START_DAGGER_ADV_TREE_MVP80_ROLLOUT
python3 scripts/experiments/run_eon_experiment.py \
  --config configs/experiments/eon/remote_ong_rollout_mvp80_dagger_tree_energy_ksp_advantage_compare.yaml
ROLLOUT_RUN=$(ls -td runs/eon/eon_ong_rollout_mvp80_dagger_tree_energy_ksp_advantage_compare/* | head -1)
export ROLLOUT_RUN
echo "ROLLOUT_RUN=$ROLLOUT_RUN"

python3 - <<'PY'
import json
import os
from pathlib import Path

for label, env_name in [
    ("ADV_XGBOOST", "CSE2026_DAGGER_ADV_XGBOOST_RANKER_PATH"),
    ("ADV_LIGHTGBM", "CSE2026_DAGGER_ADV_LIGHTGBM_RANKER_PATH"),
]:
    run = Path(os.environ[env_name]).parent
    with (run / "metrics.json").open(encoding="utf-8") as fh:
        metrics = json.load(fh)
    print(label + "_TRAIN", json.dumps({
        "ranker_path": metrics.get("ranker_path"),
        "train": metrics.get("train"),
        "eval": metrics.get("eval"),
        "advantage_gate": metrics.get("advantage_gate"),
    }, sort_keys=True))

rollout = Path(os.environ.get("ROLLOUT_RUN", ""))
if not rollout:
    candidates = sorted(Path("runs/eon/eon_ong_rollout_mvp80_dagger_tree_energy_ksp_advantage_compare").glob("*"))
    rollout = candidates[-1]
with (rollout / "metrics.json").open(encoding="utf-8") as fh:
    metrics = json.load(fh)
per_policy = {row["policy"]: row for row in metrics.get("per_policy", [])}
base = per_policy.get("energy-aware-ksp-bm-ff", {})
print("ADV_ROLLOUT_SUMMARY", json.dumps(metrics.get("per_policy"), sort_keys=True))
comparisons = {}
for policy in ("xgboost_candidate_ranker", "lightgbm_candidate_ranker"):
    row = per_policy.get(policy, {})
    comparisons[policy] = {
        "accepted_delta_vs_energy_aware": int(row.get("accepted", 0)) - int(base.get("accepted", 0)),
        "blocked_delta_vs_energy_aware": int(row.get("blocked", 0)) - int(base.get("blocked", 0)),
        "blocking_rate_delta_vs_energy_aware": float(row.get("blocking_rate", 0.0)) - float(base.get("blocking_rate", 0.0)),
        "reward_per_request_delta_vs_energy_aware": float(row.get("mean_reward_per_request", 0.0)) - float(base.get("mean_reward_per_request", 0.0)),
        "energy_delta_vs_energy_aware": float(row.get("mean_selected_energy_increment", 0.0)) - float(base.get("mean_selected_energy_increment", 0.0)),
        "fragmentation_delta_vs_energy_aware": float(row.get("mean_selected_fragmentation_after", 0.0)) - float(base.get("mean_selected_fragmentation_after", 0.0)),
        "qot_margin_delta_vs_energy_aware": float(row.get("mean_selected_qot_margin_norm", 0.0)) - float(base.get("mean_selected_qot_margin_norm", 0.0)),
        "override_rate": row.get("override_rate"),
        "mean_override_probability": row.get("mean_override_probability"),
    }
print("ADV_VS_ENERGY_AWARE", json.dumps(comparisons, sort_keys=True))
PY

echo DAGGER_TREE_ENERGY_KSP_ADVANTAGE_GPU_FINISHED
