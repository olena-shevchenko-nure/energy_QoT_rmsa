#!/usr/bin/env bash
set -euo pipefail

echo START_LOOKAHEAD_VAL_LABELS
python3 scripts/experiments/run_eon_experiment.py --config configs/experiments/eon/remote_lookahead_oracle_val16.yaml
VAL_RUN=$(ls -td runs/eon/eon_lookahead_oracle_val16/* | head -1)
VAL_CSV="$VAL_RUN/lookahead_oracle_samples.csv"
echo "VAL_RUN=$VAL_RUN"
echo "VAL_CSV=$VAL_CSV"

TRAIN_CSV=${CSE2026_LOOKAHEAD_TRAIN_PATH:-runs/eon/eon_lookahead_oracle_train16/20260531_113830_unknown/lookahead_oracle_samples.csv}
TEST_CSV=${CSE2026_LOOKAHEAD_TEST_PATH:-runs/eon/eon_lookahead_oracle_mvp16/20260531_112707_unknown/lookahead_oracle_samples.csv}
echo "TRAIN_CSV=$TRAIN_CSV"
echo "TEST_CSV=$TEST_CSV"

echo START_OVERRIDE_TRAIN
export CSE2026_LOOKAHEAD_TRAIN_PATH="$TRAIN_CSV"
export CSE2026_LOOKAHEAD_VAL_PATH="$VAL_CSV"
export CSE2026_LOOKAHEAD_TEST_PATH="$TEST_CSV"
python3 scripts/experiments/run_eon_experiment.py --config configs/experiments/eon/remote_train_lookahead_override_smoke.yaml
OVERRIDE_RUN=$(ls -td runs/eon/eon_train_lookahead_override_smoke/* | head -1)
OVERRIDE_MODEL="$OVERRIDE_RUN/override_classifier.json"
echo "OVERRIDE_RUN=$OVERRIDE_RUN"
echo "OVERRIDE_MODEL=$OVERRIDE_MODEL"
OVERRIDE_RUN="$OVERRIDE_RUN" python3 - <<'PY'
import json
import os

run = os.environ["OVERRIDE_RUN"]
with open(run + "/metrics.json", encoding="utf-8") as fh:
    metrics = json.load(fh)
print(
    "OVERRIDE_SUMMARY",
    json.dumps(
        {
            "threshold": metrics.get("threshold"),
            "threshold_tuning": metrics.get("threshold_tuning"),
            "splits": metrics.get("splits"),
        },
        sort_keys=True,
    ),
)
PY

echo START_OVERRIDE_ROLLOUT
export CSE2026_OVERRIDE_CLASSIFIER_PATH="$OVERRIDE_MODEL"
export CSE2026_DQN_CHECKPOINT=${CSE2026_DQN_CHECKPOINT:-runs/eon/eon_train_dqn_mvp_fast_residual_clamp050/20260531_101136_unknown/dqn_best.pt}
python3 scripts/experiments/run_eon_experiment.py --config configs/experiments/eon/remote_ong_rollout_mvp20_override_smoke.yaml
ROLLOUT_RUN=$(ls -td runs/eon/eon_ong_rollout_mvp20_override_smoke/* | head -1)
echo "ROLLOUT_RUN=$ROLLOUT_RUN"
ROLLOUT_RUN="$ROLLOUT_RUN" python3 - <<'PY'
import json
import os

run = os.environ["ROLLOUT_RUN"]
with open(run + "/metrics.json", encoding="utf-8") as fh:
    metrics = json.load(fh)
print("ROLLOUT_SUMMARY", json.dumps(metrics["per_policy"], sort_keys=True))
PY

echo LOOKAHEAD_OVERRIDE_SMOKE_FINISHED
