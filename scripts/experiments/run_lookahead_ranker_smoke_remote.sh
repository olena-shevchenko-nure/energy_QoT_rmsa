#!/usr/bin/env bash
set -euo pipefail

echo START_LOOKAHEAD_TRAIN_LABELS
python3 scripts/experiments/run_eon_experiment.py --config configs/experiments/eon/remote_lookahead_oracle_train16.yaml
LABEL_RUN=$(ls -td runs/eon/eon_lookahead_oracle_train16/* | head -1)
LABEL_CSV="$LABEL_RUN/lookahead_oracle_samples.csv"
echo "LABEL_RUN=$LABEL_RUN"
echo "LABEL_CSV=$LABEL_CSV"
LABEL_RUN="$LABEL_RUN" python3 - <<'PY'
import json
import os

run = os.environ["LABEL_RUN"]
with open(run + "/metrics.json", encoding="utf-8") as fh:
    metrics = json.load(fh)
print("LABEL_SUMMARY", json.dumps(metrics["summary"], sort_keys=True))
PY

echo START_LOOKAHEAD_RANKER_TRAIN
export CSE2026_LOOKAHEAD_TARGET_PATH="$LABEL_CSV"
python3 scripts/experiments/run_eon_experiment.py --config configs/experiments/eon/remote_train_dqn_mvp_fast_lookahead_ranker_smoke.yaml
TRAIN_RUN=$(ls -td runs/eon/eon_train_dqn_mvp_fast_lookahead_ranker_smoke/* | head -1)
echo "TRAIN_RUN=$TRAIN_RUN"
TRAIN_RUN="$TRAIN_RUN" python3 - <<'PY'
import json
import os

run = os.environ["TRAIN_RUN"]
with open(run + "/metrics.json", encoding="utf-8") as fh:
    metrics = json.load(fh)
print(
    "TRAIN_SUMMARY",
    json.dumps(
        {
            "best_epoch": metrics.get("best_epoch"),
            "best_checkpoint": metrics.get("best_checkpoint"),
            "history_tail": metrics.get("history", [])[-3:],
        },
        sort_keys=True,
    ),
)
PY

echo START_LOOKAHEAD_RANKER_ROLLOUT
export CSE2026_DQN_CHECKPOINT="$TRAIN_RUN/dqn_best.pt"
python3 scripts/experiments/run_eon_experiment.py --config configs/experiments/eon/remote_ong_rollout_mvp20_lookahead_ranker_smoke.yaml
ROLLOUT_RUN=$(ls -td runs/eon/eon_ong_rollout_mvp20_lookahead_ranker_smoke/* | head -1)
echo "ROLLOUT_RUN=$ROLLOUT_RUN"
ROLLOUT_RUN="$ROLLOUT_RUN" python3 - <<'PY'
import json
import os

run = os.environ["ROLLOUT_RUN"]
with open(run + "/metrics.json", encoding="utf-8") as fh:
    metrics = json.load(fh)
print("ROLLOUT_SUMMARY", json.dumps(metrics["per_policy"], sort_keys=True))
PY

echo LOOKAHEAD_RANKER_SMOKE_FINISHED
