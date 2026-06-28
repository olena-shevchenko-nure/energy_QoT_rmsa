#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN_TAG="${CSE2026_XLRON_CF_RANK_RUN_TAG:-g160_v1}"
BASE_DIR="${CSE2026_XLRON_CF_BASE_DIR:-runs/eon/quick_runtime_artifacts}"
CONFIG="${CSE2026_XLRON_LIVE_RISK_CONFIG:-configs/experiments/eon/remote_collect_online_dqn_base_alltopn_h100_train_stratified.yaml}"
INPUT_DIR="${CSE2026_XLRON_LIVE_RISK_INPUT_DIR:-$BASE_DIR/top32_xlron_cf_online_signal_stable_h100_h50_${RUN_TAG}_calibrated_protected_v1}"
XLRON_CHECKPOINT="${CSE2026_XLRON_LIVE_RISK_CHECKPOINT:-$BASE_DIR/top32_xlron_dagger_student32_16_preserve16_e4_lr6e5/top32_xlron_full_dqn_distill_best.pt}"
OUTPUT_DIR="${CSE2026_XLRON_LIVE_RISK_OUTPUT_DIR:-$BASE_DIR/top32_xlron_live_risk_selector_${RUN_TAG}_v1}"
ROLLOUT_OUTPUT_DIR="${CSE2026_XLRON_LIVE_RISK_ROLLOUT_OUTPUT_DIR:-${OUTPUT_DIR}_four_slice_rollout}"
PROTECTED_BUCKETS="${CSE2026_XLRON_LIVE_RISK_PROTECTED_BUCKETS:-bursty:high,bursty:medium,bursty:overload,hotspot:high,hotspot:medium,nonuniform:high,nonuniform:medium}"

python3 scripts/experiments/train_top32_xlron_live_risk_selector.py \
  --config "$CONFIG" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --xlron-checkpoint "$XLRON_CHECKPOINT" \
  --protected-buckets "$PROTECTED_BUCKETS" \
  --apply-buckets "${CSE2026_XLRON_LIVE_RISK_APPLY_BUCKETS:-$PROTECTED_BUCKETS}" \
  --training-buckets "${CSE2026_XLRON_LIVE_RISK_TRAINING_BUCKETS:-}" \
  --proposal-mode "${CSE2026_XLRON_LIVE_RISK_PROPOSAL_MODE:-live_or_all}" \
  --batch-size "${CSE2026_XLRON_LIVE_RISK_BATCH_SIZE:-64}" \
  --learning-rate "${CSE2026_XLRON_LIVE_RISK_LR:-0.05}" \
  --num-boost-round "${CSE2026_XLRON_LIVE_RISK_NUM_BOOST_ROUND:-160}" \
  --early-stopping-rounds "${CSE2026_XLRON_LIVE_RISK_EARLY_STOPPING_ROUNDS:-20}" \
  --max-depth "${CSE2026_XLRON_LIVE_RISK_MAX_DEPTH:-3}" \
  --loss-sample-weight "${CSE2026_XLRON_LIVE_RISK_LOSS_SAMPLE_WEIGHT:-8.0}" \
  --nonloss-sample-weight "${CSE2026_XLRON_LIVE_RISK_NONLOSS_SAMPLE_WEIGHT:-1.0}" \
  --max-veto-rate "${CSE2026_XLRON_LIVE_RISK_MAX_VETO_RATE:-0.12}" \
  --max-veto-win-rate "${CSE2026_XLRON_LIVE_RISK_MAX_VETO_WIN_RATE:-0.40}" \
  --min-delta-improvement "${CSE2026_XLRON_LIVE_RISK_MIN_DELTA_IMPROVEMENT:-0.0}" \
  --min-veto-count "${CSE2026_XLRON_LIVE_RISK_MIN_VETO_COUNT:-2}" \
  --seed "${CSE2026_XLRON_LIVE_RISK_SEED:-20260610}"

export CSE2026_XLRON_CF_RANK_CONFIG="$CONFIG"
export CSE2026_XLRON_CF_RANK_INPUT_DIR="$INPUT_DIR"
export CSE2026_XLRON_CF_RANK_INITIAL_CHECKPOINT="$XLRON_CHECKPOINT"
export CSE2026_XLRON_CF_RANK_REFERENCE_CHECKPOINT="$XLRON_CHECKPOINT"
export CSE2026_XLRON_CF_RANK_OUTPUT_DIR="$ROLLOUT_OUTPUT_DIR"
export CSE2026_XLRON_CF_RANK_EPOCHS=0
export CSE2026_XLRON_CF_RANK_BATCH_SIZE="${CSE2026_XLRON_CF_RANK_BATCH_SIZE:-32}"
export CSE2026_XLRON_CF_RANK_CHECKPOINT_SELECTION="${CSE2026_XLRON_CF_RANK_CHECKPOINT_SELECTION:-rollout_bucket_guard_score}"
export CSE2026_XLRON_CF_RANK_ROLLOUT_VAL_MAX_EPISODES="${CSE2026_XLRON_CF_RANK_ROLLOUT_VAL_MAX_EPISODES:-16}"
export CSE2026_XLRON_CF_RANK_ROLLOUT_VAL_SLICES="${CSE2026_XLRON_CF_RANK_ROLLOUT_VAL_SLICES:-4}"
export CSE2026_XLRON_CF_RANK_ROLLOUT_VAL_SLICE_STRIDE="${CSE2026_XLRON_CF_RANK_ROLLOUT_VAL_SLICE_STRIDE:-1}"
export CSE2026_XLRON_CF_RANK_ROLLOUT_NEGATIVE_BUCKET_PENALTY="${CSE2026_XLRON_CF_RANK_ROLLOUT_NEGATIVE_BUCKET_PENALTY:-0.5}"
export CSE2026_XLRON_CF_RANK_ROLLOUT_PROTECTED_BUCKETS="${CSE2026_XLRON_CF_RANK_ROLLOUT_PROTECTED_BUCKETS:-$PROTECTED_BUCKETS}"
export CSE2026_XLRON_CF_RANK_ROLLOUT_PROTECTED_BUCKET_PENALTY="${CSE2026_XLRON_CF_RANK_ROLLOUT_PROTECTED_BUCKET_PENALTY:-8.0}"
export CSE2026_XLRON_CF_RANK_LIVE_RISK_SELECTOR_PATH="$OUTPUT_DIR/xlron_live_risk_selector_artifact.json"
export CSE2026_XLRON_CF_RANK_LIVE_RISK_SELECTOR_THRESHOLD="${CSE2026_XLRON_LIVE_RISK_RUNTIME_THRESHOLD:--1.0}"
export CSE2026_XLRON_CF_RANK_LIVE_RISK_SELECTOR_BUCKETS="${CSE2026_XLRON_LIVE_RISK_APPLY_BUCKETS:-$PROTECTED_BUCKETS}"
export CSE2026_XLRON_CF_RANK_LIVE_RISK_SELECTOR_BASE_INDEX="${CSE2026_XLRON_LIVE_RISK_BASE_INDEX:-0}"
bash scripts/experiments/run_top32_xlron_cf_rank_finetune_remote.sh

python3 - <<PY
import json
from pathlib import Path

train_path = Path("$OUTPUT_DIR") / "xlron_live_risk_selector_summary.json"
rollout_path = Path("$ROLLOUT_OUTPUT_DIR") / "top32_xlron_counterfactual_rank_finetune_summary.json"
train = json.loads(train_path.read_text()) if train_path.exists() else {}
rollout = json.loads(rollout_path.read_text()) if rollout_path.exists() else {}
rollout_eval = rollout.get("best_rollout_val_eval") or rollout.get("initial_rollout_val_eval") or {}
risk = rollout_eval.get("live_risk_selector") or {}
summary = {
    "train_summary": str(train_path),
    "rollout_summary": str(rollout_path),
    "selector_examples": train.get("examples"),
    "selector_labels": train.get("labels"),
    "selector_threshold": train.get("threshold"),
    "selector_calibration_metrics": train.get("calibration_metrics"),
    "selector_eval_metrics": train.get("eval_metrics"),
    "rollout_accepted": rollout_eval.get("accepted"),
    "rollout_best_score": rollout.get("best_score"),
    "rollout_blocking_rate": rollout_eval.get("blocking_rate"),
    "risk_fallbacks": risk.get("fallbacks"),
    "risk_fallback_rate": risk.get("fallback_rate"),
    "risk_nonbase_candidates": risk.get("nonbase_candidates"),
    "risk_by_bucket": risk.get("by_bucket"),
}
out = Path("$OUTPUT_DIR") / "xlron_live_risk_selector_rollout_summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(out)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
