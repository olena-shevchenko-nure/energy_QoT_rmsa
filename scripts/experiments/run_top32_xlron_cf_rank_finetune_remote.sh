#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="${CSE2026_XLRON_CF_RANK_CONFIG:-configs/experiments/eon/remote_collect_online_dqn_base_alltopn_h100_train_stratified.yaml}"
INPUT_DIR="${CSE2026_XLRON_CF_RANK_INPUT_DIR:-runs/eon/quick_runtime_artifacts/top32_xlron_cf_online_hard_buckets_h100_g40_stable_h50confirm}"
INITIAL_CHECKPOINT="${CSE2026_XLRON_CF_RANK_INITIAL_CHECKPOINT:-runs/eon/quick_runtime_artifacts/top32_xlron_dagger_student32_16_preserve16_e4_lr6e5/top32_xlron_full_dqn_distill_best.pt}"
REFERENCE_CHECKPOINT="${CSE2026_XLRON_CF_RANK_REFERENCE_CHECKPOINT:-$INITIAL_CHECKPOINT}"
OUTPUT_DIR="${CSE2026_XLRON_CF_RANK_OUTPUT_DIR:-runs/eon/quick_runtime_artifacts/top32_xlron_cf_rank_h100_g40_stable_pilot}"

python3 scripts/experiments/train_top32_xlron_counterfactual_rank_finetune.py \
  --config "$CONFIG" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --initial-xlron-checkpoint "$INITIAL_CHECKPOINT" \
  --reference-xlron-checkpoint "$REFERENCE_CHECKPOINT" \
  --epochs "${CSE2026_XLRON_CF_RANK_EPOCHS:-6}" \
  --batch-size "${CSE2026_XLRON_CF_RANK_BATCH_SIZE:-16}" \
  --learning-rate "${CSE2026_XLRON_CF_RANK_LR:-1e-5}" \
  --weight-decay "${CSE2026_XLRON_CF_RANK_WEIGHT_DECAY:-1e-4}" \
  --freeze-mode "${CSE2026_XLRON_CF_RANK_FREEZE_MODE:-ranker_light}" \
  --partial-unfreeze-epoch "${CSE2026_XLRON_CF_RANK_PARTIAL_UNFREEZE_EPOCH:-0}" \
  --partial-unfreeze-mode "${CSE2026_XLRON_CF_RANK_PARTIAL_UNFREEZE_MODE:-ranker_action_fusion}" \
  --partial-unfreeze-lr "${CSE2026_XLRON_CF_RANK_PARTIAL_UNFREEZE_LR:-0}" \
  --partial-unfreeze-low-lr-markers "${CSE2026_XLRON_CF_RANK_PARTIAL_UNFREEZE_LOW_LR_MARKERS:-action_encoder,context_encoder,full_route_project,spectrum_encoder}" \
  --ce-weight "${CSE2026_XLRON_CF_RANK_CE_WEIGHT:-0.20}" \
  --listwise-weight "${CSE2026_XLRON_CF_RANK_LISTWISE_WEIGHT:-1.00}" \
  --pairwise-weight "${CSE2026_XLRON_CF_RANK_PAIRWISE_WEIGHT:-0.75}" \
  --base-pairwise-weight "${CSE2026_XLRON_CF_RANK_BASE_PAIRWISE_WEIGHT:-0.60}" \
  --regression-weight "${CSE2026_XLRON_CF_RANK_REGRESSION_WEIGHT:-0.03}" \
  --reference-kl-weight "${CSE2026_XLRON_CF_RANK_REFERENCE_KL_WEIGHT:-3.00}" \
  --oracle-top1-weight "${CSE2026_XLRON_CF_RANK_ORACLE_TOP1_WEIGHT:-0}" \
  --oracle-margin-weight "${CSE2026_XLRON_CF_RANK_ORACLE_MARGIN_WEIGHT:-0}" \
  --oracle-accepted-scale "${CSE2026_XLRON_CF_RANK_ORACLE_ACCEPTED_SCALE:-8.0}" \
  --oracle-margin "${CSE2026_XLRON_CF_RANK_ORACLE_MARGIN:-0.35}" \
  ${CSE2026_XLRON_CF_RANK_ORACLE_POSITIVE_ONLY:+--oracle-positive-only} \
  --target-temperature "${CSE2026_XLRON_CF_RANK_TARGET_TEMPERATURE:-1.0}" \
  --student-temperature "${CSE2026_XLRON_CF_RANK_STUDENT_TEMPERATURE:-1.0}" \
  --reference-temperature "${CSE2026_XLRON_CF_RANK_REFERENCE_TEMPERATURE:-2.0}" \
  --pairwise-margin "${CSE2026_XLRON_CF_RANK_PAIRWISE_MARGIN:-0.12}" \
  --order-epsilon "${CSE2026_XLRON_CF_RANK_ORDER_EPSILON:-0.05}" \
  --target-scale "${CSE2026_XLRON_CF_RANK_TARGET_SCALE:-4.0}" \
  --checkpoint-selection "${CSE2026_XLRON_CF_RANK_CHECKPOINT_SELECTION:-rollout_accepted}" \
  --rollout-val-split "${CSE2026_XLRON_CF_RANK_ROLLOUT_VAL_SPLIT:-val}" \
  --rollout-val-max-episodes "${CSE2026_XLRON_CF_RANK_ROLLOUT_VAL_MAX_EPISODES:-8}" \
  --rollout-val-max-requests-per-episode "${CSE2026_XLRON_CF_RANK_ROLLOUT_VAL_MAX_REQUESTS:-0}" \
  --rollout-val-episode-selection "${CSE2026_XLRON_CF_RANK_ROLLOUT_VAL_EPISODE_SELECTION:-stratified}" \
  --rollout-val-slices "${CSE2026_XLRON_CF_RANK_ROLLOUT_VAL_SLICES:-1}" \
  --rollout-val-slice-stride "${CSE2026_XLRON_CF_RANK_ROLLOUT_VAL_SLICE_STRIDE:-1}" \
  --rollout-val-episode-offset "${CSE2026_XLRON_CF_RANK_ROLLOUT_VAL_EPISODE_OFFSET:-0}" \
  --rollout-val-reference-policy "${CSE2026_XLRON_CF_RANK_ROLLOUT_REFERENCE_POLICY:-energy-aware-ksp-bm-ff}" \
  --rollout-worst-bucket-penalty "${CSE2026_XLRON_CF_RANK_ROLLOUT_WORST_BUCKET_PENALTY:-4.0}" \
  --rollout-negative-bucket-penalty "${CSE2026_XLRON_CF_RANK_ROLLOUT_NEGATIVE_BUCKET_PENALTY:-0.0}" \
  --rollout-protected-buckets "${CSE2026_XLRON_CF_RANK_ROLLOUT_PROTECTED_BUCKETS:-}" \
  --rollout-protected-bucket-min-delta "${CSE2026_XLRON_CF_RANK_ROLLOUT_PROTECTED_BUCKET_MIN_DELTA:-0}" \
  --rollout-protected-bucket-penalty "${CSE2026_XLRON_CF_RANK_ROLLOUT_PROTECTED_BUCKET_PENALTY:-0.0}" \
  --rollout-runtime-guard-buckets "${CSE2026_XLRON_CF_RANK_RUNTIME_GUARD_BUCKETS:-}" \
  --rollout-runtime-guard-bucket-margins "${CSE2026_XLRON_CF_RANK_RUNTIME_GUARD_BUCKET_MARGINS:-}" \
  --rollout-runtime-guard-min-margin "${CSE2026_XLRON_CF_RANK_RUNTIME_GUARD_MIN_MARGIN:-0.0}" \
  --rollout-runtime-guard-base-index "${CSE2026_XLRON_CF_RANK_RUNTIME_GUARD_BASE_INDEX:-0}" \
  --rollout-live-risk-selector-path "${CSE2026_XLRON_CF_RANK_LIVE_RISK_SELECTOR_PATH:-}" \
  --rollout-live-risk-selector-threshold "${CSE2026_XLRON_CF_RANK_LIVE_RISK_SELECTOR_THRESHOLD:--1.0}" \
  --rollout-live-risk-selector-buckets "${CSE2026_XLRON_CF_RANK_LIVE_RISK_SELECTOR_BUCKETS:-}" \
  --rollout-live-risk-selector-base-index "${CSE2026_XLRON_CF_RANK_LIVE_RISK_SELECTOR_BASE_INDEX:-0}" \
  --early-stop-patience "${CSE2026_XLRON_CF_RANK_EARLY_STOP_PATIENCE:-3}" \
  --seed "${CSE2026_XLRON_CF_RANK_SEED:-20260609}" \
  --progress-every-batches "${CSE2026_XLRON_CF_RANK_PROGRESS_EVERY_BATCHES:-10}"
