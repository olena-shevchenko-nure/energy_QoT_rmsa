#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BASE_CONFIG="${CSE2026_XLRON_DISTILL_CONFIG:-configs/experiments/eon/remote_train_top32_xlron_full_ppo_mvp20_quick_gpu.yaml}"
FULL_DQN_STRATIFIED32_E5="${CSE2026_FULL_DQN_STRATIFIED32_E5_CHECKPOINT:-runs/eon/quick_runtime_artifacts/full_dqn_orate60_distill_frozen_mvp20_stratified32_e5/full_dqn_orate60_distill_frozen.pt}"
RUN_NAME="${CSE2026_XLRON_FULL_DQN_DISTILL_NAME:-top32_xlron_full_dqn_distill_stratified32_e5}"
OUTPUT_DIR="runs/eon/quick_runtime_artifacts/${RUN_NAME}"

HARD_CASE_WEIGHTS="${CSE2026_XLRON_HARD_CASE_WEIGHTS:-hotspot:medium=1.5,hotspot:overload=1.8,nonuniform:medium=1.5,nonuniform:overload=1.8,uniform:medium=1.5,uniform:high=1.6,uniform:overload=1.8,bursty:high=1.6,bursty:overload=1.8}"

python3 scripts/experiments/train_top32_xlron_full_dqn_distill.py \
  --config "$BASE_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --teacher-dqn-checkpoint "$FULL_DQN_STRATIFIED32_E5" \
  --initial-xlron-checkpoint "${CSE2026_XLRON_INITIAL_CHECKPOINT:-}" \
  --behavior-policy "${CSE2026_XLRON_BEHAVIOR_POLICY:-teacher}" \
  --behavior-xlron-checkpoint "${CSE2026_XLRON_BEHAVIOR_CHECKPOINT:-}" \
  --preserve-teacher-max-episodes "${CSE2026_XLRON_PRESERVE_TEACHER_EPISODES:-0}" \
  --preserve-weight "${CSE2026_XLRON_PRESERVE_WEIGHT:-0.5}" \
  --hard-dagger-loss-buckets "${CSE2026_XLRON_HARD_DAGGER_LOSS_BUCKETS:-}" \
  --hard-dagger-disagreement-weight "${CSE2026_XLRON_HARD_DAGGER_DISAGREE_WEIGHT:-1.0}" \
  --hard-dagger-loss-bucket-weight "${CSE2026_XLRON_HARD_DAGGER_BUCKET_WEIGHT:-1.0}" \
  --hard-dagger-agreement-weight "${CSE2026_XLRON_HARD_DAGGER_AGREE_WEIGHT:-1.0}" \
  --hard-dagger-agreement-keep-frac "${CSE2026_XLRON_HARD_DAGGER_AGREE_KEEP_FRAC:-1.0}" \
  --hard-dagger-max-weight "${CSE2026_XLRON_HARD_DAGGER_MAX_WEIGHT:-0.0}" \
  --counterfactual-aux-dir "${CSE2026_XLRON_COUNTERFACTUAL_AUX_DIR:-}" \
  --counterfactual-aux-weight "${CSE2026_XLRON_COUNTERFACTUAL_AUX_WEIGHT:-1.0}" \
  --counterfactual-aux-win-weight "${CSE2026_XLRON_COUNTERFACTUAL_WIN_WEIGHT:-2.0}" \
  --counterfactual-aux-loss-weight "${CSE2026_XLRON_COUNTERFACTUAL_LOSS_WEIGHT:-1.25}" \
  --counterfactual-aux-tie-weight "${CSE2026_XLRON_COUNTERFACTUAL_TIE_WEIGHT:-0.5}" \
  --counterfactual-aux-score-boost "${CSE2026_XLRON_COUNTERFACTUAL_SCORE_BOOST:-2.0}" \
  --counterfactual-aux-target-scale "${CSE2026_XLRON_COUNTERFACTUAL_TARGET_SCALE:-4.0}" \
  --counterfactual-aux-score-clip "${CSE2026_XLRON_COUNTERFACTUAL_SCORE_CLIP:-4.0}" \
  --counterfactual-aux-magnitude-weight "${CSE2026_XLRON_COUNTERFACTUAL_MAGNITUDE_WEIGHT:-0.15}" \
  --counterfactual-aux-magnitude-cap "${CSE2026_XLRON_COUNTERFACTUAL_MAGNITUDE_CAP:-4.0}" \
  --counterfactual-aux-max-examples "${CSE2026_XLRON_COUNTERFACTUAL_MAX_EXAMPLES:-0}" \
  --counterfactual-aux-mode "${CSE2026_XLRON_COUNTERFACTUAL_AUX_MODE:-hard_masked}" \
  --train-max-episodes "${CSE2026_XLRON_DISTILL_TRAIN_EPISODES:-32}" \
  --val-max-episodes "${CSE2026_XLRON_DISTILL_VAL_EPISODES:-8}" \
  --episode-selection stratified \
  --max-requests-per-episode "${CSE2026_XLRON_DISTILL_MAX_REQUESTS:-0}" \
  --epochs "${CSE2026_XLRON_DISTILL_EPOCHS:-4}" \
  --batch-size "${CSE2026_XLRON_DISTILL_BATCH_SIZE:-64}" \
  --learning-rate "${CSE2026_XLRON_DISTILL_LR:-0.00012}" \
  --weight-decay "${CSE2026_XLRON_DISTILL_WEIGHT_DECAY:-0.0001}" \
  --grad-clip-norm "${CSE2026_XLRON_DISTILL_GRAD_CLIP:-1.0}" \
  --ce-weight "${CSE2026_XLRON_DISTILL_CE_WEIGHT:-0.5}" \
  --listwise-kl-weight "${CSE2026_XLRON_DISTILL_KL_WEIGHT:-1.0}" \
  --listwise-temperature "${CSE2026_XLRON_DISTILL_KL_TEMP:-3.0}" \
  --pairwise-rank-weight "${CSE2026_XLRON_DISTILL_PAIRWISE_WEIGHT:-0.2}" \
  --pairwise-temperature "${CSE2026_XLRON_DISTILL_PAIRWISE_TEMP:-1.0}" \
  --pairwise-teacher-gap "${CSE2026_XLRON_DISTILL_PAIRWISE_GAP:-0.0}" \
  --hard-case-weight-rules "$HARD_CASE_WEIGHTS" \
  --checkpoint-selection "${CSE2026_XLRON_CHECKPOINT_SELECTION:-rollout_accepted}" \
  --rollout-val-max-episodes "${CSE2026_XLRON_ROLLOUT_VAL_EPISODES:-8}" \
  --rollout-val-max-requests-per-episode "${CSE2026_XLRON_ROLLOUT_VAL_MAX_REQUESTS:-0}" \
  --rollout-val-episode-selection stratified \
  --rollout-val-slices "${CSE2026_XLRON_ROLLOUT_VAL_SLICES:-1}" \
  --rollout-val-slice-stride "${CSE2026_XLRON_ROLLOUT_VAL_SLICE_STRIDE:-1}" \
  --rollout-val-episode-offset "${CSE2026_XLRON_ROLLOUT_VAL_EPISODE_OFFSET:-0}" \
  --rollout-val-reference-policy "${CSE2026_XLRON_ROLLOUT_REFERENCE_POLICY:-energy-aware-ksp-bm-ff}" \
  --rollout-worst-bucket-penalty "${CSE2026_XLRON_WORST_BUCKET_PENALTY:-4.0}" \
  --rollout-negative-bucket-penalty "${CSE2026_XLRON_NEGATIVE_BUCKET_PENALTY:-0.0}" \
  --rollout-protected-buckets "${CSE2026_XLRON_PROTECTED_BUCKETS:-}" \
  --rollout-protected-bucket-min-delta "${CSE2026_XLRON_PROTECTED_BUCKET_MIN_DELTA:-0}" \
  --rollout-protected-bucket-penalty "${CSE2026_XLRON_PROTECTED_BUCKET_PENALTY:-0.0}" \
  --transformer-embedding-size "${CSE2026_XLRON_EMBEDDING_SIZE:-128}" \
  --transformer-num-layers "${CSE2026_XLRON_LAYERS:-2}" \
  --transformer-num-heads "${CSE2026_XLRON_HEADS:-8}" \
  --transformer-position-dim "${CSE2026_XLRON_POSITION_DIM:-8}" \
  --xlron-architecture full \
  --xlron-enable-spectrum-branch true \
  --xlron-enable-base-relative-branch true \
  --xlron-enable-candidate-attention true \
  --xlron-candidate-transformer-layers "${CSE2026_XLRON_CANDIDATE_LAYERS:-2}" \
  --xlron-candidate-transformer-heads "${CSE2026_XLRON_CANDIDATE_HEADS:-4}" \
  --progress-every-batches "${CSE2026_XLRON_DISTILL_PROGRESS_BATCHES:-50}" \
  --print-every-episodes "${CSE2026_XLRON_DISTILL_PRINT_EPISODES:-4}"

cat <<EOF
Top32 XLRON full-DQN distill checkpoint:
  ${OUTPUT_DIR}/top32_xlron_full_dqn_distill_best.pt
EOF
