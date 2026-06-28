#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BASE_CONFIG="${CSE2026_A3C_DISTILL_CONFIG:-configs/experiments/eon/remote_train_gnn_cnn_a3c_distill_eval_base.yaml}"
RAW_A3C_CHECKPOINT="${CSE2026_RAW_GNN_CNN_A3C_CHECKPOINT:-runs/eon/eon_train_gnn_cnn_a3c_windowed_online_full/20260608_092054_unknown/gnn_cnn_a3c_windowed_online_best.pt}"
FULL_DQN_STRATIFIED32_E5="${CSE2026_FULL_DQN_STRATIFIED32_E5_CHECKPOINT:-runs/eon/quick_runtime_artifacts/full_dqn_orate60_distill_frozen_mvp20_stratified32_e5/full_dqn_orate60_distill_frozen.pt}"

RUN_NAME="${CSE2026_A3C_FULL_DQN_LISTWISE_NAME:-gnn_cnn_a3c_full_dqn_transplant_listwise_32_8}"
INIT_DIR="runs/eon/quick_runtime_artifacts/${RUN_NAME}_init"
OUTPUT_DIR="runs/eon/quick_runtime_artifacts/${RUN_NAME}"
INIT_CHECKPOINT="${INIT_DIR}/gnn_cnn_a3c_from_full_dqn_stratified32_e5.pt"

HARD_CASE_WEIGHTS="${CSE2026_A3C_HARD_CASE_WEIGHTS:-hotspot:medium=1.5,hotspot:overload=1.8,nonuniform:medium=1.5,nonuniform:overload=1.8,uniform:medium=1.5,uniform:high=1.6,uniform:overload=1.8}"

python3 scripts/experiments/init_gnn_cnn_a3c_from_dqn.py \
  --config "$BASE_CONFIG" \
  --initial-a3c-checkpoint "$RAW_A3C_CHECKPOINT" \
  --initial-dqn-checkpoint "$FULL_DQN_STRATIFIED32_E5" \
  --output "$INIT_CHECKPOINT"

python3 scripts/experiments/train_gnn_cnn_a3c_distill.py \
  --config "$BASE_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --initial-a3c-checkpoint "$INIT_CHECKPOINT" \
  --teacher-kind full_dqn \
  --teacher-dqn-checkpoint "$FULL_DQN_STRATIFIED32_E5" \
  --behavior-policy student_a3c \
  --rollout-policy full \
  --checkpoint-selection rollout_accepted \
  --train-max-episodes "${CSE2026_A3C_DISTILL_TRAIN_EPISODES:-32}" \
  --val-max-episodes "${CSE2026_A3C_DISTILL_VAL_EPISODES:-8}" \
  --episode-selection stratified \
  --max-requests-per-episode "${CSE2026_A3C_DISTILL_MAX_REQUESTS:-0}" \
  --epochs "${CSE2026_A3C_DISTILL_EPOCHS:-4}" \
  --batch-size "${CSE2026_A3C_DISTILL_BATCH_SIZE:-64}" \
  --learning-rate "${CSE2026_A3C_DISTILL_LR:-0.00005}" \
  --weight-decay "${CSE2026_A3C_DISTILL_WEIGHT_DECAY:-0.0001}" \
  --grad-clip-norm "${CSE2026_A3C_DISTILL_GRAD_CLIP:-1.0}" \
  --ce-weight "${CSE2026_A3C_DISTILL_CE_WEIGHT:-0.5}" \
  --listwise-kl-weight "${CSE2026_A3C_DISTILL_KL_WEIGHT:-1.0}" \
  --listwise-temperature "${CSE2026_A3C_DISTILL_KL_TEMP:-3.0}" \
  --pairwise-rank-weight "${CSE2026_A3C_DISTILL_PAIRWISE_WEIGHT:-0.2}" \
  --pairwise-temperature "${CSE2026_A3C_DISTILL_PAIRWISE_TEMP:-1.0}" \
  --pairwise-teacher-gap "${CSE2026_A3C_DISTILL_PAIRWISE_GAP:-0.0}" \
  --hard-case-weight-rules "$HARD_CASE_WEIGHTS" \
  --rollout-val-max-episodes "${CSE2026_A3C_ROLLOUT_VAL_EPISODES:-8}" \
  --rollout-val-episode-selection stratified \
  --progress-every-batches "${CSE2026_A3C_DISTILL_PROGRESS_BATCHES:-50}" \
  --print-every-episodes "${CSE2026_A3C_DISTILL_PRINT_EPISODES:-4}"

cat <<EOF
A3C full-DQN transplant listwise checkpoint:
  ${OUTPUT_DIR}/gnn_cnn_a3c_distill_best.pt
EOF
