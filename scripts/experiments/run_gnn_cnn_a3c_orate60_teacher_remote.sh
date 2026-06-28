#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BASE_CONFIG="configs/experiments/eon/remote_train_gnn_cnn_a3c_distill_eval_base.yaml"
RAW_A3C_CHECKPOINT="${CSE2026_RAW_GNN_CNN_A3C_CHECKPOINT:-runs/eon/eon_train_gnn_cnn_a3c_windowed_online_full/20260608_092054_unknown/gnn_cnn_a3c_windowed_online_best.pt}"
DQN_TRANSPLANT_CHECKPOINT="${CSE2026_A3C_TRANSPLANT_DQN_CHECKPOINT:-runs/eon/quick_runtime_artifacts/full_dqn_orate60_distill_frozen_mvp20_stratified32_e5/full_dqn_orate60_distill_frozen.pt}"
DQN_ORATE60_ARTIFACT="${CSE2026_DQN_ORATE60_ARTIFACT:-runs/eon/quick_runtime_artifacts/dqn_distill_old10_override_targetband/torch_dqn_distill_old10_orate60_tree_ranker.json}"

INIT_DIR="runs/eon/quick_runtime_artifacts/gnn_cnn_a3c_orate60_teacher_transplant_init"
INIT_CHECKPOINT="$INIT_DIR/gnn_cnn_a3c_from_dqn_orate60.pt"
TRAIN_DIR="runs/eon/quick_runtime_artifacts/gnn_cnn_a3c_orate60_teacher_full"

HARD_CASE_WEIGHTS="bursty:high=1.6,bursty:overload=1.8,hotspot:high=1.6,hotspot:overload=1.8,nonuniform:high=1.6,nonuniform:overload=1.8,uniform:high=1.6,uniform:overload=1.8"

python3 scripts/experiments/init_gnn_cnn_a3c_from_dqn.py \
  --config "$BASE_CONFIG" \
  --initial-a3c-checkpoint "$RAW_A3C_CHECKPOINT" \
  --initial-dqn-checkpoint "$DQN_TRANSPLANT_CHECKPOINT" \
  --output "$INIT_CHECKPOINT"

python3 scripts/experiments/train_gnn_cnn_a3c_distill.py \
  --config "$BASE_CONFIG" \
  --output-dir "$TRAIN_DIR" \
  --initial-a3c-checkpoint "$INIT_CHECKPOINT" \
  --teacher-kind tree \
  --teacher-artifact "$DQN_ORATE60_ARTIFACT" \
  --teacher-selection-mode positive_advantage \
  --teacher-safety-guard emergency \
  --teacher-base-policy energy-aware-ksp-bm-ff \
  --behavior-policy student_a3c \
  --rollout-policy full \
  --checkpoint-selection rollout_accepted \
  --train-max-episodes "${CSE2026_A3C_ORATE60_TRAIN_EPISODES:-64}" \
  --val-max-episodes "${CSE2026_A3C_ORATE60_VAL_EPISODES:-16}" \
  --episode-selection stratified \
  --epochs "${CSE2026_A3C_ORATE60_EPOCHS:-5}" \
  --batch-size 64 \
  --learning-rate "${CSE2026_A3C_ORATE60_LR:-0.00005}" \
  --ce-weight 1.0 \
  --listwise-kl-weight "${CSE2026_A3C_ORATE60_KL_WEIGHT:-0.25}" \
  --listwise-temperature 2.0 \
  --teacher-score-selected-boost 0.5 \
  --hard-case-weight-rules "$HARD_CASE_WEIGHTS" \
  --rollout-val-max-episodes 16 \
  --print-every-episodes 4

cat <<EOF
A3C orate60-teacher full-policy checkpoints:
  init:  $INIT_CHECKPOINT
  train: $TRAIN_DIR/gnn_cnn_a3c_distill_best.pt
EOF
