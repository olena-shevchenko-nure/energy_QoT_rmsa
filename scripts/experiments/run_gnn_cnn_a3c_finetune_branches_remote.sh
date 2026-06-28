#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BASE_CONFIG="configs/experiments/eon/remote_train_gnn_cnn_a3c_distill_eval_base.yaml"
RAW_A3C_CHECKPOINT="${CSE2026_RAW_GNN_CNN_A3C_CHECKPOINT:-runs/eon/eon_train_gnn_cnn_a3c_windowed_online_full/20260608_092054_unknown/gnn_cnn_a3c_windowed_online_best.pt}"
LIGHTGBM_OLD10_ARTIFACT="runs/eon/quick_runtime_artifacts/energy_ksp_top8_smoke/lightgbm_lightgbm_old10_tree_ranker.json"
DQN_ORATE60_ARTIFACT="runs/eon/quick_runtime_artifacts/dqn_distill_old10_override_targetband/torch_dqn_distill_old10_orate60_tree_ranker.json"
FULL_DQN_STRATIFIED32_E5="runs/eon/quick_runtime_artifacts/full_dqn_orate60_distill_frozen_mvp20_stratified32_e5/full_dqn_orate60_distill_frozen.pt"

HARD_CASE_WEIGHTS="bursty:high=1.6,bursty:overload=1.8,hotspot:high=1.6,hotspot:overload=1.8,nonuniform:high=1.6,nonuniform:overload=1.8,uniform:high=1.6,uniform:overload=1.8"

python3 scripts/experiments/train_gnn_cnn_a3c_distill.py \
  --config "$BASE_CONFIG" \
  --output-dir runs/eon/quick_runtime_artifacts/gnn_cnn_a3c_distill_full_dqn_stratified32_e5 \
  --initial-a3c-checkpoint "$RAW_A3C_CHECKPOINT" \
  --teacher-kind full_dqn \
  --teacher-dqn-checkpoint "$FULL_DQN_STRATIFIED32_E5" \
  --behavior-policy student_a3c \
  --rollout-policy full \
  --checkpoint-selection rollout_accepted \
  --train-max-episodes "${CSE2026_A3C_DISTILL_TRAIN_EPISODES:-32}" \
  --val-max-episodes "${CSE2026_A3C_DISTILL_VAL_EPISODES:-8}" \
  --episode-selection stratified \
  --epochs "${CSE2026_A3C_DISTILL_EPOCHS:-3}" \
  --batch-size 64 \
  --learning-rate 0.0001 \
  --ce-weight 1.0 \
  --listwise-kl-weight 0.5 \
  --listwise-temperature 2.0 \
  --hard-case-weight-rules "$HARD_CASE_WEIGHTS" \
  --rollout-val-max-episodes 8 \
  --print-every-episodes 2

python3 scripts/experiments/train_gnn_cnn_a3c_distill.py \
  --config "$BASE_CONFIG" \
  --output-dir runs/eon/quick_runtime_artifacts/gnn_cnn_a3c_distill_lightgbm_old10 \
  --initial-a3c-checkpoint "$RAW_A3C_CHECKPOINT" \
  --teacher-kind tree \
  --teacher-artifact "$LIGHTGBM_OLD10_ARTIFACT" \
  --teacher-selection-mode positive_advantage \
  --teacher-safety-guard emergency \
  --teacher-base-policy energy-aware-ksp-bm-ff \
  --behavior-policy student_a3c \
  --rollout-policy override \
  --checkpoint-selection rollout_accepted \
  --train-max-episodes "${CSE2026_A3C_DISTILL_TRAIN_EPISODES:-32}" \
  --val-max-episodes "${CSE2026_A3C_DISTILL_VAL_EPISODES:-8}" \
  --episode-selection stratified \
  --epochs "${CSE2026_A3C_DISTILL_EPOCHS:-3}" \
  --batch-size 64 \
  --learning-rate 0.0001 \
  --ce-weight 1.0 \
  --listwise-kl-weight 0.5 \
  --listwise-temperature 2.0 \
  --teacher-score-selected-boost 0.5 \
  --hard-case-weight-rules "$HARD_CASE_WEIGHTS" \
  --rollout-val-max-episodes 8 \
  --print-every-episodes 2

python3 scripts/experiments/train_gnn_cnn_a3c_distill.py \
  --config "$BASE_CONFIG" \
  --output-dir runs/eon/quick_runtime_artifacts/gnn_cnn_a3c_distill_dqn_orate60 \
  --initial-a3c-checkpoint "$RAW_A3C_CHECKPOINT" \
  --teacher-kind tree \
  --teacher-artifact "$DQN_ORATE60_ARTIFACT" \
  --teacher-selection-mode positive_advantage \
  --teacher-safety-guard emergency \
  --teacher-base-policy energy-aware-ksp-bm-ff \
  --behavior-policy student_a3c \
  --rollout-policy override \
  --checkpoint-selection rollout_accepted \
  --train-max-episodes "${CSE2026_A3C_DISTILL_TRAIN_EPISODES:-32}" \
  --val-max-episodes "${CSE2026_A3C_DISTILL_VAL_EPISODES:-8}" \
  --episode-selection stratified \
  --epochs "${CSE2026_A3C_DISTILL_EPOCHS:-3}" \
  --batch-size 64 \
  --learning-rate 0.0001 \
  --ce-weight 1.0 \
  --listwise-kl-weight 0.5 \
  --listwise-temperature 2.0 \
  --teacher-score-selected-boost 0.5 \
  --hard-case-weight-rules "$HARD_CASE_WEIGHTS" \
  --rollout-val-max-episodes 8 \
  --print-every-episodes 2

cat <<'EOF'
A3C fine-tune checkpoints:
  full-DQN: runs/eon/quick_runtime_artifacts/gnn_cnn_a3c_distill_full_dqn_stratified32_e5/gnn_cnn_a3c_distill_best.pt
  old10:    runs/eon/quick_runtime_artifacts/gnn_cnn_a3c_distill_lightgbm_old10/gnn_cnn_a3c_distill_best.pt
  orate60:  runs/eon/quick_runtime_artifacts/gnn_cnn_a3c_distill_dqn_orate60/gnn_cnn_a3c_distill_best.pt
EOF
