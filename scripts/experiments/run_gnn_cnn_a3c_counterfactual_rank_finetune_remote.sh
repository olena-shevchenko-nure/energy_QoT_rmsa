#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BASE_CONFIG="${CSE2026_A3C_CF_CONFIG:-configs/experiments/eon/remote_train_gnn_cnn_a3c_distill_eval_base.yaml}"
INPUT_DIR="${CSE2026_A3C_CF_INPUT_DIR:-runs/eon/quick_runtime_artifacts/online_dqn_base_topn_all_h100_train_g1600_e80_s20}"
INITIAL_CHECKPOINT="${CSE2026_A3C_CF_INITIAL_CHECKPOINT:-runs/eon/quick_runtime_artifacts/gnn_cnn_a3c_distill_full_dqn_stratified32_e5/gnn_cnn_a3c_distill_best.pt}"
REFERENCE_CHECKPOINT="${CSE2026_A3C_CF_REFERENCE_CHECKPOINT:-$INITIAL_CHECKPOINT}"
OUTPUT_DIR="${CSE2026_A3C_CF_OUTPUT_DIR:-runs/eon/quick_runtime_artifacts/gnn_cnn_a3c_cf_rank_full_dqn_h100_g1325_smoke}"

python3 scripts/experiments/train_gnn_cnn_a3c_counterfactual_rank_finetune.py \
  --config "$BASE_CONFIG" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --initial-checkpoint "$INITIAL_CHECKPOINT" \
  --reference-checkpoint "$REFERENCE_CHECKPOINT" \
  --epochs "${CSE2026_A3C_CF_EPOCHS:-3}" \
  --batch-size "${CSE2026_A3C_CF_BATCH_SIZE:-64}" \
  --learning-rate "${CSE2026_A3C_CF_LR:-0.00002}" \
  --weight-decay "${CSE2026_A3C_CF_WEIGHT_DECAY:-0.0001}" \
  --ce-weight "${CSE2026_A3C_CF_CE_WEIGHT:-0.30}" \
  --listwise-weight "${CSE2026_A3C_CF_LISTWISE_WEIGHT:-0.80}" \
  --pairwise-weight "${CSE2026_A3C_CF_PAIRWISE_WEIGHT:-0.50}" \
  --base-pairwise-weight "${CSE2026_A3C_CF_BASE_PAIRWISE_WEIGHT:-0.50}" \
  --regression-weight "${CSE2026_A3C_CF_REGRESSION_WEIGHT:-0.05}" \
  --reference-kl-weight "${CSE2026_A3C_CF_REFERENCE_KL_WEIGHT:-1.00}" \
  --preservation-weight "${CSE2026_A3C_CF_PRESERVATION_WEIGHT:-0.0}" \
  --preservation-temperature "${CSE2026_A3C_CF_PRESERVATION_TEMP:-2.0}" \
  --preservation-split "${CSE2026_A3C_CF_PRESERVATION_SPLIT:-train}" \
  --preservation-max-episodes "${CSE2026_A3C_CF_PRESERVATION_EPISODES:-0}" \
  --preservation-max-requests-per-episode "${CSE2026_A3C_CF_PRESERVATION_REQUESTS:-0}" \
  --preservation-episode-selection "${CSE2026_A3C_CF_PRESERVATION_EPISODE_SELECTION:-stratified}" \
  --preservation-max-examples "${CSE2026_A3C_CF_PRESERVATION_MAX_EXAMPLES:-2048}" \
  --preservation-batch-size "${CSE2026_A3C_CF_PRESERVATION_BATCH_SIZE:-0}" \
  --preservation-print-every-episodes "${CSE2026_A3C_CF_PRESERVATION_PRINT_EPISODES:-4}" \
  --preservation-weight-rules "${CSE2026_A3C_CF_PRESERVATION_WEIGHT_RULES:-}" \
  --target-temperature "${CSE2026_A3C_CF_TARGET_TEMP:-1.0}" \
  --student-temperature "${CSE2026_A3C_CF_STUDENT_TEMP:-1.0}" \
  --reference-temperature "${CSE2026_A3C_CF_REFERENCE_TEMP:-2.0}" \
  --pairwise-margin "${CSE2026_A3C_CF_PAIRWISE_MARGIN:-0.15}" \
  --order-epsilon "${CSE2026_A3C_CF_ORDER_EPSILON:-0.05}" \
  --target-scale "${CSE2026_A3C_CF_TARGET_SCALE:-4.0}" \
  --grad-clip-norm "${CSE2026_A3C_CF_GRAD_CLIP:-1.0}" \
  --checkpoint-selection "${CSE2026_A3C_CF_CHECKPOINT_SELECTION:-rollout_accepted}" \
  --rollout-policy full \
  --rollout-val-max-episodes "${CSE2026_A3C_CF_ROLLOUT_VAL_EPISODES:-8}" \
  --rollout-val-episode-selection stratified \
  --progress-every-batches "${CSE2026_A3C_CF_PROGRESS_BATCHES:-10}"

cat <<EOF
A3C counterfactual rank checkpoint:
  ${OUTPUT_DIR}/gnn_cnn_a3c_counterfactual_rank_finetune.pt
EOF
