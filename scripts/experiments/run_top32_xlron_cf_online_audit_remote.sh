#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="${CSE2026_XLRON_CF_CONFIG:-configs/experiments/eon/remote_collect_online_dqn_base_alltopn_h100_train_stratified.yaml}"
CHECKPOINT="${CSE2026_XLRON_CF_CHECKPOINT:-runs/eon/quick_runtime_artifacts/top32_xlron_dagger_student32_16_preserve16_e4_lr6e5/top32_xlron_full_dqn_distill_best.pt}"
OUTPUT_DIR="${CSE2026_XLRON_CF_OUTPUT_DIR:-runs/eon/quick_runtime_artifacts/top32_xlron_cf_online_hard_buckets_h20_g160}"

python3 scripts/experiments/collect_online_base_topn_counterfactuals.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --base-policy "${CSE2026_XLRON_CF_BASE_POLICY:-top32_xlron}" \
  --rollout-policy "${CSE2026_XLRON_CF_ROLLOUT_POLICY:-top32_xlron}" \
  --top32-xlron-checkpoint "$CHECKPOINT" \
  --candidate-pool "${CSE2026_XLRON_CF_CANDIDATE_POOL:-all_topn}" \
  --top-k "${CSE2026_XLRON_CF_TOP_K:-32}" \
  --lookahead-horizon "${CSE2026_XLRON_CF_HORIZON:-20}" \
  --episode-selection "${CSE2026_XLRON_CF_EPISODE_SELECTION:-stratified}" \
  --include-buckets "${CSE2026_XLRON_CF_BUCKETS:-hotspot:high,nonuniform:medium,uniform:low,uniform:high,uniform:overload,bursty:high}" \
  --collection-stride "${CSE2026_XLRON_CF_STRIDE:-20}" \
  --max-groups-per-episode "${CSE2026_XLRON_CF_GROUPS_PER_EPISODE:-8}" \
  --max-collected-groups "${CSE2026_XLRON_CF_MAX_GROUPS:-160}" \
  --progress-every "${CSE2026_XLRON_CF_PROGRESS_EVERY:-20}"
