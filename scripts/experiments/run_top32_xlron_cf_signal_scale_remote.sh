#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="${CSE2026_XLRON_CF_CONFIG:-configs/experiments/eon/remote_collect_online_dqn_base_alltopn_h100_train_stratified.yaml}"
CHECKPOINT="${CSE2026_XLRON_CF_CHECKPOINT:-runs/eon/quick_runtime_artifacts/top32_xlron_dagger_student32_16_preserve16_e4_lr6e5/top32_xlron_full_dqn_distill_best.pt}"
TEACHER_DQN_CHECKPOINT="${CSE2026_XLRON_CF_TEACHER_DQN_CHECKPOINT:-runs/eon/quick_runtime_artifacts/full_dqn_orate60_distill_frozen_mvp20_stratified32_e5/full_dqn_orate60_distill_frozen.pt}"
RUN_TAG="${CSE2026_XLRON_CF_RUN_TAG:-g160_v1}"
BASE_DIR="${CSE2026_XLRON_CF_BASE_DIR:-runs/eon/quick_runtime_artifacts}"

MAX_GROUPS="${CSE2026_XLRON_CF_MAX_GROUPS:-160}"
COLLECTION_STRIDE="${CSE2026_XLRON_CF_STRIDE:-20}"
GROUPS_PER_EPISODE="${CSE2026_XLRON_CF_GROUPS_PER_EPISODE:-8}"
BUCKETS="${CSE2026_XLRON_CF_BUCKETS:-hotspot:high,nonuniform:medium,uniform:low,uniform:high,uniform:overload,bursty:high}"

H100_DIR="${CSE2026_XLRON_CF_H100_DIR:-$BASE_DIR/top32_xlron_cf_online_signal_h100_${RUN_TAG}}"
H50_DIR="${CSE2026_XLRON_CF_H50_DIR:-$BASE_DIR/top32_xlron_cf_online_signal_h50_${RUN_TAG}}"
STABLE_DIR="${CSE2026_XLRON_CF_STABLE_DIR:-$BASE_DIR/top32_xlron_cf_online_signal_stable_h100_h50_${RUN_TAG}}"
AUDIT_DIR="${CSE2026_XLRON_CF_AUDIT_DIR:-$BASE_DIR/top32_xlron_cf_signal_audit_${RUN_TAG}}"

collect() {
  local horizon="$1"
  local output_dir="$2"
  mkdir -p "$output_dir"
  python3 scripts/experiments/collect_online_base_topn_counterfactuals.py \
    --config "$CONFIG" \
    --output-dir "$output_dir" \
    --base-policy "${CSE2026_XLRON_CF_BASE_POLICY:-top32_xlron}" \
    --rollout-policy "${CSE2026_XLRON_CF_ROLLOUT_POLICY:-top32_xlron}" \
    --top32-xlron-checkpoint "$CHECKPOINT" \
    --candidate-pool "${CSE2026_XLRON_CF_CANDIDATE_POOL:-all_topn}" \
    --top-k "${CSE2026_XLRON_CF_TOP_K:-32}" \
    --lookahead-horizon "$horizon" \
    --episode-selection "${CSE2026_XLRON_CF_EPISODE_SELECTION:-stratified}" \
    --include-buckets "$BUCKETS" \
    --collection-stride "$COLLECTION_STRIDE" \
    --max-groups-per-episode "$GROUPS_PER_EPISODE" \
    --max-collected-groups "$MAX_GROUPS" \
    --progress-every "${CSE2026_XLRON_CF_PROGRESS_EVERY:-20}" \
    > "$output_dir/collector.log" 2>&1
}

collect 100 "$H100_DIR"
collect 50 "$H50_DIR"

mkdir -p "$STABLE_DIR"
python3 scripts/experiments/build_stable_online_counterfactual_dataset.py \
  --primary-dir "$H100_DIR" \
  --confirmation-dir "$H50_DIR" \
  --output-dir "$STABLE_DIR" \
  --min-win-delta "${CSE2026_XLRON_CF_MIN_WIN_DELTA:-1.0}" \
  --min-secondary-abs "${CSE2026_XLRON_CF_MIN_SECONDARY_ABS:-0.05}" \
  > "$STABLE_DIR/build_stable.log" 2>&1

mkdir -p "$AUDIT_DIR"
python3 scripts/experiments/audit_top32_counterfactual_signal.py \
  --config "$CONFIG" \
  --input-dir "$H100_DIR" \
  --output-dir "$AUDIT_DIR" \
  --teacher-dqn-checkpoint "$TEACHER_DQN_CHECKPOINT" \
  --batch-size "${CSE2026_XLRON_CF_AUDIT_BATCH_SIZE:-16}" \
  --label raw_h100 \
  > "$AUDIT_DIR/raw_h100_audit.log" 2>&1

python3 scripts/experiments/audit_top32_counterfactual_signal.py \
  --config "$CONFIG" \
  --input-dir "$STABLE_DIR" \
  --output-dir "$AUDIT_DIR" \
  --teacher-dqn-checkpoint "$TEACHER_DQN_CHECKPOINT" \
  --batch-size "${CSE2026_XLRON_CF_AUDIT_BATCH_SIZE:-16}" \
  --label stable_h100_h50 \
  > "$AUDIT_DIR/stable_h100_h50_audit.log" 2>&1

python3 - <<PY
import json
from pathlib import Path

h100 = Path("$H100_DIR")
h50 = Path("$H50_DIR")
stable = Path("$STABLE_DIR")
audit = Path("$AUDIT_DIR")
summary = {
    "run_tag": "$RUN_TAG",
    "h100_dir": str(h100),
    "h50_dir": str(h50),
    "stable_dir": str(stable),
    "audit_dir": str(audit),
    "max_groups": int("$MAX_GROUPS"),
    "collection_stride": int("$COLLECTION_STRIDE"),
    "groups_per_episode": int("$GROUPS_PER_EPISODE"),
    "buckets": "$BUCKETS",
}
for label, path in [
    ("h100_collection", h100 / "online_base_topn_summary.json"),
    ("h50_collection", h50 / "online_base_topn_summary.json"),
    ("stable_dataset", stable / "stable_online_counterfactual_summary.json"),
    ("raw_signal", audit / "raw_h100_signal_summary.json"),
    ("stable_signal", audit / "stable_h100_h50_signal_summary.json"),
]:
    if path.exists():
        summary[label] = json.load(open(path))
out = Path("$BASE_DIR") / "top32_xlron_cf_signal_scale_${RUN_TAG}_summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(out)
PY
