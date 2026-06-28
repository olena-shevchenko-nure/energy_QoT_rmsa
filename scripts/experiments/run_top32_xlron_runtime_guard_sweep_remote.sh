#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN_TAG="${CSE2026_XLRON_CF_RANK_RUN_TAG:-g160_v1}"
BASE_DIR="${CSE2026_XLRON_CF_BASE_DIR:-runs/eon/quick_runtime_artifacts}"
INPUT_DIR="${CSE2026_XLRON_CF_RANK_INPUT_DIR:-$BASE_DIR/top32_xlron_cf_online_signal_stable_h100_h50_${RUN_TAG}_calibrated_protected_v1}"
OUTPUT_BASE="${CSE2026_XLRON_RUNTIME_GUARD_OUTPUT_BASE:-$BASE_DIR/top32_xlron_runtime_guard_${RUN_TAG}_one_slice}"
PROTECTED_BUCKETS="${CSE2026_XLRON_RUNTIME_GUARD_BUCKETS:-bursty:high,bursty:medium,bursty:overload,hotspot:high,hotspot:medium,nonuniform:high,nonuniform:medium}"
MARGINS_CSV="${CSE2026_XLRON_RUNTIME_GUARD_MARGINS:-0.0,0.05,0.10,0.20,0.35,0.50}"

IFS=',' read -r -a MARGINS <<< "$MARGINS_CSV"
for raw_margin in "${MARGINS[@]}"; do
  margin="$(echo "$raw_margin" | xargs)"
  if [[ -z "$margin" ]]; then
    continue
  fi
  margin_tag="$(python3 - <<PY
margin = float("$margin")
print(str(margin).replace("-", "m").replace(".", "p"))
PY
)"
  export CSE2026_XLRON_CF_RANK_INPUT_DIR="$INPUT_DIR"
  export CSE2026_XLRON_CF_RANK_OUTPUT_DIR="${OUTPUT_BASE}_m${margin_tag}"
  export CSE2026_XLRON_CF_RANK_EPOCHS=0
  export CSE2026_XLRON_CF_RANK_BATCH_SIZE="${CSE2026_XLRON_CF_RANK_BATCH_SIZE:-32}"
  export CSE2026_XLRON_CF_RANK_CHECKPOINT_SELECTION="${CSE2026_XLRON_CF_RANK_CHECKPOINT_SELECTION:-rollout_bucket_guard_score}"
  export CSE2026_XLRON_CF_RANK_ROLLOUT_VAL_MAX_EPISODES="${CSE2026_XLRON_CF_RANK_ROLLOUT_VAL_MAX_EPISODES:-16}"
  export CSE2026_XLRON_CF_RANK_ROLLOUT_VAL_SLICES="${CSE2026_XLRON_CF_RANK_ROLLOUT_VAL_SLICES:-1}"
  export CSE2026_XLRON_CF_RANK_ROLLOUT_VAL_SLICE_STRIDE="${CSE2026_XLRON_CF_RANK_ROLLOUT_VAL_SLICE_STRIDE:-1}"
  export CSE2026_XLRON_CF_RANK_ROLLOUT_NEGATIVE_BUCKET_PENALTY="${CSE2026_XLRON_CF_RANK_ROLLOUT_NEGATIVE_BUCKET_PENALTY:-0.5}"
  export CSE2026_XLRON_CF_RANK_ROLLOUT_PROTECTED_BUCKETS="${CSE2026_XLRON_CF_RANK_ROLLOUT_PROTECTED_BUCKETS:-$PROTECTED_BUCKETS}"
  export CSE2026_XLRON_CF_RANK_ROLLOUT_PROTECTED_BUCKET_PENALTY="${CSE2026_XLRON_CF_RANK_ROLLOUT_PROTECTED_BUCKET_PENALTY:-8.0}"
  export CSE2026_XLRON_CF_RANK_RUNTIME_GUARD_BUCKETS="$PROTECTED_BUCKETS"
  export CSE2026_XLRON_CF_RANK_RUNTIME_GUARD_MIN_MARGIN="$margin"
  export CSE2026_XLRON_CF_RANK_RUNTIME_GUARD_BASE_INDEX="${CSE2026_XLRON_CF_RANK_RUNTIME_GUARD_BASE_INDEX:-0}"
  bash scripts/experiments/run_top32_xlron_cf_rank_finetune_remote.sh
done

python3 - <<PY
import json
from pathlib import Path

base = Path("$OUTPUT_BASE")
margins = [item.strip() for item in "$MARGINS_CSV".split(",") if item.strip()]
rows = []
for raw in margins:
    margin = float(raw)
    tag = str(margin).replace("-", "m").replace(".", "p")
    path = Path(f"{base}_m{tag}") / "top32_xlron_counterfactual_rank_finetune_summary.json"
    if not path.exists():
        continue
    summary = json.loads(path.read_text())
    rollout = summary.get("best_rollout_val_eval") or summary.get("initial_rollout_val_eval") or {}
    guard = rollout.get("runtime_guard") or {}
    rows.append({
        "margin": margin,
        "output_dir": str(path.parent),
        "best_epoch": int(summary.get("best_epoch", 0)),
        "best_score": float(summary.get("best_score", 0.0)),
        "accepted": int(rollout.get("accepted", 0)),
        "blocking_rate": float(rollout.get("blocking_rate", 0.0)),
        "guard_requests": int(guard.get("requests", 0)),
        "guard_nonbase_candidates": int(guard.get("nonbase_candidates", 0)),
        "guard_fallbacks": int(guard.get("fallbacks", 0)),
        "guard_fallback_rate": float(guard.get("fallback_rate", 0.0)),
        "guard_by_bucket": guard.get("by_bucket", {}),
    })
out = Path(f"{base}_summary.json")
out.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n")
print(out)
print(json.dumps({"rows": rows}, indent=2, sort_keys=True))
PY
