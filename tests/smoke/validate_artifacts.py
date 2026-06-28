"""Validate that required paper artifacts are present in the snapshot."""

from __future__ import annotations

from pathlib import Path


REQUIRED_FILES = [
    "artifacts/models/calibrated_dqn_override/torch_dqn_distill_old10_orate60_tree_ranker.json",
    "artifacts/models/calibrated_dqn_override/torch_dqn_distill_ranker.pt",
    "artifacts/models/calibrated_dqn_override/torch_dqn_distill_advantage_win.pt",
    "artifacts/models/calibrated_dqn_override/torch_dqn_distill_advantage_loss.pt",
    "artifacts/models/calibrated_dqn_override/torch_dqn_distill_advantage_delta.pt",
    "artifacts/models/lightgbm_override_old10/lightgbm_lightgbm_old10_tree_ranker.json",
    "artifacts/models/full_dqn_stratified32_e5/full_dqn_orate60_distill_frozen.pt",
    "artifacts/models/xlron_cf_rank_g160_bucket_guard/top32_xlron_counterfactual_rank_finetune_best.pt",
    "artifacts/models/a3c_distill_full_dqn/gnn_cnn_a3c_distill_best.pt",
    "data/eon/generated/nsfnet_mvp_ong_expert_hybrid_mix/manifest.json",
    "data/eon/generated/nsfnet_mvp_ong_expert_hybrid_mix/checksums.sha256",
    "results/mvp80/raw/mvp80_selected_topn_p95_policy_episode_metrics_20260626.csv",
    "results/mvp80/tables/mvp80_selected_topn_p95_comparison_20260626.csv",
]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    missing = [path for path in REQUIRED_FILES if not (root / path).is_file()]
    if missing:
        for path in missing:
            print(f"missing: {path}")
        raise SystemExit(1)
    print(f"validated {len(REQUIRED_FILES)} required files")


if __name__ == "__main__":
    main()

