# Training Lineage Evidence

This document indexes the evidence kept in the public repository for how the
learned policies used in the paper were trained and selected.

The repository is primarily an evaluation/reproducibility snapshot for the final
MVP80 comparison. It includes final artifacts, selected training summaries,
training/export logs, historical configs, dataset manifests, and provenance
records. It does not include every intermediate run directory or every temporary
checkpoint from the private experiment workspace.

## Common Evidence

| Evidence | Location | Purpose |
|---|---|---|
| Final paper-to-runtime mapping | `README.md`, `docs/paper_model_mapping.md` | Maps paper names to runtime policy ids and artifact folders. |
| Model lineage | `artifacts/provenance/model_lineage.yaml` | Records source branch/commit, source run labels, selected checkpoints/exports, and artifact folders. |
| Artifact checksums | `artifacts/provenance/model_artifact_checksums.sha256` | Verifies model artifacts and copied training evidence. |
| Dataset manifest/checksums | `data/eon/generated/nsfnet_mvp_ong_expert_hybrid_mix/manifest.json`, `checksums.sha256` | Identifies the generated Top-N candidate dataset and train/val/test split sizes. |
| Historical training/evaluation configs | `configs/experiments/eon/` | Preserves configs from the experiment series, including DQN, LightGBM/XGBoost, XLRON, A3C, DAgger, and rollout branches. |
| Final MVP80 evaluation config | `configs/evaluation/mvp80_selected_topn_p95_compare_clean.yaml` | Clean path-adjusted config used by `scripts/reproduce_mvp80.py`. |

## Model-Specific Evidence

| Paper policy | Runtime id | Main training signal | Key evidence in public repo | Final selected artifact |
|---|---|---|---|---|
| Calibrated DQN-Override baseline ranker | `torch_dqn_candidate_ranker_distill_old10` | DQN-like MLP candidate ranker trained from old10/LightGBM-style supervised ranking targets, then exported with override-rate calibration. | `artifacts/models/calibrated_dqn_override/torch_dqn_override_rate_calibration_summary.json`; selected export JSONs contain `teacher_artifact` and `source_artifact`; model card in `docs/model_cards/calibrated_dqn_override.md`; lineage entry `calibrated_dqn_override`. | `artifacts/models/calibrated_dqn_override/torch_dqn_distill_old10_orate60_tree_ranker.json` and `torch_dqn_distill_ranker.pt`. |
| LightGBM-override ranker | `lightgbm_candidate_ranker_old10` | LightGBM LambdaRank-style candidate ranking over Top-N candidate features, exported as an override-style runtime model. | `artifacts/models/lightgbm_override_old10/lightgbm_quick_runtime_export_summary.json`; `logs/export_lightgbm.log`; model dumps under `artifacts/models/lightgbm_override_old10/`; model card in `docs/model_cards/lightgbm_override_old10.md`; lineage entry `lightgbm_override_old10`. | `artifacts/models/lightgbm_override_old10/lightgbm_lightgbm_old10_tree_ranker.json`. |
| GNN-CNN Full DQN Ranker | `gnn_cnn_dqn` | Full neural candidate ranker with graph/network, spectrum, request, and action/candidate encoders; trained as a distillation-style DQN candidate ranker from the selected orate60 teacher lineage. | `artifacts/models/full_dqn_stratified32_e5/training_summary.json`; `collection_summary.json`; model card in `docs/model_cards/full_dqn_stratified32_e5.md`; lineage entry `full_dqn_stratified32_e5`; related train/eval configs referencing `full_dqn_orate60_distill_frozen_mvp20_stratified32_e5`. | `artifacts/models/full_dqn_stratified32_e5/full_dqn_orate60_distill_frozen.pt`. |
| XLRON Counterfactual Ranker | `top32_xlron_stabilized_ppo` | XLRON-style Top-32 neural policy initialized from an earlier XLRON/full-DQN distillation checkpoint and fine-tuned with counterfactual listwise/pairwise ranking, reference KL, and rollout-based bucket-guard checkpoint selection. | `artifacts/models/xlron_cf_rank_g160_bucket_guard/top32_xlron_counterfactual_rank_finetune_summary.json`; `train.log`; model card in `docs/model_cards/xlron_cf_rank_g160_bucket_guard.md`; lineage entry `xlron_cf_rank_g160_bucket_guard`; scripts such as `scripts/experiments/run_top32_xlron_cf_rank_finetune_remote.sh`. | `artifacts/models/xlron_cf_rank_g160_bucket_guard/top32_xlron_counterfactual_rank_finetune_best.pt`. |
| A3C Policy Distilled from Full DQN | `gnn_cnn_a3c` | GNN-CNN actor-critic student trained from online student-state data with Full DQN as teacher; checkpoint selected by rollout validation accepted count. | `artifacts/models/a3c_distill_full_dqn/training_summary.json`; `collection_summary.json`; model card in `docs/model_cards/a3c_distill_full_dqn.md`; lineage entry `a3c_distill_full_dqn`; scripts such as `scripts/experiments/run_gnn_cnn_a3c_finetune_branches_remote.sh`. | `artifacts/models/a3c_distill_full_dqn/gnn_cnn_a3c_distill_best.pt`. |

## Heuristic Policies

The deterministic baselines are not trained. Their evidence is the runtime
implementation and evaluation configuration:

- `energy-aware-ksp-bm-ff`
- `ksp-ff`
- `ksp-bm-ff`

They are included in `configs/evaluation/mvp80_selected_topn_p95_compare_clean.yaml`
and evaluated by the same `scripts/reproduce_mvp80.py` pipeline as the learned
policies.

## What Can Be Reproduced Directly

The public repository is set up to reproduce the final MVP80 evaluation table:

```bash
python scripts/reproduce_mvp80.py --dry-run
python scripts/reproduce_mvp80.py
```

The script automatically prepares the pinned Optical Networking Gym source
checkout from `third_party/optical-networking-gym.lock` when needed.

## What Is Preserved as Evidence, Not a One-Command Retrain

The repository preserves training evidence and historical configs, but it does
not provide a single normalized `retrain_all_models.py` entry point. Some
training configs are historical remote-run configs and still reflect the
experiment series rather than a clean local retraining workflow.

For reviewer inspection, the strongest evidence chain is:

1. `artifacts/provenance/model_lineage.yaml`
2. model-specific summaries/logs under `artifacts/models/<model>/`
3. model cards under `docs/model_cards/`
4. historical configs and scripts under `configs/experiments/eon/` and `scripts/experiments/`
5. final MVP80 evaluation through `scripts/reproduce_mvp80.py`
