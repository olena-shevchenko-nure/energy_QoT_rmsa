# Energy/QoT-Aware RMSA Reproducibility Repository

This repository contains the reproducibility snapshot for the CSE 2026 RMSA experiments reported in the paper.
It includes the final model artifacts, the generated NSFNET MVP dataset, the Optical Networking Gym evaluation configuration, source code, and raw result/statistical tables used for the paper comparison.

## Paper Policy Mapping

| Paper name | Repository artifact / policy id | Runtime type |
|---|---|---|
| Calibrated DQN-Override baseline ranker | `artifacts/models/calibrated_dqn_override`, `torch_dqn_candidate_ranker_distill_old10` | calibrated DQN-like override ranker |
| LightGBM-override ranker | `artifacts/models/lightgbm_override_old10`, `lightgbm_candidate_ranker_old10` | LightGBM LambdaRank-style override ranker |
| GNN-CNN Full DQN Ranker | `artifacts/models/full_dqn_stratified32_e5`, `gnn_cnn_dqn` | full neural GNN/CNN DQN candidate ranker |
| XLRON Counterfactual Ranker | `artifacts/models/xlron_cf_rank_g160_bucket_guard`, `top32_xlron_stabilized_ppo` | XLRON-style neural/counterfactual candidate ranker |
| A3C Policy Distilled from Full DQN | `artifacts/models/a3c_distill_full_dqn`, `gnn_cnn_a3c` | GNN-CNN actor-critic policy distilled from Full DQN |
| Energy-Aware-KSP-BM-FF | source policy `energy-aware-ksp-bm-ff` | deterministic heuristic |
| KSP-FF | source policy `ksp-ff` | deterministic heuristic |
| KSP-BM-FF | source policy `ksp-bm-ff` | deterministic heuristic |

## Included Snapshot

- Final model artifacts for the five learned policies from the paper.
- Generated dataset: `data/eon/generated/nsfnet_mvp_ong_expert_hybrid_mix`.
- NSFNET topology and modulation data used by the experiments.
- Source files under `src/cse2026`, including data generation, policy runtime, and experiment code.
- Evaluation config: `configs/experiments/eon/remote_ong_rollout_mvp80_selected_topn_p95_compare.yaml`.
- Paper result tables and statistical analysis files under `results/mvp80` and `docs/experiments/raw`.
- Provenance and checksums under `artifacts/provenance`.

## Main Reproduction Entry Points

1. Install the Python dependencies from `requirements-data.txt` and `requirements-experiments.txt`.
2. Install or clone Optical Networking Gym as described in `docs/optical_networking_gym_setup.md`.
3. Run the MVP80 comparison using the clean local config `configs/evaluation/mvp80_selected_topn_p95_compare_clean.yaml`.
4. Compare the generated output with `results/mvp80/tables` and `results/mvp80/statistics`.

The exact execution command depends on the local Optical Networking Gym checkout path and GPU availability. See `docs/experiment_runbook.md`.

## Current Paper Results

The table used in the paper is available at:

- `results/mvp80/tables/mvp80_selected_topn_p95_comparison_20260626.csv`
- `docs/experiments/raw/mvp80_selected_topn_p95_results_table_reward_per_request_20260626.docx`

The per-episode statistical tests and scenario breakdowns are available at:

- `results/mvp80/statistics/mvp80_statistical_summary_20260626.csv`
- `results/mvp80/statistics/mvp80_paired_tests_vs_energy_aware_20260626.csv`
- `results/mvp80/statistics/mvp80_per_scenario_breakdown_20260626.csv`

## Provenance

This clean repository was assembled from the source project branch `energy_aware_ksp_bm_ff_based` at source commit `5d6fff1b26c9141c1e601d2404fbb82885459618`.
The final artifacts were copied from the remote experiment tree documented in `artifacts/provenance/model_lineage.yaml`.
