# Experiment Runbook

## MVP80 Evaluation

The original paper comparison config is preserved at:

```bash
configs/experiments/eon/remote_ong_rollout_mvp80_selected_topn_p95_compare.yaml
```

For this clean repository, use the path-adjusted config:

```bash
configs/evaluation/mvp80_selected_topn_p95_compare_clean.yaml
```

The reviewer-facing entry point is:

```bash
python scripts/reproduce_mvp80.py --dry-run
python scripts/reproduce_mvp80.py
```

For a local smoke check before the full 80-episode rollout:

```bash
python scripts/reproduce_mvp80.py --max-episodes 1 --max-requests-per-episode 200
```

If Optical Networking Gym is installed elsewhere, use:

```bash
python scripts/reproduce_mvp80.py --ong-source-path /path/to/optical-networking-gym
```

The configuration evaluates these policies on the same NSFNET MVP80 test split:

- `torch_dqn_candidate_ranker_distill_old10`
- `lightgbm_candidate_ranker_old10`
- `gnn_cnn_dqn`
- `top32_xlron_stabilized_ppo`
- `gnn_cnn_a3c`
- `energy-aware-ksp-bm-ff`
- `ksp-ff`
- `ksp-bm-ff`

The paper reports Accepted, Blocking, reward per request, Energy, fragmentation, QoT, mean latency, P95 latency, average selected Top-N index, and P95 selected Top-N index.

## Expected Data

The main dataset is:

```text
data/eon/generated/nsfnet_mvp_ong_expert_hybrid_mix
```

It contains the train/validation/test traffic, candidate, CNN, GNN, and DQN feature files used by the training and evaluation pipeline.

## Expected Results

The expected MVP80 summary table is:

```text
results/mvp80/tables/mvp80_selected_topn_p95_comparison_20260626.csv
```

The expected per-episode metrics are:

```text
results/mvp80/raw/mvp80_selected_topn_p95_policy_episode_metrics_20260626.csv
```

The statistical tests versus Energy-Aware-KSP-BM-FF are:

```text
results/mvp80/statistics/mvp80_paired_tests_vs_energy_aware_20260626.csv
```

## Notes

The original full runs were executed on a remote Linux/GPU host. CPU-only execution is suitable for smoke checks but can be slow for full MVP80 evaluation.
The final evaluation configuration uses CUDA where available and expects an Optical Networking Gym checkout path to be configured locally.
