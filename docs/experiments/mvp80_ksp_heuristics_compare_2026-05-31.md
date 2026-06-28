# MVP80 KSP Heuristics Compare - 2026-05-31

## Scope

Dataset: `data/eon/generated/nsfnet_mvp_ong_expert_hybrid_mix`

Evaluation split: `test`, full 80 episodes / 60000 requests. The split contains 5 episodes for every `(traffic_scenario, load_name)` pair:

- scenarios: `uniform`, `hotspot`, `nonuniform`, `bursty`
- loads: `low`, `medium`, `high`, `overload`

New policies evaluated in ONG rollout:

- `ksp-ff`: lowest route id, then first-fit slot start, then smaller modulation offset, width, energy.
- `ksp-bm-ff`: lowest route id, then first-fit slot start, then smaller width / better modulation, then energy.
- `energy-aware-ksp-bm-ff`: lowest energy increment first, then route id, first-fit slot start, width / modulation.

Important implementation note: these are KSP-style policies over the current solver Top-N candidate set, not a raw exhaustive ONG action-space scan.

## Aggregate Results

Rows are sorted by accepted requests. Blocking delta is shown in percentage points against `q_head_heuristic`.

| Policy | Accepted | Blocking | Reward/request | Energy | Fragmentation | QoT | Delay | Accepted delta vs q_head | Blocking delta vs q_head, pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `energy-aware-ksp-bm-ff` | 40487 | 0.325217 | 0.589505 | 562.305 | 0.379986 | 0.262877 | 8.682 | 1827 | -3.045 |
| `ksp-ff` | 40369 | 0.327183 | 0.587338 | 562.804 | 0.378475 | 0.263078 | 8.688 | 1709 | -2.848 |
| `ksp-bm-ff` | 40369 | 0.327183 | 0.587338 | 562.804 | 0.378475 | 0.263078 | 8.688 | 1709 | -2.848 |
| `xgboost_candidate_ranker` | 39468 | 0.342200 | 0.556075 | 576.907 | 0.237892 | 0.386548 | 8.965 | 808 | -1.347 |
| `lightgbm_candidate_ranker` | 39455 | 0.342417 | 0.556025 | 575.527 | 0.234981 | 0.387260 | 8.938 | 795 | -1.325 |
| `q_head_heuristic` | 38660 | 0.355667 | 0.533681 | 573.721 | 0.216431 | 0.450210 | 8.909 | 0 | 0.000 |
| `j_total_heuristic` | 38645 | 0.355917 | 0.538114 | 596.013 | 0.222143 | 0.389020 | 9.324 | -15 | 0.025 |
| `deeprmsa_a3c_corrected` | 38645 | 0.355917 | 0.538114 | 596.013 | 0.222143 | 0.389020 | 9.324 | -15 | 0.025 |
| `residual_dqn_clamp010` | 38610 | 0.356500 | 0.533142 | 573.982 | 0.224164 | 0.446630 | 8.913 | -50 | 0.083 |
| `residual_dqn_clamp050` | 38601 | 0.356650 | 0.532989 | 574.833 | 0.230624 | 0.445958 | 8.929 | -59 | 0.098 |
| `xlron_graph_transformer_ppo` | 38599 | 0.356683 | 0.534684 | 585.294 | 0.221503 | 0.421669 | 9.128 | -61 | 0.102 |
| `old_residual_dqn` | 37926 | 0.367900 | 0.518879 | 580.529 | 0.232092 | 0.454511 | 9.039 | -734 | 1.223 |
| `random_feasible` | 37874 | 0.368767 | 0.525575 | 591.235 | 0.464324 | 0.359800 | 9.232 | -786 | 1.310 |

## KSP Scenario Breakdown

Accepted requests by traffic scenario:

| Policy | Bursty | Hotspot | Nonuniform | Uniform |
|---|---:|---:|---:|---:|
| `energy-aware-ksp-bm-ff` | 10270 | 10365 | 9363 | 10489 |
| `ksp-bm-ff` | 10266 | 10348 | 9267 | 10488 |
| `ksp-ff` | 10266 | 10348 | 9267 | 10488 |

Mean blocking rate by traffic scenario:

| Policy | Bursty | Hotspot | Nonuniform | Uniform |
|---|---:|---:|---:|---:|
| `energy-aware-ksp-bm-ff` | 0.315333 | 0.309000 | 0.375800 | 0.300733 |
| `ksp-bm-ff` | 0.315600 | 0.310133 | 0.382200 | 0.300800 |
| `ksp-ff` | 0.315600 | 0.310133 | 0.382200 | 0.300800 |

## Interpretation

The KSP-style heuristics are now the strongest policies on accepted requests, blocking, and the current scalar rollout reward.

`energy-aware-ksp-bm-ff` is the best aggregate result in this run:

- +1827 accepted requests versus `q_head_heuristic`.
- -3.045 percentage points blocking versus `q_head_heuristic`.
- +1019 accepted requests versus `xgboost_candidate_ranker`.
- -1.698 percentage points blocking versus `xgboost_candidate_ranker`.

The result is not a clean dominance result. KSP improves acceptance and lowers mean selected energy, but it substantially worsens fragmentation and QoT margin:

- fragmentation: `energy-aware-ksp-bm-ff` 0.379986 vs `q_head_heuristic` 0.216431.
- QoT margin: `energy-aware-ksp-bm-ff` 0.262877 vs `q_head_heuristic` 0.450210.

`ksp-ff` and `ksp-bm-ff` are identical in this run. In the current candidate construction, the additional BM tie-breaks did not change the selected action after route and first-fit slot priority.

The practical reading is:

1. If the primary conference metric is service acceptance / blocking under the current reward, `energy-aware-ksp-bm-ff` is the new strongest baseline.
2. If the problem statement must emphasize energy plus spectrum health and QoT margin, the KSP result needs a hybrid constraint or penalty; otherwise it buys lower blocking by using spectrally/QoT-weaker placements.
3. The next ML comparison should not use `q_head` alone as the expert ceiling. It should also compare to `energy-aware-ksp-bm-ff`, or train a conservative model/ranker to imitate KSP only when fragmentation/QoT constraints remain acceptable.

## Artifacts

Summary:

- `docs/experiments/mvp80_ksp_heuristics_compare_2026-05-31.csv`

Raw KSP run:

- `docs/experiments/raw/eon_ong_rollout_mvp80_ksp_heuristics_20260531_231403/policy_summary.csv`
- `docs/experiments/raw/eon_ong_rollout_mvp80_ksp_heuristics_20260531_231403/policy_episode_metrics.csv`
- `docs/experiments/raw/eon_ong_rollout_mvp80_ksp_heuristics_20260531_231403/metrics.json`
- `docs/experiments/raw/eon_ong_rollout_mvp80_ksp_heuristics_20260531_231403/config.resolved.yaml`

