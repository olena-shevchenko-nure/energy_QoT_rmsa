# Top32 XLRON CF-Rank G160 MVP80 Rollout - 2026-06-10

## Scope

Full MVP80 rollout for the current best XLRON counterfactual branch:

- checkpoint: `runs/eon/quick_runtime_artifacts/top32_xlron_cf_rank_h100_h50_g160_v1_bucket_guard_e8_lr2e5/top32_xlron_counterfactual_rank_finetune_best.pt`
- config: `configs/experiments/eon/remote_ong_rollout_mvp80_top32_xlron_stabilized_ppo_gpu_compare.yaml`
- dataset: `data/eon/generated/nsfnet_mvp_ong_expert_hybrid_mix`
- split: `test`
- scope: 80 episodes, 60000 requests
- device: CUDA on `sw-l40`

Remote run:

```text
/home/oshevchenko/experiments/cse2026-ong-solver-20260530-0325/runs/eon/eon_ong_rollout_mvp80_top32_xlron_stabilized_ppo_gpu_compare/20260610_064934_unknown
```

## MVP80 Summary

| policy | accepted | blocking | reward/request | mean TopN index | mean latency ms |
|---|---:|---:|---:|---:|---:|
| `energy-aware-ksp-bm-ff` | 40487 | 0.325217 | 0.589505 | 2.518 | 1.790 |
| `q_head_heuristic` | 38660 | 0.355667 | 0.533681 | 0.450 | 1.656 |
| XLRON cf-rank g160 bucket-guard | 40832 | 0.319467 | 0.595022 | 1.747 | 8.066 |

Against the current MVP80 leaders:

| method | accepted | delta vs XLRON |
|---|---:|---:|
| DQN `orate60` | 40946 | +114 |
| LightGBM `old10` | 40898 | +66 |
| full-DQN `stratified32_e5` | 40873 | +41 |
| XLRON cf-rank g160 bucket-guard | 40832 | 0 |
| A3C distill from full-DQN | 40722 | -110 |
| `energy-aware-ksp-bm-ff` | 40487 | -345 |

## Bucket Deltas

| scenario | load | XLRON | base | delta vs base | q_head | delta vs q_head |
|---|---|---:|---:|---:|---:|---:|
| bursty | low | 3105 | 3084 | +21 | 2980 | +125 |
| bursty | medium | 2813 | 2761 | +52 | 2623 | +190 |
| bursty | high | 2384 | 2380 | +4 | 2254 | +130 |
| bursty | overload | 2019 | 2045 | -26 | 1906 | +113 |
| hotspot | low | 3155 | 3119 | +36 | 3033 | +122 |
| hotspot | medium | 2827 | 2760 | +67 | 2686 | +141 |
| hotspot | high | 2426 | 2449 | -23 | 2310 | +116 |
| hotspot | overload | 2077 | 2037 | +40 | 1862 | +215 |
| nonuniform | low | 2946 | 2901 | +45 | 2849 | +97 |
| nonuniform | medium | 2554 | 2532 | +22 | 2431 | +123 |
| nonuniform | high | 2170 | 2142 | +28 | 2038 | +132 |
| nonuniform | overload | 1812 | 1788 | +24 | 1675 | +137 |
| uniform | low | 3232 | 3184 | +48 | 3096 | +136 |
| uniform | medium | 2867 | 2844 | +23 | 2749 | +118 |
| uniform | high | 2372 | 2373 | -1 | 2246 | +126 |
| uniform | overload | 2073 | 2088 | -15 | 1922 | +151 |

XLRON improves over `energy-aware-ksp-bm-ff` in 12 of 16 buckets and beats `q_head_heuristic` in all 16 buckets. The remaining losses versus base are concentrated in `bursty/overload`, `hotspot/high`, `uniform/overload`, and a near-tie in `uniform/high`.

## Decision

This is the first XLRON branch that clearly beats `energy-aware-ksp-bm-ff` on full MVP80. It is now a legitimate MVP80 candidate, but it is still not the overall leader: it remains behind DQN `orate60` by 114 accepted requests, LightGBM `old10` by 66, and full-DQN `stratified32_e5` by 41.

The next XLRON improvement should target the four weak buckets without applying a coarse runtime guard. The previous bucket-margin and live-risk selector experiments worsened rollout behavior, so the safer direction is to improve the training signal or checkpoint selection for these buckets rather than add a threshold veto.

## Raw Artifacts

- `docs/experiments/raw/top32_xlron_cf_rank_g160_mvp80_policy_summary_20260610.csv`
- `docs/experiments/raw/top32_xlron_cf_rank_g160_mvp80_policy_episode_metrics_20260610.csv`
- `docs/experiments/raw/top32_xlron_cf_rank_g160_mvp80_metrics_20260610.json`
- `docs/experiments/raw/top32_xlron_cf_rank_g160_mvp80_config_resolved_20260610.yaml`
