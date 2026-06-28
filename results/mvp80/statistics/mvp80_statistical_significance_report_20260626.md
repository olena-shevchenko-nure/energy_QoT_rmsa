# MVP80 Statistical Significance Analysis

Source: `mvp80_selected_topn_p95_policy_episode_metrics_20260626.csv`. All statistics are computed over 80 paired MVP80 episodes per policy. The paired baseline is `Energy-Aware-KSP-BM-FF`; pairing key is `episode_id` (equivalent to traffic scenario, load, and seed in this rollout).

Wilcoxon signed-rank p-values use a normal approximation with tie correction; Holm correction is applied separately within each metric across the seven baseline comparisons. Confidence intervals use the two-sided 95% t interval.

## Overall episode-level summary

| Method                                  | Total Acc. | Acc.         | Block           | Reward/request  | Reward 95% CI    | Energy       | Frag.           | QoT             | Mean latency, ms | P95 latency, ms | Avg Top-N     | P95 Top-N |
| --------------------------------------- | ---------- | ------------ | --------------- | --------------- | ---------------- | ------------ | --------------- | --------------- | ---------------- | --------------- | ------------- | --------- |
| Calibrated DQN-Override baseline ranker | 40 946     | 511.8 ± 90.4 | 0.3176 ± 0.1205 | 0.5971 ± 0.1292 | [0.5684, 0.6259] | 571.7 ± 41.3 | 0.3091 ± 0.0371 | 0.2658 ± 0.0098 | 2.307 ± 0.342    | 4.794 ± 0.560   | 1.936 ± 0.283 | 7.9 ± 1.3 |
| LightGBM-override ranker                | 40 898     | 511.2 ± 88.3 | 0.3184 ± 0.1177 | 0.5958 ± 0.1258 | [0.5678, 0.6238] | 571.8 ± 41.8 | 0.3299 ± 0.0414 | 0.2683 ± 0.0095 | 2.061 ± 0.283    | 4.366 ± 0.556   | 2.069 ± 0.296 | 8.1 ± 1.1 |
| GNN-CNN Full DQN Ranker                 | 40 873     | 510.9 ± 88.0 | 0.3188 ± 0.1173 | 0.5957 ± 0.1256 | [0.5677, 0.6236] | 572.6 ± 42.1 | 0.3109 ± 0.0352 | 0.2639 ± 0.0093 | 4.972 ± 0.795    | 8.589 ± 0.567   | 1.735 ± 0.260 | 7.3 ± 1.1 |
| XLRON Counterfactual Ranker             | 40 832     | 510.4 ± 88.7 | 0.3195 ± 0.1182 | 0.5950 ± 0.1266 | [0.5669, 0.6232] | 571.1 ± 42.2 | 0.3152 ± 0.0350 | 0.2652 ± 0.0088 | 4.738 ± 0.736    | 8.282 ± 0.560   | 1.747 ± 0.253 | 7.3 ± 1.1 |
| A3C Policy Distilled from Full DQN      | 40 722     | 509.0 ± 89.6 | 0.3213 ± 0.1195 | 0.5933 ± 0.1281 | [0.5648, 0.6218] | 572.1 ± 41.8 | 0.3138 ± 0.0380 | 0.2591 ± 0.0102 | 5.125 ± 0.816    | 8.836 ± 0.551   | 1.668 ± 0.290 | 7.1 ± 1.1 |
| Energy-Aware-KSP-BM-FF                  | 40 487     | 506.1 ± 85.8 | 0.3252 ± 0.1144 | 0.5895 ± 0.1229 | [0.5622, 0.6168] | 562.3 ± 37.9 | 0.3800 ± 0.0331 | 0.2629 ± 0.0088 | 1.806 ± 0.224    | 3.948 ± 0.523   | 2.518 ± 0.441 | 9.6 ± 1.5 |
| KSP-FF                                  | 40 369     | 504.6 ± 86.9 | 0.3272 ± 0.1158 | 0.5873 ± 0.1245 | [0.5596, 0.6150] | 562.8 ± 38.6 | 0.3785 ± 0.0316 | 0.2631 ± 0.0086 | 1.804 ± 0.231    | 3.958 ± 0.570   | 2.529 ± 0.462 | 9.6 ± 1.6 |
| KSP-BM-FF                               | 40 369     | 504.6 ± 86.9 | 0.3272 ± 0.1158 | 0.5873 ± 0.1245 | [0.5596, 0.6150] | 562.8 ± 38.6 | 0.3785 ± 0.0316 | 0.2631 ± 0.0086 | 1.803 ± 0.232    | 3.955 ± 0.566   | 2.529 ± 0.462 | 9.6 ± 1.6 |

## Paired significance tests vs Energy-Aware-KSP-BM-FF

| Method                                  | Metric         | Mean Δ vs Energy-Aware | 95% CI Δ           | Wilcoxon p (Holm) | Result               |
| --------------------------------------- | -------------- | ---------------------- | ------------------ | ----------------- | -------------------- |
| Calibrated DQN-Override baseline ranker | Acc.           | 5.7                    | [3.2, 8.2]         | 0.0002            | significantly better |
| Calibrated DQN-Override baseline ranker | Block          | -0.0076                | [-0.0110, -0.0043] | 0.0002            | significantly better |
| Calibrated DQN-Override baseline ranker | Reward/request | 0.0076                 | [0.0041, 0.0112]   | 0.0006            | significantly better |
| LightGBM-override ranker                | Acc.           | 5.1                    | [2.9, 7.4]         | 0.0002            | significantly better |
| LightGBM-override ranker                | Block          | -0.0069                | [-0.0098, -0.0039] | 0.0002            | significantly better |
| LightGBM-override ranker                | Reward/request | 0.0062                 | [0.0031, 0.0094]   | 0.0012            | significantly better |
| GNN-CNN Full DQN Ranker                 | Acc.           | 4.8                    | [2.1, 7.5]         | 0.0034            | significantly better |
| GNN-CNN Full DQN Ranker                 | Block          | -0.0064                | [-0.0100, -0.0028] | 0.0034            | significantly better |
| GNN-CNN Full DQN Ranker                 | Reward/request | 0.0062                 | [0.0023, 0.0101]   | 0.0072            | significantly better |
| XLRON Counterfactual Ranker             | Acc.           | 4.3                    | [2.1, 6.5]         | 0.0016            | significantly better |
| XLRON Counterfactual Ranker             | Block          | -0.0058                | [-0.0087, -0.0028] | 0.0016            | significantly better |
| XLRON Counterfactual Ranker             | Reward/request | 0.0055                 | [0.0023, 0.0087]   | 0.0064            | significantly better |
| A3C Policy Distilled from Full DQN      | Acc.           | 2.9                    | [0.6, 5.3]         | 0.0674            | not significant      |
| A3C Policy Distilled from Full DQN      | Block          | -0.0039                | [-0.0070, -0.0008] | 0.0674            | not significant      |
| A3C Policy Distilled from Full DQN      | Reward/request | 0.0038                 | [0.0004, 0.0072]   | 0.1224            | not significant      |
| KSP-FF                                  | Acc.           | -1.5                   | [-3.3, 0.3]        | 0.2163            | not significant      |
| KSP-FF                                  | Block          | 0.0020                 | [-0.0005, 0.0044]  | 0.2163            | not significant      |
| KSP-FF                                  | Reward/request | -0.0022                | [-0.0048, 0.0004]  | 0.1791            | not significant      |
| KSP-BM-FF                               | Acc.           | -1.5                   | [-3.3, 0.3]        | 0.2163            | not significant      |
| KSP-BM-FF                               | Block          | 0.0020                 | [-0.0005, 0.0044]  | 0.2163            | not significant      |
| KSP-BM-FF                               | Reward/request | -0.0022                | [-0.0048, 0.0004]  | 0.1791            | not significant      |

## Per-scenario best method by reward/request

| Scenario   | Load     | Best method                             | Best reward | Energy-aware reward | Delta reward | Best Acc. | Energy-aware Acc. | Delta Acc. |
| ---------- | -------- | --------------------------------------- | ----------- | ------------------- | ------------ | --------- | ----------------- | ---------- |
| uniform    | low      | XLRON Counterfactual Ranker             | 0.7857      | 0.7742              | 0.0115       | 646.4     | 636.8             | 9.6        |
| uniform    | medium   | GNN-CNN Full DQN Ranker                 | 0.6913      | 0.6794              | 0.0119       | 577.6     | 568.8             | 8.8        |
| uniform    | high     | LightGBM-override ranker                | 0.5524      | 0.5434              | 0.0090       | 481.4     | 474.6             | 6.8        |
| uniform    | overload | LightGBM-override ranker                | 0.4686      | 0.4638              | 0.0047       | 421.4     | 417.6             | 3.8        |
| hotspot    | low      | Calibrated DQN-Override baseline ranker | 0.7843      | 0.7650              | 0.0193       | 638.0     | 623.8             | 14.2       |
| hotspot    | medium   | Calibrated DQN-Override baseline ranker | 0.6880      | 0.6632              | 0.0248       | 569.6     | 552.0             | 17.6       |
| hotspot    | high     | Calibrated DQN-Override baseline ranker | 0.5775      | 0.5743              | 0.0031       | 492.4     | 489.8             | 2.6        |
| hotspot    | overload | XLRON Counterfactual Ranker             | 0.4669      | 0.4557              | 0.0112       | 415.4     | 407.4             | 8.0        |
| nonuniform | low      | Calibrated DQN-Override baseline ranker | 0.7093      | 0.6863              | 0.0230       | 596.6     | 580.2             | 16.4       |
| nonuniform | medium   | Calibrated DQN-Override baseline ranker | 0.6028      | 0.5840              | 0.0188       | 519.6     | 506.4             | 13.2       |
| nonuniform | high     | XLRON Counterfactual Ranker             | 0.4819      | 0.4743              | 0.0077       | 434.0     | 428.4             | 5.6        |
| nonuniform | overload | GNN-CNN Full DQN Ranker                 | 0.3862      | 0.3724              | 0.0138       | 367.2     | 357.6             | 9.6        |
| bursty     | low      | A3C Policy Distilled from Full DQN      | 0.7535      | 0.7449              | 0.0086       | 623.6     | 616.8             | 6.8        |
| bursty     | medium   | XLRON Counterfactual Ranker             | 0.6687      | 0.6539              | 0.0148       | 562.6     | 552.2             | 10.4       |
| bursty     | high     | LightGBM-override ranker                | 0.5558      | 0.5464              | 0.0094       | 483.2     | 476.0             | 7.2        |
| bursty     | overload | GNN-CNN Full DQN Ranker                 | 0.4561      | 0.4507              | 0.0054       | 412.8     | 409.0             | 3.8        |

## Analysis

The overall ordering remains consistent with the aggregate MVP80 table: Calibrated DQN-Override baseline ranker has the strongest accepted count and reward/request (0.5971 ± 0.1292 over episodes), while Energy-Aware-KSP-BM-FF remains lower on the main acceptance-oriented metrics (0.5895 ± 0.1229).

- For Acc., Calibrated DQN-Override baseline ranker is significantly better than Energy-Aware-KSP-BM-FF (mean delta 5.7, Holm-adjusted Wilcoxon p=0.0002).
- For Block, Calibrated DQN-Override baseline ranker is significantly better than Energy-Aware-KSP-BM-FF (mean delta -0.0076, Holm-adjusted Wilcoxon p=0.0002).
- For Reward/request, Calibrated DQN-Override baseline ranker is significantly better than Energy-Aware-KSP-BM-FF (mean delta 0.0076, Holm-adjusted Wilcoxon p=0.0006).

The scenario breakdown shows that the advantage is not concentrated in a single cell: ML candidate-rankers are selected as the best reward/request method in most traffic/load cells. At the same time, deterministic heuristics keep lower decision latency and lower energy in several cells, so the statistical gain should be interpreted primarily as an acceptance/reward improvement rather than a universal improvement across all secondary metrics.

## Conclusions

The MVP80 comparison supports the claim that the best candidate-ranking/override models outperform Energy-Aware-KSP-BM-FF on the primary online objective with paired episode-level evidence. The evidence is strongest for reward/request and accepted requests; energy and latency remain trade-off dimensions where simpler heuristics can still be preferable. For the paper, the most defensible statement is that the top ML policies give statistically significant online acceptance/reward gains under the same paired episode set, not that they dominate every metric.