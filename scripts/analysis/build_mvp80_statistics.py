"""Rebuild MVP80 statistical summaries from per-episode metrics.

Usage:
    python scripts/analysis/build_mvp80_statistics.py \
        --episodes results/mvp80/raw/mvp80_selected_topn_p95_policy_episode_metrics_20260626.csv \
        --out-dir results/mvp80/statistics_recomputed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - optional dependency fallback
    stats = None


METRICS = [
    "accepted",
    "blocking_rate",
    "mean_reward",
    "mean_selected_energy_increment",
    "mean_selected_fragmentation_after",
    "mean_selected_qot_margin_norm",
    "mean_decision_latency_ms",
    "p95_decision_latency_ms",
    "mean_selected_topn_index",
    "p95_selected_topn_index",
]

DISPLAY_NAMES = {
    "torch_dqn_candidate_ranker_distill_old10": "Calibrated DQN-Override ranker",
    "lightgbm_candidate_ranker_old10": "LightGBM-override ranker",
    "gnn_cnn_dqn": "GNN-CNN Full DQN Ranker",
    "top32_xlron_stabilized_ppo": "XLRON Counterfactual Ranker",
    "gnn_cnn_a3c": "A3C Policy Distilled from Full DQN",
    "energy-aware-ksp-bm-ff": "Energy-Aware-KSP-BM-FF",
    "ksp-ff": "KSP-FF",
    "ksp-bm-ff": "KSP-BM-FF",
}


def confidence_interval_95(values: pd.Series) -> tuple[float, float]:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) < 2:
        mean = float(np.mean(arr)) if len(arr) else np.nan
        return mean, mean
    mean = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
    if stats is not None:
        delta = float(stats.t.ppf(0.975, len(arr) - 1) * sem)
    else:
        delta = 1.96 * sem
    return mean - delta, mean + delta


def paired_tests(df: pd.DataFrame, baseline: str, metric: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base = df[df["policy"] == baseline][["episode_id", metric]].rename(columns={metric: "baseline"})
    for policy in sorted(df["policy"].unique()):
        if policy == baseline:
            continue
        cur = df[df["policy"] == policy][["episode_id", metric]].rename(columns={metric: "candidate"})
        merged = cur.merge(base, on="episode_id", how="inner").dropna()
        delta = merged["candidate"] - merged["baseline"]
        t_p = np.nan
        wilcoxon_p = np.nan
        if stats is not None and len(delta) > 1:
            t_p = float(stats.ttest_rel(merged["candidate"], merged["baseline"]).pvalue)
            if np.any(delta.to_numpy() != 0):
                wilcoxon_p = float(stats.wilcoxon(merged["candidate"], merged["baseline"]).pvalue)
        rows.append(
            {
                "policy": policy,
                "policy_name": DISPLAY_NAMES.get(policy, policy),
                "baseline": baseline,
                "metric": metric,
                "n_pairs": len(merged),
                "mean_delta": float(delta.mean()) if len(delta) else np.nan,
                "paired_t_p": t_p,
                "wilcoxon_p": wilcoxon_p,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline", default="energy-aware-ksp-bm-ff")
    args = parser.parse_args()

    df = pd.read_csv(args.episodes)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for policy, group in df.groupby("policy", sort=True):
        for metric in METRICS:
            lo, hi = confidence_interval_95(group[metric])
            summary_rows.append(
                {
                    "policy": policy,
                    "policy_name": DISPLAY_NAMES.get(policy, policy),
                    "metric": metric,
                    "episodes": int(group["episode_id"].nunique()),
                    "mean": float(group[metric].mean()),
                    "std": float(group[metric].std(ddof=1)),
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )
    pd.DataFrame(summary_rows).to_csv(args.out_dir / "mvp80_statistical_summary_recomputed.csv", index=False)

    test_rows = []
    for metric in METRICS:
        test_rows.extend(paired_tests(df, args.baseline, metric))
    pd.DataFrame(test_rows).to_csv(args.out_dir / "mvp80_paired_tests_vs_energy_aware_recomputed.csv", index=False)

    scenario = (
        df.groupby(["policy", "traffic_scenario", "load_name"], as_index=False)
        .agg(
            episodes=("episode_id", "nunique"),
            accepted=("accepted", "mean"),
            blocking_rate=("blocking_rate", "mean"),
            reward=("mean_reward", "mean"),
            energy=("mean_selected_energy_increment", "mean"),
            fragmentation=("mean_selected_fragmentation_after", "mean"),
            qot=("mean_selected_qot_margin_norm", "mean"),
        )
        .assign(policy_name=lambda x: x["policy"].map(DISPLAY_NAMES).fillna(x["policy"]))
    )
    scenario.to_csv(args.out_dir / "mvp80_per_scenario_breakdown_recomputed.csv", index=False)


if __name__ == "__main__":
    main()

