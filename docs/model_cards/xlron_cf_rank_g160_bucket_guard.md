# XLRON Counterfactual Ranker

Paper name: XLRON Counterfactual Ranker.

Runtime policy id: `top32_xlron_stabilized_ppo`.

Artifact folder:

```text
artifacts/models/xlron_cf_rank_g160_bucket_guard
```

This model is an XLRON-style neural candidate ranker/policy adapted to the shared Top-32 candidate surface.
The selected branch combines stabilized neural training with counterfactual ranking fine-tuning and a bucket guard used at runtime.

The selected checkpoint is:

```text
top32_xlron_counterfactual_rank_finetune_best.pt
```

The folder also includes training logs, the counterfactual-rank fine-tuning summary, and validation rollout folders from the selected run.

