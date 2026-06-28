# LightGBM-Override Ranker

Paper name: LightGBM-override ranker.

Runtime policy id: `lightgbm_candidate_ranker_old10`.

Artifact folder:

```text
artifacts/models/lightgbm_override_old10
```

This model is a gradient-boosted candidate ranker exported for online RMSA candidate selection.
The main ranker was trained with a ranking objective over candidate features and is used in a constrained override-style runtime regime.

The selected paper artifact is:

```text
lightgbm_lightgbm_old10_tree_ranker.json
```

The folder also contains auxiliary LightGBM/XGBoost exports and summaries from the same final quick-runtime artifact directory.
The `old10` suffix is a lineage label from the experiment series and is preserved to keep configs and result files traceable.

