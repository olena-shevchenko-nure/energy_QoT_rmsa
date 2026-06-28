# GNN-CNN Full DQN Ranker

Paper name: GNN-CNN Full DQN Ranker.

Runtime policy id: `gnn_cnn_dqn`.

Artifact folder:

```text
artifacts/models/full_dqn_stratified32_e5
```

This is the full neural candidate-ranking DQN model.
It uses graph/network, spectrum, request, and candidate/action encoders, followed by a DQN-style Q head that scores feasible Top-N candidates.

The selected checkpoint is:

```text
full_dqn_orate60_distill_frozen.pt
```

The `stratified32_e5` suffix denotes the selected training branch/checkpoint used in the MVP80 comparison.

