# A3C Policy Distilled from Full DQN

Paper name: A3C Policy Distilled from Full DQN.

Runtime policy id: `gnn_cnn_a3c`.

Artifact folder:

```text
artifacts/models/a3c_distill_full_dqn
```

This model is a GNN-CNN actor-critic policy.
The actor outputs a masked distribution over feasible Top-N candidates, while the critic/value head is used during training to support the actor-critic objective.

The selected checkpoint is:

```text
gnn_cnn_a3c_distill_best.pt
```

The training branch used Full DQN as the distillation teacher.

