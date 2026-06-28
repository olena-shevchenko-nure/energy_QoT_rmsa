# Data Description

The generated dataset used by the paper is:

```text
data/eon/generated/nsfnet_mvp_ong_expert_hybrid_mix
```

It contains train, validation, and test splits for NSFNET RMSA experiments.
The paper uses the MVP80 test setting: 80 test episodes and 60,000 online requests.

Traffic scenarios combine traffic pattern and load:

- traffic pattern: uniform, hotspot, nonuniform, bursty
- load: low, medium, high, overload

The dataset directory includes:

- `traffic/`: request sequences and split metadata.
- `candidates/`: generated feasible candidate actions and candidate-level features.
- `cnn/`: spectrum tensor features.
- `gnn/`: graph features.
- `dqn/`: DQN-style transition/features.
- `topology/`: topology representation used by generation.
- `reports/`: generation summaries.
- `manifest.json` and `checksums.sha256`: dataset provenance and integrity checks.

