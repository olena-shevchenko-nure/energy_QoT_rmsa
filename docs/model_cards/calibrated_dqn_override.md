# Calibrated DQN-Override Baseline Ranker

Paper name: Calibrated DQN-Override baseline ranker.

Runtime policy id: `torch_dqn_candidate_ranker_distill_old10`.

Artifact folder:

```text
artifacts/models/calibrated_dqn_override
```

This model is a DQN-like MLP candidate scorer with auxiliary override-rate calibration artifacts.
It operates over feasible candidate actions and is used as a calibrated override policy relative to the Energy-Aware-KSP-BM-FF baseline decision.

The runtime export includes:

- ranker model: `torch_dqn_distill_ranker.pt`
- advantage/gate models: `torch_dqn_distill_advantage_win.pt`, `torch_dqn_distill_advantage_loss.pt`, `torch_dqn_distill_advantage_delta.pt`
- selected runtime config: `torch_dqn_distill_old10_orate60_tree_ranker.json`
- override calibration summary: `torch_dqn_override_rate_calibration_summary.json`

The historical `orate60` suffix denotes the selected override-rate calibration setting used in the MVP80 comparison.

