# Deterministic Heuristics

The paper compares three deterministic baselines implemented directly in source code:

- `energy-aware-ksp-bm-ff`
- `ksp-ff`
- `ksp-bm-ff`

These policies do not use learned model checkpoints.
They operate over the same feasible candidate surface as the learned policies and select actions using explicit route, modulation, spectrum, energy, and fragmentation rules.

The runtime implementation is under:

```text
src/cse2026/experiments/eon/ong_rollout.py
src/cse2026/experiments/eon/topn_baseline.py
src/cse2026/ong_solver/
```

