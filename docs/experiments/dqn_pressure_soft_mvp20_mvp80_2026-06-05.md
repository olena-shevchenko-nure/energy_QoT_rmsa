# DQN Pressure-Soft Override Gate MVP20/MVP80

Date: 2026-06-05

Branch: `energy_aware_ksp_bm_ff_based`

## Change

Added `advantage_gate.context_gate_rules` for the distilled DQN candidate ranker. The tested pressure-soft artifact keeps the old `orate60` gate and applies an additional contextual threshold:

- condition: `pool_size_norm <= 0.3125`
- contextual `min_gate_score`: `0.0`
- artifact: `runs/eon/quick_runtime_artifacts/dqn_distill_old10_pressure_gate/torch_dqn_distill_old10_orate60_pressure_soft_tree_ranker.json`

## MVP20 Result

| policy | accepted | blocking | reward | override |
|---|---:|---:|---:|---:|
| `energy-aware-ksp-bm-ff` | 10218 | 0.318800 | 8944.187 | 0.000000 |
| `lightgbm_candidate_ranker_old10` | 10278 | 0.314800 | 8990.752 | 0.378770 |
| DQN `orate60` | 10306 | 0.312933 | 9030.273 | 0.432272 |
| DQN `pressure_soft` | 10321 | 0.311933 | 9040.340 | 0.310144 |

MVP20 conclusion: pressure-soft improved over DQN `orate60` by `+15` accepted requests and reduced override rate by about 12.2 percentage points.

## MVP80 Result

| policy | accepted | blocking | reward | override |
|---|---:|---:|---:|---:|
| `energy-aware-ksp-bm-ff` | 40487 | 0.325217 | 35370.313 | 0.000000 |
| `lightgbm_candidate_ranker_old10` | 40898 | 0.318367 | 35745.308 | 0.389970 |
| DQN `orate60` | 40946 | 0.317567 | 35828.680 | 0.426195 |
| DQN `pressure_soft` | 40808 | 0.319867 | 35667.160 | 0.310086 |

MVP80 conclusion: pressure-soft did not generalize. It lost `-138` accepted requests versus DQN `orate60` and `-90` versus LightGBM old10, despite reducing override rate.

## Scenario Notes

Pressure-soft helped the intended weak regimes versus DQN `orate60`:

| regime | pressure-soft vs DQN `orate60` |
|---|---:|
| `bursty/high` | +67 |
| `nonuniform/overload` | +16 |
| `uniform/high` | -12 |
| `uniform/overload` | 0 |

But it over-pruned useful overrides in other regimes:

| regime | pressure-soft vs DQN `orate60` |
|---|---:|
| `nonuniform/medium` | -66 |
| `hotspot/medium` | -44 |
| `bursty/medium` | -43 |
| `nonuniform/high` | -31 |
| `bursty/overload` | -30 |

## Raw Outputs

Local workspace copies:

- `C:\work\projects\CSE 2026\tmp_mvp20_dqn_distill_old10_pressure_soft_smoke`
- `C:\work\projects\CSE 2026\tmp_mvp80_dqn_distill_old10_pressure_soft_only`

Remote run:

- `/home/oshevchenko/experiments/cse2026-ong-solver-20260530-0325/runs/eon/eon_ong_rollout_mvp80_dqn_distill_old10_pressure_soft_only/20260605_175447_unknown`
