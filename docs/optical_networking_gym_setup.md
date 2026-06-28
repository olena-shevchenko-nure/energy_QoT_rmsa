# Optical Networking Gym Setup

The experiments use Optical Networking Gym as the online RMSA simulation environment.
The source evaluation config contains a machine-specific path:

```yaml
ong_source_path: /home/oshevchenko/experiments/optical-networking-gym
ong_topology_id: nsfnet_chen
```

For local reproduction, clone or install the same Optical Networking Gym version used by the experiment host and update `ong_source_path` in the evaluation config.

Required simulation settings in the paper comparison:

- topology: NSFNET / `nsfnet_chen`
- routes per demand: `k_routes: 5`
- spectrum slots: `slots: 100`
- candidate surface: Top-32 feasible candidates
- QoT constraint mode: `DIST`

The repository includes the NSFNET topology and modulation data under `data/eon`.
The Optical Networking Gym package itself is treated as an external dependency and is not vendored here.

