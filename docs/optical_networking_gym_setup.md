# Optical Networking Gym Setup

The experiments use Optical Networking Gym as the online RMSA simulation environment.
The exact external dependency version is pinned in:

```text
third_party/optical-networking-gym.lock
```

Pinned version:

```text
repository: https://github.com/carlosnatalino/optical-networking-gym.git
commit: 622d0741ff75388161f7c468757ae880471d6d2b
branch at experiment time: main
```

The source evaluation config contains a machine-specific path:

```yaml
ong_source_path: /home/oshevchenko/experiments/optical-networking-gym
ong_topology_id: nsfnet_chen
```

For local reproduction, clone and checkout the locked Optical Networking Gym version:

```bash
git clone https://github.com/carlosnatalino/optical-networking-gym.git external/optical-networking-gym
git -C external/optical-networking-gym checkout 622d0741ff75388161f7c468757ae880471d6d2b
```

Then pass the local path to the evaluation wrapper if your checkout path differs:

```bash
python scripts/reproduce_mvp80.py --ong-source-path /path/to/optical-networking-gym
```

Required simulation settings in the paper comparison:

- topology: NSFNET / `nsfnet_chen`
- routes per demand: `k_routes: 5`
- spectrum slots: `slots: 100`
- candidate surface: Top-32 feasible candidates
- QoT constraint mode: `DIST`

The repository includes the NSFNET topology and modulation data under `data/eon`.
The Optical Networking Gym package itself is treated as an external dependency and is not vendored here.
