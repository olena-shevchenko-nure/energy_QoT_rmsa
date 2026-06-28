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

The source evaluation config contains a machine-specific path, while the clean reproduction config uses an auto-installable local path:

```yaml
ong_source_path: /home/oshevchenko/experiments/optical-networking-gym
ong_topology_id: nsfnet_chen
```

For local reproduction, the wrapper clones and checks out the locked Optical Networking Gym version automatically when `external/optical-networking-gym` is missing:

```bash
python scripts/reproduce_mvp80.py
```

Equivalent manual commands are:

```bash
git clone https://github.com/carlosnatalino/optical-networking-gym.git external/optical-networking-gym
git -C external/optical-networking-gym checkout 622d0741ff75388161f7c468757ae880471d6d2b
```

Pass the local path to the evaluation wrapper if your checkout path differs; the wrapper can also clone to that target path if it is missing:

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
