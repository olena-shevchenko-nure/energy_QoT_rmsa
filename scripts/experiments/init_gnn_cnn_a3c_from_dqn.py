#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cse2026.experiments.config import load_experiment_config
from cse2026.experiments.eon.ong_rollout import _raw_int, _solver_config
from cse2026.experiments.eon.train_dqn import _device
from cse2026.ong_solver.models import GnnCnnA3CNetwork, require_torch


def _resolve_path(path_text: str) -> Path:
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return ROOT / path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _copy_dqn_policy_weights(a3c_state: dict[str, Any], dqn_state: dict[str, Any]) -> dict[str, Any]:
    copied: list[str] = []
    missing: list[str] = []
    copied_state = dict(a3c_state)
    shared_prefixes = (
        "gnn.",
        "slot_cnn.",
        "route_pool.",
        "request_encoder.",
        "action_encoder.",
    )
    for key in list(copied_state):
        dqn_key = key
        if key.startswith("policy_head."):
            dqn_key = "q_head." + key[len("policy_head.") :]
        elif not key.startswith(shared_prefixes):
            continue
        if dqn_key not in dqn_state:
            missing.append(f"{key} <- {dqn_key}")
            continue
        if tuple(copied_state[key].shape) != tuple(dqn_state[dqn_key].shape):
            missing.append(f"{key} shape {tuple(copied_state[key].shape)} <- {dqn_key} shape {tuple(dqn_state[dqn_key].shape)}")
            continue
        copied_state[key] = dqn_state[dqn_key].detach().clone()
        copied.append(f"{key} <- {dqn_key}")
    return {"state_dict": copied_state, "copied": copied, "missing": missing}


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize a GNN+CNN A3C checkpoint from a full CandidateQNetwork DQN checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--initial-a3c-checkpoint", required=True)
    parser.add_argument("--initial-dqn-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    torch = require_torch()
    device = _device(config, torch)

    a3c_path = _resolve_path(args.initial_a3c_checkpoint)
    dqn_path = _resolve_path(args.initial_dqn_checkpoint)
    output_path = _resolve_path(args.output)

    # Weight surgery does not require GPU memory; keep checkpoint loading on CPU
    # so transplant initialization does not contend with training rollouts.
    a3c_checkpoint = torch.load(a3c_path, map_location="cpu", weights_only=False)
    dqn_checkpoint = torch.load(dqn_path, map_location="cpu", weights_only=False)
    if not isinstance(a3c_checkpoint, dict):
        raise ValueError("initial A3C checkpoint must be a dictionary")
    if not isinstance(dqn_checkpoint, dict):
        raise ValueError("initial DQN checkpoint must be a dictionary")

    action_feature_dim = int(a3c_checkpoint.get("action_feature_dim", dqn_checkpoint.get("action_feature_dim", 10)))
    hidden_dim = int(a3c_checkpoint.get("hidden_dim", dqn_checkpoint.get("hidden_dim", _raw_int(config, "hidden_dim", 128))))
    model = GnnCnnA3CNetwork(action_feature_dim=action_feature_dim, hidden_dim=hidden_dim)
    model.load_state_dict(a3c_checkpoint["model_state_dict"])

    dqn_state = dqn_checkpoint.get("model_state_dict", dqn_checkpoint)
    result = _copy_dqn_policy_weights(model.state_dict(), dqn_state)
    model.load_state_dict(result["state_dict"], strict=True)

    checkpoint_config = dict(a3c_checkpoint.get("config") or {})
    checkpoint_config.update(
        {
            "initialization": "dqn_policy_transplant",
            "initial_a3c_checkpoint": str(a3c_path),
            "initial_dqn_checkpoint": str(dqn_path),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "policy": "gnn_cnn_a3c",
            "n_max": int(a3c_checkpoint.get("n_max", _raw_int(config, "n_max", 32))),
            "action_feature_dim": int(action_feature_dim),
            "hidden_dim": int(hidden_dim),
            "config": checkpoint_config,
            "solver_config": a3c_checkpoint.get("solver_config") or asdict(_solver_config(config, neural=False)),
            "metrics": {
                "initialization": "dqn_policy_transplant",
                "copied_parameters": int(len(result["copied"])),
                "missing_parameters": int(len(result["missing"])),
            },
            "training_mode": "gnn_cnn_a3c_dqn_policy_transplant",
        },
        output_path,
    )

    summary = {
        "output": str(output_path),
        "initial_a3c_checkpoint": str(a3c_path),
        "initial_dqn_checkpoint": str(dqn_path),
        "device": str(device),
        "load_device": "cpu",
        "action_feature_dim": int(action_feature_dim),
        "hidden_dim": int(hidden_dim),
        "copied_parameters": int(len(result["copied"])),
        "missing_parameters": result["missing"],
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe(summary), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
