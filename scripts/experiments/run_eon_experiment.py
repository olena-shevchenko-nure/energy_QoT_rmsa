#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cse2026.experiments.config import PLACEHOLDER_STAGES, ExperimentConfig
from cse2026.experiments.eon.data_checks import summarize_data, validate_data
from cse2026.experiments.eon.dataloader_smoke import run_dataloader_smoke
from cse2026.experiments.eon.dagger_tree_ranker import run_train_dagger_tree_ranker
from cse2026.experiments.eon.ong_expert_dataset import run_collect_ong_expert_dataset
from cse2026.experiments.eon.lookahead_oracle import run_lookahead_oracle_eval
from cse2026.experiments.eon.lookahead_override import run_train_lookahead_override
from cse2026.experiments.eon.lookahead_tree_ranker import run_train_tree_ranker
from cse2026.experiments.eon.ong_solver_eval import run_ong_solver_eval
from cse2026.experiments.eon.ong_rollout import run_ong_rollout
from cse2026.experiments.eon.topn_baseline import evaluate_topn_baseline
from cse2026.experiments.eon.train_cnn import run_pretrain_cnn
from cse2026.experiments.eon.train_deeprmsa_a3c import run_train_deeprmsa_a3c
from cse2026.experiments.eon.train_deeprmsa_a3c_windowed_online import run_train_deeprmsa_a3c_windowed_online
from cse2026.experiments.eon.train_dqn import run_train_dqn
from cse2026.experiments.eon.train_dqn_online import run_train_dqn_online
from cse2026.experiments.eon.train_gnn_cnn_a3c_windowed_online import run_train_gnn_cnn_a3c_windowed_online
from cse2026.experiments.eon.train_gnn import run_pretrain_gnn
from cse2026.experiments.eon.train_top32_xlron_stabilized_ppo import run_train_top32_xlron_stabilized_ppo
from cse2026.experiments.eon.train_xlron_transformer_ppo import run_train_xlron_graph_transformer_ppo
from cse2026.experiments.logging_utils import configure_logging, tee_stdout
from cse2026.experiments.run_manager import RunManager
from cse2026.experiments.seeds import set_random_seeds


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_stage(config: ExperimentConfig, run_dir: Path):
    if config.stage == "validate_data":
        return validate_data(config, run_dir)
    if config.stage == "summarize_data":
        return summarize_data(config, run_dir)
    if config.stage == "topn_baseline_eval":
        return evaluate_topn_baseline(config, run_dir)
    if config.stage == "dataloader_smoke":
        return run_dataloader_smoke(config, run_dir)
    if config.stage == "ong_solver_eval":
        return run_ong_solver_eval(config, run_dir)
    if config.stage == "collect_ong_expert_dataset":
        return run_collect_ong_expert_dataset(config, run_dir)
    if config.stage == "pretrain_cnn":
        return run_pretrain_cnn(config, run_dir)
    if config.stage == "pretrain_gnn":
        return run_pretrain_gnn(config, run_dir)
    if config.stage == "train_dqn":
        return run_train_dqn(config, run_dir)
    if config.stage == "train_dqn_online":
        return run_train_dqn_online(config, run_dir)
    if config.stage == "train_deeprmsa_a3c":
        return run_train_deeprmsa_a3c(config, run_dir)
    if config.stage == "train_deeprmsa_a3c_windowed_online":
        return run_train_deeprmsa_a3c_windowed_online(config, run_dir)
    if config.stage == "train_gnn_cnn_a3c_windowed_online":
        return run_train_gnn_cnn_a3c_windowed_online(config, run_dir)
    if config.stage == "train_xlron_graph_transformer_ppo":
        return run_train_xlron_graph_transformer_ppo(config, run_dir)
    if config.stage == "train_top32_xlron_stabilized_ppo":
        return run_train_top32_xlron_stabilized_ppo(config, run_dir)
    if config.stage == "evaluate_policy":
        return run_ong_rollout(config, run_dir)
    if config.stage == "lookahead_oracle_eval":
        return run_lookahead_oracle_eval(config, run_dir)
    if config.stage == "train_lookahead_override":
        return run_train_lookahead_override(config, run_dir)
    if config.stage == "train_tree_ranker":
        return run_train_tree_ranker(config, run_dir)
    if config.stage == "train_dagger_tree_ranker":
        return run_train_dagger_tree_ranker(config, run_dir)
    if config.stage in PLACEHOLDER_STAGES:
        raise NotImplementedError(f"Stage {config.stage} is not implemented in this infrastructure PR.")
    raise ValueError(f"Unsupported stage: {config.stage}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an EON experiment stage.")
    parser.add_argument("--config", required=True, help="Experiment YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and exit without creating a run.")
    parser.add_argument("--resume", help="Resume latest or a specific run directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = ExperimentConfig.from_file(args.config, root=ROOT)
    if args.dry_run:
        print(json.dumps(config.resolved, indent=2, sort_keys=True))
        return 0

    manager = RunManager(config, root=ROOT)
    context = manager.create(resume=args.resume)
    logger = configure_logging(context.run_dir)
    exit_code = 0
    with tee_stdout(context.run_dir / "stdout.log"):
        logger.info("Starting EON stage %s in %s", config.stage, context.run_dir)
        seed_info = set_random_seeds(config.seed)
        _write_json(context.run_dir / "seed_info.json", seed_info)
        try:
            metrics = run_stage(config, context.run_dir)
            _write_json(context.run_dir / "metrics.json", metrics)
            logger.info("Completed EON stage %s", config.stage)
            print(f"RUN_DIR={context.run_dir}")
        except Exception as exc:
            exit_code = 1
            error = {
                "stage": config.stage,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _write_json(context.run_dir / "error.json", error)
            logger.exception("EON stage failed: %s", exc)
            print(f"RUN_DIR={context.run_dir}")
            print(f"ERROR={type(exc).__name__}: {exc}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
