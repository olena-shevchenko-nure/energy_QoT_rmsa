from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import ExperimentConfig


ROUTE_REGRESSION_LABELS = (
    "delay_norm",
    "energy_norm",
    "route_fragmentation",
    "c_route_max_norm",
    "qot_margin_norm",
    "qot_risk",
)

ROUTE_BASIC_COLUMNS = (
    "route_length_norm",
    "hop_count_norm",
    "required_slots_norm",
    "c_route_max_norm",
)


def _write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _splits(config: ExperimentConfig) -> tuple[str, str, str]:
    values = tuple(config.splits or ("train", "val", "test"))
    train = values[0] if len(values) >= 1 else "train"
    val = values[1] if len(values) >= 2 else "val"
    test = values[2] if len(values) >= 3 else "test"
    return train, val, test


def _device(config: ExperimentConfig, torch: Any) -> str:
    requested = str(config.resolved.get("device", config.device))
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _raw_int(config: ExperimentConfig, key: str, default: int) -> int:
    return int(config.resolved.get(key, config.raw.get(key, default)))


def _raw_float(config: ExperimentConfig, key: str, default: float) -> float:
    return float(config.resolved.get(key, config.raw.get(key, default)))


def _json_ints(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return ()
        if isinstance(parsed, list):
            return tuple(int(item) for item in parsed)
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return ()


def _iter_batches(size: int, batch_size: int, *, shuffle: bool, rng: np.random.Generator) -> list[np.ndarray]:
    indices = np.arange(size, dtype=np.int64)
    if shuffle:
        rng.shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, size, batch_size)]


def _load_split(dataset_path: Path, split: str) -> dict[str, Any]:
    graphs = np.load(dataset_path / "gnn" / f"{split}_graphs.npz")
    routes = pd.read_parquet(dataset_path / "gnn" / f"{split}_routes.parquet")
    if routes.empty or not set(ROUTE_BASIC_COLUMNS).issubset(routes.columns):
        return _load_split_from_candidates(dataset_path, split, graphs)
    sample_ids = [str(value) for value in graphs["sample_ids"]]
    sample_index = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    row_sample_indices = routes["sample_id"].map(sample_index).to_numpy(dtype=np.int64)
    if np.any(row_sample_indices < 0):
        raise RuntimeError(f"{split}: some route rows do not match graph sample_ids")

    edge_count = int(graphs["link_features"].shape[1])
    route_mask = np.zeros((len(routes), edge_count), dtype=np.float32)
    for row_idx, value in enumerate(routes["route_directed_link_ids"].tolist()):
        for link_id in _json_ints(value):
            if 0 <= link_id < edge_count:
                route_mask[row_idx, link_id] = 1.0

    return {
        "node_features": graphs["node_features"].astype(np.float32),
        "link_features": graphs["link_features"].astype(np.float32),
        "global_features": graphs["global_features"].astype(np.float32),
        "request_features": graphs["request_features"].astype(np.float32),
        "edge_index": graphs["edge_index"].astype(np.int64),
        "sample_indices": row_sample_indices,
        "route_mask": route_mask,
        "route_basic": routes[list(ROUTE_BASIC_COLUMNS)].to_numpy(dtype=np.float32),
        "route_labels": routes[list(ROUTE_REGRESSION_LABELS)].to_numpy(dtype=np.float32),
        "feasible": routes["feasible_label"].to_numpy(dtype=np.float32),
        "rank_targets": (-routes["heuristic_route_score"].to_numpy(dtype=np.float32)).astype(np.float32),
        "group_ids": routes["sample_id"].astype(str).to_numpy(),
        "modulation_id": routes["modulation_id"].to_numpy(dtype=np.int64),
        "block_now": routes["block_now"].to_numpy(dtype=np.float32),
        "num_feasible_norm": routes["num_feasible_norm"].to_numpy(dtype=np.float32),
        "global_fragmentation": routes["global_fragmentation"].to_numpy(dtype=np.float32),
    }


def _load_split_from_candidates(dataset_path: Path, split: str, graphs: np.lib.npyio.NpzFile) -> dict[str, Any]:
    candidates = pd.read_parquet(dataset_path / "candidates" / f"{split}.parquet")
    candidates = candidates[candidates["candidate_mask"].astype(bool)].reset_index(drop=True)
    if candidates.empty:
        raise RuntimeError(f"{split}: no real candidate rows available for GNN pretraining fallback")

    manifest_path = dataset_path / "manifest.json"
    params: dict[str, Any] = {}
    slot_count = 100
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        params = dict(manifest.get("parameters", {}))
        slot_count = int(manifest.get("slot_total", params.get("slots", slot_count)))
    n_max = int(params.get("n_max", candidates["n_max"].max() if "n_max" in candidates else 32))
    max_route_length = float(params.get("max_route_length_norm_km", 6000.0))

    sample_ids = [str(value) for value in graphs["sample_ids"]]
    sample_index = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    state_ids = candidates["episode_id"].astype(str) + ":" + candidates["request_id"].astype(str)
    row_sample_indices = np.asarray([sample_index[str(state_id)] for state_id in state_ids], dtype=np.int64)

    edge_count = int(graphs["link_features"].shape[1])
    route_mask = np.zeros((len(candidates), edge_count), dtype=np.float32)
    for row_idx, value in enumerate(candidates["route_directed_link_ids"].tolist()):
        for link_id in _json_ints(value):
            if 0 <= link_id < edge_count:
                route_mask[row_idx, link_id] = 1.0

    c_route_max_norm = candidates["largest_free_block_after"].to_numpy(dtype=np.float32) / float(max(slot_count, 1))
    route_basic = np.stack(
        [
            candidates["route_length_km"].to_numpy(dtype=np.float32) / float(max(max_route_length, 1e-9)),
            candidates["hop_count"].to_numpy(dtype=np.float32) / 8.0,
            candidates["required_slots"].to_numpy(dtype=np.float32) / float(max(slot_count, 1)),
            c_route_max_norm,
        ],
        axis=1,
    ).astype(np.float32)
    route_labels = np.stack(
        [
            candidates["delay_norm"].to_numpy(dtype=np.float32),
            candidates["energy_increment_norm"].to_numpy(dtype=np.float32),
            candidates["fragmentation_after"].to_numpy(dtype=np.float32),
            c_route_max_norm,
            candidates["qot_margin_norm"].to_numpy(dtype=np.float32),
            candidates["qot_risk"].to_numpy(dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    num_feasible_norm = np.clip(
        candidates.groupby(["episode_id", "request_id"])["candidate_id"].transform("count").to_numpy(dtype=np.float32)
        / float(max(n_max, 1)),
        0.0,
        1.0,
    )

    return {
        "node_features": graphs["node_features"].astype(np.float32),
        "link_features": graphs["link_features"].astype(np.float32),
        "global_features": graphs["global_features"].astype(np.float32),
        "request_features": graphs["request_features"].astype(np.float32),
        "edge_index": graphs["edge_index"].astype(np.int64),
        "sample_indices": row_sample_indices,
        "route_mask": route_mask,
        "route_basic": route_basic,
        "route_labels": route_labels,
        "feasible": candidates["is_feasible"].astype(bool).to_numpy(dtype=np.float32),
        "rank_targets": candidates["q_head_score"].to_numpy(dtype=np.float32),
        "group_ids": state_ids.astype(str).to_numpy(),
        "modulation_id": candidates["modulation_id"].to_numpy(dtype=np.int64),
        "block_now": np.zeros((len(candidates),), dtype=np.float32),
        "num_feasible_norm": num_feasible_norm.astype(np.float32),
        "global_fragmentation": candidates["fragmentation_before"].to_numpy(dtype=np.float32),
    }


def _classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    pred = probs >= 0.5
    truth = labels.astype(bool)
    tp = int(np.logical_and(pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    total = max(len(labels), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": float((tp + tn) / total),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _ranking_loss(rank_scores: Any, group_ids: np.ndarray, targets: Any, torch: Any) -> Any:
    losses = []
    for group_id in np.unique(group_ids):
        positions_np = np.flatnonzero(group_ids == group_id)
        if positions_np.size < 2:
            continue
        positions = torch.as_tensor(positions_np, dtype=torch.long, device=rank_scores.device)
        group_scores = rank_scores.index_select(0, positions)
        group_targets = targets.index_select(0, positions)
        diff_target = group_targets[:, None] - group_targets[None, :]
        good = diff_target > 1e-6
        if not bool(good.any()):
            continue
        diff_pred = group_scores[:, None] - group_scores[None, :]
        losses.append(torch.nn.functional.softplus(-diff_pred[good]).mean())
    if not losses:
        return rank_scores.new_tensor(0.0)
    return torch.stack(losses).mean()


def _ranking_metrics(scores: np.ndarray, targets: np.ndarray, group_ids: np.ndarray) -> dict[str, float | int | None]:
    pair_correct = 0
    pair_total = 0
    top1 = 0
    top3 = 0
    groups = 0
    for group_id in np.unique(group_ids):
        idx = np.flatnonzero(group_ids == group_id)
        if idx.size < 2:
            continue
        groups += 1
        group_scores = scores[idx]
        group_targets = targets[idx]
        best = int(np.argmax(group_targets))
        order = np.argsort(-group_scores)
        top1 += int(order[0] == best)
        top3 += int(best in set(int(x) for x in order[: min(3, len(order))]))
        for left in range(idx.size):
            for right in range(left + 1, idx.size):
                diff = float(group_targets[left] - group_targets[right])
                if abs(diff) <= 1e-6:
                    continue
                pair_total += 1
                pred_diff = float(group_scores[left] - group_scores[right])
                pair_correct += int((diff > 0 and pred_diff > 0) or (diff < 0 and pred_diff < 0))
    return {
        "groups": int(groups),
        "pairwise_accuracy": None if pair_total == 0 else float(pair_correct / pair_total),
        "top1_accuracy": None if groups == 0 else float(top1 / groups),
        "top3_accuracy": None if groups == 0 else float(top3 / groups),
    }


def _build_model(hidden_dim: int, modulation_count: int):
    from cse2026.ong_solver.models import EdgeStateGNN, MLP, RequestEncoder, require_torch

    torch = require_torch()
    nn = torch.nn

    class GnnPretrainer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gnn = EdgeStateGNN(hidden_dim=hidden_dim)
            self.route_pool = nn.Sequential(
                nn.Linear(hidden_dim * 2 + len(ROUTE_BASIC_COLUMNS), hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.request_encoder = RequestEncoder(out_dim=64)
            self.modulation_encoder = nn.Embedding(max(modulation_count, 1), 32)
            fusion_dim = hidden_dim + hidden_dim + 64 + 32 + len(ROUTE_BASIC_COLUMNS)
            self.fusion = MLP(fusion_dim, hidden_dim * 2, hidden_dim)
            self.route_regression_head = nn.Linear(hidden_dim, len(ROUTE_REGRESSION_LABELS))
            self.feasible_head = nn.Linear(hidden_dim, 1)
            self.rank_head = nn.Linear(hidden_dim, 1)
            self.block_head = nn.Sequential(nn.Linear(hidden_dim + 64, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
            self.global_regression_head = nn.Sequential(nn.Linear(hidden_dim + 64, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))

        def route_embeddings(self, link_embeddings, route_mask, route_basic):
            mask = route_mask.unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            mean_pool = (link_embeddings * mask).sum(dim=1) / denom
            masked = link_embeddings.masked_fill(~route_mask.unsqueeze(-1).bool(), -1e9)
            max_pool = masked.max(dim=1).values
            max_pool = torch.where(max_pool < -1e8, torch.zeros_like(max_pool), max_pool)
            return self.route_pool(torch.cat([mean_pool, max_pool, route_basic], dim=-1))

        def forward(
            self,
            node_features,
            link_features,
            global_features,
            edge_index,
            request_features,
            route_mask,
            route_basic,
            modulation_id,
        ):
            h_global, h_links = self.gnn(node_features, link_features, global_features, edge_index)
            h_route = self.route_embeddings(h_links, route_mask, route_basic)
            h_req = self.request_encoder(request_features)
            h_mod = self.modulation_encoder(modulation_id.clamp_min(0).clamp_max(self.modulation_encoder.num_embeddings - 1))
            fused = self.fusion(torch.cat([h_global, h_route, h_req, h_mod, route_basic], dim=-1))
            global_fused = torch.cat([h_global, h_req], dim=-1)
            global_reg = self.global_regression_head(global_fused)
            return {
                "route_regression": self.route_regression_head(fused),
                "feasible_logit": self.feasible_head(fused).squeeze(-1),
                "rank_score": self.rank_head(fused).squeeze(-1),
                "block_logit": self.block_head(global_fused).squeeze(-1),
                "num_feasible_norm": global_reg[:, 0],
                "global_fragmentation": global_reg[:, 1],
            }

    return torch, GnnPretrainer()


def _batch_tensors(data: dict[str, Any], idx: np.ndarray, device: str, torch: Any) -> dict[str, Any]:
    graph_idx = data["sample_indices"][idx]
    return {
        "node_features": torch.as_tensor(data["node_features"][graph_idx], dtype=torch.float32, device=device),
        "link_features": torch.as_tensor(data["link_features"][graph_idx], dtype=torch.float32, device=device),
        "global_features": torch.as_tensor(data["global_features"][graph_idx], dtype=torch.float32, device=device),
        "request_features": torch.as_tensor(data["request_features"][graph_idx], dtype=torch.float32, device=device),
        "route_mask": torch.as_tensor(data["route_mask"][idx], dtype=torch.float32, device=device),
        "route_basic": torch.as_tensor(data["route_basic"][idx], dtype=torch.float32, device=device),
        "route_labels": torch.as_tensor(data["route_labels"][idx], dtype=torch.float32, device=device),
        "feasible": torch.as_tensor(data["feasible"][idx], dtype=torch.float32, device=device),
        "rank_targets": torch.as_tensor(data["rank_targets"][idx], dtype=torch.float32, device=device),
        "modulation_id": torch.as_tensor(data["modulation_id"][idx], dtype=torch.long, device=device),
        "block_now": torch.as_tensor(data["block_now"][idx], dtype=torch.float32, device=device),
        "num_feasible_norm": torch.as_tensor(data["num_feasible_norm"][idx], dtype=torch.float32, device=device),
        "global_fragmentation": torch.as_tensor(data["global_fragmentation"][idx], dtype=torch.float32, device=device),
    }


def _evaluate(
    model: Any,
    data: dict[str, Any],
    batch_size: int,
    device: str,
    torch: Any,
    *,
    max_batches: int = 0,
) -> dict[str, Any]:
    model.eval()
    route_preds = []
    feasible_logits = []
    rank_scores = []
    block_logits = []
    num_preds = []
    frag_preds = []
    batches = _iter_batches(len(data["route_labels"]), batch_size, shuffle=False, rng=np.random.default_rng(0))
    if max_batches > 0:
        batches = batches[:max_batches]
    eval_idx = np.concatenate(batches, axis=0) if batches else np.asarray([], dtype=np.int64)
    with torch.no_grad():
        for idx in batches:
            batch = _batch_tensors(data, idx, device, torch)
            out = model(
                batch["node_features"],
                batch["link_features"],
                batch["global_features"],
                torch.as_tensor(data["edge_index"], dtype=torch.long, device=device),
                batch["request_features"],
                batch["route_mask"],
                batch["route_basic"],
                batch["modulation_id"],
            )
            route_preds.append(out["route_regression"].detach().cpu().numpy())
            feasible_logits.append(out["feasible_logit"].detach().cpu().numpy())
            rank_scores.append(out["rank_score"].detach().cpu().numpy())
            block_logits.append(out["block_logit"].detach().cpu().numpy())
            num_preds.append(out["num_feasible_norm"].detach().cpu().numpy())
            frag_preds.append(out["global_fragmentation"].detach().cpu().numpy())

    route_pred = np.concatenate(route_preds, axis=0)
    route_true = data["route_labels"][eval_idx]
    mae = np.mean(np.abs(route_pred - route_true), axis=0)
    num_pred = np.concatenate(num_preds, axis=0)
    frag_pred = np.concatenate(frag_preds, axis=0)
    return {
        "samples": int(len(route_true)),
        "route_regression_mae": {name: float(value) for name, value in zip(ROUTE_REGRESSION_LABELS, mae)},
        "feasible": _classification_metrics(np.concatenate(feasible_logits, axis=0), data["feasible"][eval_idx]),
        "ranking": _ranking_metrics(
            np.concatenate(rank_scores, axis=0),
            data["rank_targets"][eval_idx],
            data["group_ids"][eval_idx],
        ),
        "global": {
            "block": _classification_metrics(np.concatenate(block_logits, axis=0), data["block_now"][eval_idx]),
            "num_feasible_norm_mae": float(np.mean(np.abs(num_pred - data["num_feasible_norm"][eval_idx]))),
            "global_fragmentation_mae": float(np.mean(np.abs(frag_pred - data["global_fragmentation"][eval_idx]))),
        },
    }


def run_pretrain_gnn(config: ExperimentConfig, run_dir: str | Path) -> dict[str, Any]:
    if config.dataset_path is None:
        raise ValueError("pretrain_gnn requires dataset_path")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    train_split, val_split, test_split = _splits(config)
    train = _load_split(config.dataset_path, train_split)
    val = _load_split(config.dataset_path, val_split)
    test = _load_split(config.dataset_path, test_split)
    modulation_count = int(max(train["modulation_id"].max(), val["modulation_id"].max(), test["modulation_id"].max()) + 1)
    torch, model = _build_model(hidden_dim=_raw_int(config, "hidden_dim", 128), modulation_count=modulation_count)
    device = _device(config, torch)
    model.to(device)

    epochs = _raw_int(config, "epochs", 8)
    patience = _raw_int(config, "patience", 3)
    lr = _raw_float(config, "learning_rate", 8e-4)
    batch_size = int(config.batch_size)
    max_batches = int(config.max_batches)
    eval_max_batches = _raw_int(config, "eval_max_batches", 0)
    progress_every_batches = _raw_int(config, "progress_every_batches", 0)
    ranking_loss_weight = _raw_float(config, "ranking_loss_weight", 1.0)
    validation_ranking_weight = _raw_float(config, "validation_ranking_weight", 1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=_raw_float(config, "weight_decay", 1e-4))
    bce = torch.nn.BCEWithLogitsLoss()
    reg_weights = torch.as_tensor([1.0, 0.7, 1.0, 1.0, 0.5, 0.1], dtype=torch.float32, device=device)
    rng = np.random.default_rng(config.seed)

    history: list[dict[str, Any]] = []
    best_val = math.inf
    best_epoch = -1
    stale = 0
    best_path = run_path / "gnn_encoder_best.pt"
    edge_index = torch.as_tensor(train["edge_index"], dtype=torch.long, device=device)

    for epoch in range(1, epochs + 1):
        model.train()
        batches = _iter_batches(len(train["route_labels"]), batch_size, shuffle=True, rng=rng)
        if max_batches > 0:
            batches = batches[:max_batches]
        losses = []
        for batch_index, idx in enumerate(batches, start=1):
            batch = _batch_tensors(train, idx, device, torch)
            out = model(
                batch["node_features"],
                batch["link_features"],
                batch["global_features"],
                edge_index,
                batch["request_features"],
                batch["route_mask"],
                batch["route_basic"],
                batch["modulation_id"],
            )
            route_loss = (torch.nn.functional.smooth_l1_loss(out["route_regression"], batch["route_labels"], reduction="none") * reg_weights).mean()
            feasible_loss = bce(out["feasible_logit"], batch["feasible"])
            if ranking_loss_weight > 0.0:
                rank_loss = _ranking_loss(out["rank_score"], train["group_ids"][idx], batch["rank_targets"], torch)
            else:
                rank_loss = out["rank_score"].new_tensor(0.0)
            global_loss = (
                bce(out["block_logit"], batch["block_now"])
                + torch.nn.functional.smooth_l1_loss(out["num_feasible_norm"], batch["num_feasible_norm"])
                + torch.nn.functional.smooth_l1_loss(out["global_fragmentation"], batch["global_fragmentation"])
            )
            loss = route_loss + feasible_loss + float(ranking_loss_weight) * rank_loss + 0.5 * global_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if progress_every_batches > 0 and batch_index % progress_every_batches == 0:
                print(
                    json.dumps(
                        {
                            "event": "train_progress",
                            "epoch": epoch,
                            "batch": batch_index,
                            "batches": len(batches),
                            "train_loss_so_far": float(np.mean(losses)),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        val_metrics = _evaluate(model, val, batch_size, device, torch, max_batches=eval_max_batches)
        val_score = (
            val_metrics["route_regression_mae"]["delay_norm"]
            + val_metrics["route_regression_mae"]["energy_norm"]
            + val_metrics["route_regression_mae"]["route_fragmentation"]
            + val_metrics["route_regression_mae"]["c_route_max_norm"]
            + (1.0 - float(val_metrics["feasible"]["f1"]))
            + float(validation_ranking_weight) * (1.0 - float(val_metrics["ranking"]["top1_accuracy"] or 0.0))
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "val_score": float(val_score),
            "val": val_metrics,
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, sort_keys=True), flush=True)
        if val_score < best_val:
            best_val = float(val_score)
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "gnn_state_dict": model.gnn.state_dict(),
                    "epoch": epoch,
                    "val_score": best_val,
                    "config": config.resolved,
                    "route_regression_labels": list(ROUTE_REGRESSION_LABELS),
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= patience:
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = {
        "stage": "pretrain_gnn",
        "dataset_path": str(config.dataset_path),
        "device": device,
        "best_epoch": int(best_epoch),
        "best_checkpoint": str(best_path),
        "train_samples": int(len(train["route_labels"])),
        "eval_max_batches": int(eval_max_batches),
        "ranking_loss_weight": float(ranking_loss_weight),
        "validation_ranking_weight": float(validation_ranking_weight),
        "val": _evaluate(model, val, batch_size, device, torch, max_batches=eval_max_batches),
        "test": _evaluate(model, test, batch_size, device, torch, max_batches=eval_max_batches),
        "history": history,
    }
    _write_json(run_path / "metrics.json", final_metrics)
    return final_metrics
