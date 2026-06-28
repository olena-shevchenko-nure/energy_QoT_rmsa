from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised when optional torch is absent
    torch = None
    nn = None


def require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for the neural GNN+CNN+DQN solver path")
    return torch


if torch is not None:

    class MLP(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, dropout: float = 0.10, norm: bool = True) -> None:
            super().__init__()
            layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
            if norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.extend([nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim)])
            if norm:
                layers.append(nn.LayerNorm(out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    class EdgeStateBlock(nn.Module):
        def __init__(self, hidden_dim: int, dropout: float = 0.10) -> None:
            super().__init__()
            self.edge_mlp = MLP(hidden_dim * 4, hidden_dim * 2, hidden_dim, dropout=dropout, norm=False)
            self.node_mlp = MLP(hidden_dim * 6, hidden_dim * 2, hidden_dim, dropout=dropout, norm=False)
            self.global_mlp = MLP(hidden_dim * 6, hidden_dim * 2, hidden_dim, dropout=dropout, norm=False)
            self.edge_norm = nn.LayerNorm(hidden_dim)
            self.node_norm = nn.LayerNorm(hidden_dim)
            self.global_norm = nn.LayerNorm(hidden_dim)
            self.edge_attn = nn.Linear(hidden_dim, 1)

        def _node_aggregate(self, he: torch.Tensor, edge_index: torch.Tensor, node_count: int) -> torch.Tensor:
            batch, edge_count, hidden = he.shape
            src = edge_index[0].long()
            dst = edge_index[1].long()
            zeros = he.new_zeros(batch, node_count, hidden)
            count_in = he.new_zeros(batch, node_count, 1)
            count_out = he.new_zeros(batch, node_count, 1)
            mean_in = zeros.clone()
            mean_out = zeros.clone()
            mean_in.index_add_(1, dst, he)
            mean_out.index_add_(1, src, he)
            ones = he.new_ones(batch, edge_count, 1)
            count_in.index_add_(1, dst, ones)
            count_out.index_add_(1, src, ones)
            mean_in = mean_in / count_in.clamp_min(1.0)
            mean_out = mean_out / count_out.clamp_min(1.0)

            max_in = he.new_full((batch, node_count, hidden), -1e9)
            max_out = he.new_full((batch, node_count, hidden), -1e9)
            dst_index = dst.view(1, edge_count, 1).expand(batch, edge_count, hidden)
            src_index = src.view(1, edge_count, 1).expand(batch, edge_count, hidden)
            max_in = max_in.scatter_reduce(1, dst_index, he, reduce="amax", include_self=True)
            max_out = max_out.scatter_reduce(1, src_index, he, reduce="amax", include_self=True)
            max_in = torch.where(max_in < -1e8, torch.zeros_like(max_in), max_in)
            max_out = torch.where(max_out < -1e8, torch.zeros_like(max_out), max_out)
            return torch.cat([mean_in, max_in, mean_out, max_out], dim=-1)

        def forward(
            self,
            hv: torch.Tensor,
            he: torch.Tensor,
            hg: torch.Tensor,
            edge_index: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            src = edge_index[0].long()
            dst = edge_index[1].long()
            edge_global = hg[:, None, :].expand(-1, he.shape[1], -1)
            edge_msg = self.edge_mlp(torch.cat([hv[:, src], hv[:, dst], he, edge_global], dim=-1))
            he = self.edge_norm(he + edge_msg)

            node_agg = self._node_aggregate(he, edge_index, hv.shape[1])
            node_global = hg[:, None, :].expand(-1, hv.shape[1], -1)
            node_msg = self.node_mlp(torch.cat([hv, node_agg, node_global], dim=-1))
            hv = self.node_norm(hv + node_msg)

            edge_attn = torch.softmax(self.edge_attn(he), dim=1)
            pooled_edges = torch.cat([he.mean(dim=1), he.max(dim=1).values, (edge_attn * he).sum(dim=1)], dim=-1)
            pooled_nodes = torch.cat([hv.mean(dim=1), hv.max(dim=1).values], dim=-1)
            global_msg = self.global_mlp(torch.cat([hg, pooled_edges, pooled_nodes], dim=-1))
            hg = self.global_norm(hg + global_msg)
            return hv, he, hg


    class EdgeStateGNN(nn.Module):
        def __init__(self, node_dim: int = 4, link_dim: int = 8, global_dim: int = 8, hidden_dim: int = 128, layers: int = 3) -> None:
            super().__init__()
            self.node_in = MLP(node_dim, hidden_dim, hidden_dim)
            self.link_in = MLP(link_dim, hidden_dim, hidden_dim)
            self.global_in = MLP(global_dim, hidden_dim, hidden_dim)
            self.blocks = nn.ModuleList([EdgeStateBlock(hidden_dim) for _ in range(layers)])
            self.link_out = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.global_out = MLP(hidden_dim * 5, hidden_dim * 2, hidden_dim)

        def forward(
            self,
            node_features: torch.Tensor,
            link_features: torch.Tensor,
            global_features: torch.Tensor,
            edge_index: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            hv = self.node_in(node_features)
            he = self.link_in(link_features)
            hg = self.global_in(global_features)
            for block in self.blocks:
                hv, he, hg = block(hv, he, hg, edge_index)
            link_embeddings = self.link_out(he)
            global_embeddings = self.global_out(
                torch.cat([hg, he.mean(dim=1), he.max(dim=1).values, hv.mean(dim=1), hv.max(dim=1).values], dim=-1)
            )
            return global_embeddings, link_embeddings


    class ResidualConvBlock1D(nn.Module):
        def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.10) -> None:
            super().__init__()
            padding = dilation * (kernel_size - 1) // 2
            self.net = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
                nn.GroupNorm(8, channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
                nn.GroupNorm(8, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.gelu(x + self.net(x))


    class SlotCNNEncoder(nn.Module):
        def __init__(self, channels: int = 6, hidden_channels: int = 64, out_dim: int = 128) -> None:
            super().__init__()
            self.slot_net = nn.Sequential(
                nn.Conv1d(channels, hidden_channels, kernel_size=7, padding=3),
                nn.GroupNorm(8, hidden_channels),
                nn.GELU(),
                ResidualConvBlock1D(hidden_channels, kernel_size=5, dilation=1),
                ResidualConvBlock1D(hidden_channels, kernel_size=5, dilation=2),
                ResidualConvBlock1D(hidden_channels, kernel_size=3, dilation=4),
                ResidualConvBlock1D(hidden_channels, kernel_size=3, dilation=8),
                nn.Conv1d(hidden_channels, out_dim, kernel_size=1),
                nn.GroupNorm(8, out_dim),
                nn.GELU(),
            )
            self.pool_mlp = nn.Sequential(
                nn.Linear(out_dim * 8 + 5, 256),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(256, out_dim),
                nn.LayerNorm(out_dim),
            )

        def _pool_mask(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            weights = mask.to(dtype=h.dtype).unsqueeze(-1)
            counts = weights.sum(dim=1)
            mean = (h * weights).sum(dim=1) / counts.clamp_min(1.0)
            masked = h.masked_fill(~mask.unsqueeze(-1), -1e9)
            maximum = masked.max(dim=1).values
            maximum = torch.where(counts > 0.0, maximum, torch.zeros_like(maximum))
            return mean, maximum

        def forward(self, x: torch.Tensor, block_bounds: torch.Tensor) -> torch.Tensor:
            _, _, slots = x.shape
            h = self.slot_net(x).transpose(1, 2)
            start = block_bounds[:, 0].long().clamp(0, max(slots - 1, 0))
            width = block_bounds[:, 1].long().clamp_min(1)
            end = (start + width).clamp(max=slots)
            left_start = (start - 16).clamp_min(0)
            right_end = (end + 16).clamp(max=slots)
            positions = torch.arange(slots, device=x.device).unsqueeze(0)

            block_mask = (positions >= start.unsqueeze(1)) & (positions < end.unsqueeze(1))
            left_mask = (positions >= left_start.unsqueeze(1)) & (positions < start.unsqueeze(1))
            right_mask = (positions >= end.unsqueeze(1)) & (positions < right_end.unsqueeze(1))
            block_mean, block_max = self._pool_mask(h, block_mask)
            left_mean, left_max = self._pool_mask(h, left_mask)
            right_mean, right_max = self._pool_mask(h, right_mask)
            global_mean = h.mean(dim=1)
            global_max = h.max(dim=1).values
            scalars = torch.stack(
                [
                    start.to(dtype=h.dtype) / max(slots - 1, 1),
                    width.to(dtype=h.dtype) / max(slots, 1),
                    (start - left_start).to(dtype=h.dtype) / max(slots, 1),
                    (right_end - end).to(dtype=h.dtype) / max(slots, 1),
                    (end - start).to(dtype=h.dtype) / max(slots, 1),
                ],
                dim=1,
            )
            pooled = torch.cat(
                [
                    block_mean,
                    block_max,
                    left_mean,
                    left_max,
                    right_mean,
                    right_max,
                    global_mean,
                    global_max,
                    scalars,
                ],
                dim=1,
            )
            return self.pool_mlp(pooled)


    class RequestEncoder(nn.Module):
        def __init__(self, in_dim: int = 3, out_dim: int = 64) -> None:
            super().__init__()
            self.net = MLP(in_dim, 64, out_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    class ActionFeatureEncoder(nn.Module):
        def __init__(self, in_dim: int, out_dim: int = 64) -> None:
            super().__init__()
            self.net = MLP(in_dim, 128, out_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    class CandidateQNetwork(nn.Module):
        """Candidate-wise GNN+CNN+DQN scorer with masked selection outside the head."""

        def __init__(self, action_feature_dim: int, hidden_dim: int = 128) -> None:
            super().__init__()
            self.gnn = EdgeStateGNN(hidden_dim=hidden_dim)
            self.slot_cnn = SlotCNNEncoder(out_dim=hidden_dim)
            self.route_pool = nn.Sequential(
                nn.Linear(hidden_dim * 2 + 2, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(256, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            self.request_encoder = RequestEncoder(out_dim=64)
            self.action_encoder = ActionFeatureEncoder(action_feature_dim, out_dim=64)
            fusion_dim = hidden_dim + hidden_dim + hidden_dim + 64 + 64
            self.q_head = nn.Sequential(
                nn.Linear(fusion_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, 1),
            )

        def _route_embeddings(
            self,
            link_embeddings: torch.Tensor,
            route_link_mask: torch.Tensor,
            route_basic_features: torch.Tensor,
        ) -> torch.Tensor:
            mask = route_link_mask.unsqueeze(-1)
            denom = mask.sum(dim=2).clamp_min(1.0)
            mean_pool = (link_embeddings[:, None, :, :] * mask).sum(dim=2) / denom
            masked_links = link_embeddings[:, None, :, :].masked_fill(~route_link_mask.unsqueeze(-1).bool(), -1e9)
            max_pool = masked_links.max(dim=2).values
            max_pool = torch.where(max_pool < -1e8, torch.zeros_like(max_pool), max_pool)
            return self.route_pool(torch.cat([mean_pool, max_pool, route_basic_features], dim=-1))

        def forward(
            self,
            *,
            node_features: torch.Tensor,
            link_features: torch.Tensor,
            global_features: torch.Tensor,
            edge_index: torch.Tensor,
            request_features: torch.Tensor,
            spectrum_tensors: torch.Tensor,
            action_features: torch.Tensor,
            route_link_mask: torch.Tensor,
            route_basic_features: torch.Tensor,
            block_bounds: torch.Tensor,
        ) -> torch.Tensor:
            batch, n_max = action_features.shape[:2]
            h_global, h_links = self.gnn(node_features, link_features, global_features, edge_index)
            h_route = self._route_embeddings(h_links, route_link_mask, route_basic_features)
            h_req = self.request_encoder(request_features)[:, None, :].expand(-1, n_max, -1)
            h_action = self.action_encoder(action_features)
            h_block = self.slot_cnn(
                spectrum_tensors.reshape(batch * n_max, spectrum_tensors.shape[2], spectrum_tensors.shape[3]),
                block_bounds.reshape(batch * n_max, 2),
            ).reshape(batch, n_max, -1)
            h_global_rep = h_global[:, None, :].expand(-1, n_max, -1)
            fused = torch.cat([h_global_rep, h_route, h_block, h_req, h_action], dim=-1)
            return self.q_head(fused).squeeze(-1)


    class GnnCnnA3CNetwork(nn.Module):
        """Full GNN+CNN candidate actor-critic with the same encoders as CandidateQNetwork."""

        def __init__(self, action_feature_dim: int, hidden_dim: int = 128) -> None:
            super().__init__()
            self.gnn = EdgeStateGNN(hidden_dim=hidden_dim)
            self.slot_cnn = SlotCNNEncoder(out_dim=hidden_dim)
            self.route_pool = nn.Sequential(
                nn.Linear(hidden_dim * 2 + 2, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(256, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            self.request_encoder = RequestEncoder(out_dim=64)
            self.action_encoder = ActionFeatureEncoder(action_feature_dim, out_dim=64)
            fusion_dim = hidden_dim + hidden_dim + hidden_dim + 64 + 64
            self.policy_head = nn.Sequential(
                nn.Linear(fusion_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, 1),
            )
            self.value_head = nn.Sequential(
                nn.Linear(fusion_dim * 2, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, 1),
            )

        def _route_embeddings(
            self,
            link_embeddings: torch.Tensor,
            route_link_mask: torch.Tensor,
            route_basic_features: torch.Tensor,
        ) -> torch.Tensor:
            mask = route_link_mask.unsqueeze(-1)
            denom = mask.sum(dim=2).clamp_min(1.0)
            mean_pool = (link_embeddings[:, None, :, :] * mask).sum(dim=2) / denom
            masked_links = link_embeddings[:, None, :, :].masked_fill(~route_link_mask.unsqueeze(-1).bool(), -1e9)
            max_pool = masked_links.max(dim=2).values
            max_pool = torch.where(max_pool < -1e8, torch.zeros_like(max_pool), max_pool)
            return self.route_pool(torch.cat([mean_pool, max_pool, route_basic_features], dim=-1))

        def forward(
            self,
            *,
            node_features: torch.Tensor,
            link_features: torch.Tensor,
            global_features: torch.Tensor,
            edge_index: torch.Tensor,
            request_features: torch.Tensor,
            spectrum_tensors: torch.Tensor,
            action_features: torch.Tensor,
            route_link_mask: torch.Tensor,
            route_basic_features: torch.Tensor,
            block_bounds: torch.Tensor,
            candidate_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            batch, n_max = action_features.shape[:2]
            h_global, h_links = self.gnn(node_features, link_features, global_features, edge_index)
            h_route = self._route_embeddings(h_links, route_link_mask, route_basic_features)
            h_req = self.request_encoder(request_features)[:, None, :].expand(-1, n_max, -1)
            h_action = self.action_encoder(action_features)
            h_block = self.slot_cnn(
                spectrum_tensors.reshape(batch * n_max, spectrum_tensors.shape[2], spectrum_tensors.shape[3]),
                block_bounds.reshape(batch * n_max, 2),
            ).reshape(batch, n_max, -1)
            h_global_rep = h_global[:, None, :].expand(-1, n_max, -1)
            fused = torch.cat([h_global_rep, h_route, h_block, h_req, h_action], dim=-1)
            logits = self.policy_head(fused).squeeze(-1)

            if candidate_mask is None:
                mean_pool = fused.mean(dim=1)
                max_pool = fused.max(dim=1).values
            else:
                mask = candidate_mask.to(dtype=fused.dtype).unsqueeze(-1)
                denom = mask.sum(dim=1).clamp_min(1.0)
                mean_pool = (fused * mask).sum(dim=1) / denom
                masked = fused.masked_fill(~candidate_mask.bool().unsqueeze(-1), -1e9)
                max_pool = masked.max(dim=1).values
                max_pool = torch.where(max_pool < -1e8, torch.zeros_like(max_pool), max_pool)
            value = self.value_head(torch.cat([mean_pool, max_pool], dim=-1)).squeeze(-1)
            return logits, value


    class XlronGraphTransformerPpoNetwork(nn.Module):
        """XLRON-inspired actor-critic over the fixed Top-N RMSA candidate surface.

        The adaptation keeps XLRON's link-token transformer shape and path pooling
        idea, while matching this project by scoring already-feasible Top-N
        route/modulation/spectrum candidates instead of emitting raw RMSA actions.
        """

        def __init__(
            self,
            *,
            action_feature_dim: int,
            link_feature_dim: int = 8,
            global_feature_dim: int = 8,
            request_feature_dim: int = 3,
            embedding_dim: int = 128,
            num_layers: int = 2,
            num_heads: int = 8,
            dropout: float = 0.05,
            position_dim: int = 8,
            architecture: str = "link_transformer",
            spectrum_channels: int = 6,
            route_basic_dim: int = 2,
            candidate_transformer_layers: int = 0,
            candidate_transformer_heads: int = 4,
            enable_spectrum_branch: bool | None = None,
            enable_candidate_attention: bool | None = None,
            enable_base_relative_branch: bool | None = None,
            enable_auxiliary_heads: bool = False,
        ) -> None:
            super().__init__()
            self.action_feature_dim = int(action_feature_dim)
            self.link_feature_dim = int(link_feature_dim)
            self.global_feature_dim = int(global_feature_dim)
            self.request_feature_dim = int(request_feature_dim)
            self.embedding_dim = int(embedding_dim)
            self.position_dim = int(position_dim)
            self.architecture = str(architecture or "link_transformer").strip().lower()
            if self.architecture not in {"link_transformer", "full"}:
                raise ValueError(f"Unsupported XLRON architecture: {architecture}")
            self.spectrum_channels = int(spectrum_channels)
            self.route_basic_dim = int(route_basic_dim)
            self.candidate_transformer_layers = int(candidate_transformer_layers)
            self.candidate_transformer_heads = int(candidate_transformer_heads)
            full_default = self.architecture == "full"
            self.enable_spectrum_branch = full_default if enable_spectrum_branch is None else bool(enable_spectrum_branch)
            self.enable_candidate_attention = full_default if enable_candidate_attention is None else bool(enable_candidate_attention)
            self.enable_base_relative_branch = (
                full_default if enable_base_relative_branch is None else bool(enable_base_relative_branch)
            )
            self.enable_auxiliary_heads = bool(enable_auxiliary_heads)

            context_dim = self.global_feature_dim + self.request_feature_dim
            self.link_input = nn.Linear(self.link_feature_dim + context_dim, self.embedding_dim)
            self.position_input = nn.Linear(self.position_dim, self.embedding_dim) if self.position_dim > 0 else None
            self.input_norm = nn.LayerNorm(self.embedding_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=int(num_heads),
                dim_feedforward=self.embedding_dim * 4,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
            self.action_encoder = MLP(self.action_feature_dim, self.embedding_dim, self.embedding_dim, dropout=dropout)

            actor_dim = self.embedding_dim * 4 + context_dim
            self.policy_head = nn.Sequential(
                nn.Linear(actor_dim, self.embedding_dim * 2),
                nn.LayerNorm(self.embedding_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.embedding_dim * 2, self.embedding_dim),
                nn.GELU(),
                nn.Linear(self.embedding_dim, 1),
            )
            self.value_query = nn.Parameter(torch.empty(self.embedding_dim))
            nn.init.normal_(self.value_query, mean=0.0, std=self.embedding_dim**-0.5)
            self.value_head = nn.Sequential(
                nn.Linear(self.embedding_dim + context_dim, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.embedding_dim, 1),
            )

            if self.architecture == "full":
                self.context_encoder = MLP(context_dim, self.embedding_dim, self.embedding_dim, dropout=dropout)
                self.full_route_project = MLP(
                    self.embedding_dim * 3 + self.route_basic_dim,
                    self.embedding_dim * 2,
                    self.embedding_dim,
                    dropout=dropout,
                )
                if self.enable_spectrum_branch:
                    self.spectrum_encoder = SlotCNNEncoder(channels=self.spectrum_channels, out_dim=self.embedding_dim)
                base_relative_dim = self.action_feature_dim * 3 + self.route_basic_dim * 3 + 2
                if self.enable_base_relative_branch:
                    self.base_relative_encoder = MLP(
                        base_relative_dim,
                        self.embedding_dim * 2,
                        self.embedding_dim,
                        dropout=dropout,
                    )
                full_parts = 4
                full_parts += int(self.enable_spectrum_branch)
                full_parts += int(self.enable_base_relative_branch)
                self.full_candidate_fusion = MLP(
                    self.embedding_dim * full_parts,
                    self.embedding_dim * 2,
                    self.embedding_dim,
                    dropout=dropout,
                )
                if self.enable_candidate_attention and self.candidate_transformer_layers > 0:
                    candidate_layer = nn.TransformerEncoderLayer(
                        d_model=self.embedding_dim,
                        nhead=max(1, self.candidate_transformer_heads),
                        dim_feedforward=self.embedding_dim * 4,
                        dropout=float(dropout),
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    self.candidate_transformer = nn.TransformerEncoder(
                        candidate_layer,
                        num_layers=self.candidate_transformer_layers,
                    )
                else:
                    self.candidate_transformer = None
                self.full_policy_head = nn.Sequential(
                    nn.Linear(self.embedding_dim, self.embedding_dim),
                    nn.LayerNorm(self.embedding_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.embedding_dim, 1),
                )
                self.full_value_head = nn.Sequential(
                    nn.Linear(self.embedding_dim * 4, self.embedding_dim * 2),
                    nn.LayerNorm(self.embedding_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.embedding_dim * 2, self.embedding_dim),
                    nn.GELU(),
                    nn.Linear(self.embedding_dim, 1),
                )
                self.advantage_head = nn.Sequential(
                    nn.Linear(self.embedding_dim, self.embedding_dim),
                    nn.GELU(),
                    nn.Linear(self.embedding_dim, 1),
                )
                self.win_head = nn.Sequential(
                    nn.Linear(self.embedding_dim, self.embedding_dim),
                    nn.GELU(),
                    nn.Linear(self.embedding_dim, 1),
                )
                self.loss_head = nn.Sequential(
                    nn.Linear(self.embedding_dim, self.embedding_dim),
                    nn.GELU(),
                    nn.Linear(self.embedding_dim, 1),
                )

        def _line_graph_laplacian_pe(
            self,
            edge_index: torch.Tensor,
            edge_count: int,
            *,
            device: torch.device,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            if self.position_dim <= 0:
                return torch.zeros(edge_count, 0, dtype=dtype, device=device)
            if edge_count <= 1:
                return torch.zeros(edge_count, self.position_dim, dtype=dtype, device=device)

            edge_index = edge_index.to(device=device)
            src = edge_index[0].long()
            dst = edge_index[1].long()
            adjacent = (
                (src[:, None] == src[None, :])
                | (src[:, None] == dst[None, :])
                | (dst[:, None] == src[None, :])
                | (dst[:, None] == dst[None, :])
            )
            adjacent.fill_diagonal_(False)
            adjacency = adjacent.to(dtype=torch.float32)
            degree = adjacency.sum(dim=1)
            inv_sqrt = degree.clamp_min(1.0).rsqrt()
            laplacian = torch.eye(edge_count, dtype=torch.float32, device=device) - (
                inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
            )
            eigvals, eigvecs = torch.linalg.eigh(laplacian)
            del eigvals
            take = min(self.position_dim, max(edge_count - 1, 1))
            pe = eigvecs[:, 1 : 1 + take] if edge_count > 1 else eigvecs[:, :take]
            if pe.numel():
                max_abs_idx = pe.abs().argmax(dim=0)
                signs = torch.sign(pe[max_abs_idx, torch.arange(pe.shape[1], device=device)])
                signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                pe = pe * signs
            if take < self.position_dim:
                pad = torch.zeros(edge_count, self.position_dim - take, dtype=pe.dtype, device=device)
                pe = torch.cat([pe, pad], dim=1)
            return pe.to(dtype=dtype)

        def _route_pool(self, link_tokens: torch.Tensor, route_link_mask: torch.Tensor) -> torch.Tensor:
            mask = route_link_mask.to(dtype=torch.bool)
            weights = mask.to(dtype=link_tokens.dtype).unsqueeze(-1)
            denom = weights.sum(dim=2).clamp_min(1.0)
            mean_pool = (link_tokens[:, None, :, :] * weights).sum(dim=2) / denom

            expanded = link_tokens[:, None, :, :]
            max_pool = expanded.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=2).values
            min_pool = expanded.masked_fill(~mask.unsqueeze(-1), 1e9).min(dim=2).values
            has_links = mask.any(dim=2, keepdim=True)
            max_pool = torch.where(has_links, max_pool, torch.zeros_like(max_pool))
            min_pool = torch.where(has_links, min_pool, torch.zeros_like(min_pool))
            return torch.cat([min_pool, mean_pool, max_pool], dim=-1)

        def _graph_pool(self, link_tokens: torch.Tensor) -> torch.Tensor:
            attention_logits = (link_tokens * self.value_query).sum(dim=-1) * (self.embedding_dim**-0.5)
            attention = torch.softmax(attention_logits, dim=1)
            return (attention.unsqueeze(-1) * link_tokens).sum(dim=1)

        def _candidate_pool(
            self,
            candidate_tokens: torch.Tensor,
            candidate_mask: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if candidate_mask is None:
                return candidate_tokens.mean(dim=1), candidate_tokens.max(dim=1).values
            mask = candidate_mask.to(dtype=torch.bool)
            weights = mask.to(dtype=candidate_tokens.dtype).unsqueeze(-1)
            denom = weights.sum(dim=1).clamp_min(1.0)
            mean_pool = (candidate_tokens * weights).sum(dim=1) / denom
            masked = candidate_tokens.masked_fill(~mask.unsqueeze(-1), -1.0e9)
            max_pool = masked.max(dim=1).values
            has_candidates = mask.any(dim=1, keepdim=True)
            max_pool = torch.where(has_candidates, max_pool, torch.zeros_like(max_pool))
            return mean_pool, max_pool

        def _base_relative_features(
            self,
            *,
            action_features: torch.Tensor,
            route_basic_features: torch.Tensor,
        ) -> torch.Tensor:
            batch, n_max = action_features.shape[:2]
            base_action = action_features[:, :1, :].expand(-1, n_max, -1)
            action_delta = action_features - base_action
            base_route = route_basic_features[:, :1, :].expand(-1, n_max, -1)
            route_delta = route_basic_features - base_route
            topn = torch.arange(n_max, dtype=action_features.dtype, device=action_features.device)
            topn = (topn / max(n_max - 1, 1)).view(1, n_max, 1).expand(batch, -1, -1)
            is_base = torch.zeros((batch, n_max, 1), dtype=action_features.dtype, device=action_features.device)
            if n_max > 0:
                is_base[:, 0, 0] = 1.0
            return torch.cat(
                [
                    action_features,
                    base_action,
                    action_delta,
                    route_basic_features,
                    base_route,
                    route_delta,
                    topn,
                    is_base,
                ],
                dim=-1,
            )

        def forward(
            self,
            *,
            link_features: torch.Tensor,
            edge_index: torch.Tensor,
            global_features: torch.Tensor,
            request_features: torch.Tensor,
            action_features: torch.Tensor,
            route_link_mask: torch.Tensor,
            spectrum_tensors: torch.Tensor | None = None,
            route_basic_features: torch.Tensor | None = None,
            block_bounds: torch.Tensor | None = None,
            candidate_mask: torch.Tensor | None = None,
            return_aux: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
            batch, edge_count = link_features.shape[:2]
            n_max = action_features.shape[1]
            context = torch.cat([global_features, request_features], dim=1)
            token_context = context[:, None, :].expand(batch, edge_count, -1)
            tokens = self.link_input(torch.cat([link_features, token_context], dim=-1))
            if self.position_input is not None:
                pe = self._line_graph_laplacian_pe(
                    edge_index,
                    edge_count,
                    device=link_features.device,
                    dtype=link_features.dtype,
                )
                tokens = tokens + self.position_input(pe)[None, :, :]
            tokens = self.input_norm(tokens)
            link_tokens = self.transformer(tokens)

            route_tokens = self._route_pool(link_tokens, route_link_mask)
            action_tokens = self.action_encoder(action_features)
            candidate_context = context[:, None, :].expand(batch, n_max, -1)
            graph_token = self._graph_pool(link_tokens)
            if self.architecture != "full":
                logits = self.policy_head(torch.cat([route_tokens, action_tokens, candidate_context], dim=-1)).squeeze(-1)
                value = self.value_head(torch.cat([graph_token, context], dim=1)).squeeze(-1)
                return logits, value

            if route_basic_features is None:
                route_basic_features = action_features.new_zeros(batch, n_max, self.route_basic_dim)
            if route_basic_features.shape[-1] != self.route_basic_dim:
                raise ValueError(
                    f"route_basic_features last dimension must be {self.route_basic_dim}, "
                    f"got {route_basic_features.shape[-1]}"
                )
            context_token = self.context_encoder(context)
            h_route = self.full_route_project(torch.cat([route_tokens, route_basic_features], dim=-1))
            graph_rep = graph_token[:, None, :].expand(-1, n_max, -1)
            context_rep = context_token[:, None, :].expand(-1, n_max, -1)
            parts = [graph_rep, h_route, action_tokens, context_rep]

            if self.enable_spectrum_branch:
                if spectrum_tensors is None or block_bounds is None:
                    h_spectrum = action_tokens.new_zeros(batch, n_max, self.embedding_dim)
                else:
                    h_spectrum = self.spectrum_encoder(
                        spectrum_tensors.reshape(batch * n_max, spectrum_tensors.shape[2], spectrum_tensors.shape[3]),
                        block_bounds.reshape(batch * n_max, 2),
                    ).reshape(batch, n_max, -1)
                parts.append(h_spectrum)

            if self.enable_base_relative_branch:
                base_relative_features = self._base_relative_features(
                    action_features=action_features,
                    route_basic_features=route_basic_features,
                )
                parts.append(self.base_relative_encoder(base_relative_features))

            candidate_tokens = self.full_candidate_fusion(torch.cat(parts, dim=-1))
            if self.candidate_transformer is not None:
                padding_mask = None
                if candidate_mask is not None:
                    padding_mask = ~candidate_mask.to(dtype=torch.bool)
                    if bool(padding_mask.all(dim=1).any()):
                        padding_mask = padding_mask.clone()
                        padding_mask[padding_mask.all(dim=1)] = False
                candidate_tokens = self.candidate_transformer(candidate_tokens, src_key_padding_mask=padding_mask)

            logits = self.full_policy_head(candidate_tokens).squeeze(-1)
            mean_pool, max_pool = self._candidate_pool(candidate_tokens, candidate_mask)
            value = self.full_value_head(torch.cat([graph_token, mean_pool, max_pool, context_token], dim=1)).squeeze(-1)
            if return_aux:
                aux = {
                    "advantage_over_base": self.advantage_head(candidate_tokens).squeeze(-1),
                    "win_logits": self.win_head(candidate_tokens).squeeze(-1),
                    "loss_logits": self.loss_head(candidate_tokens).squeeze(-1),
                }
                return logits, value, aux
            return logits, value


    class DeepRmsaA3CNetwork(nn.Module):
        """DeepRMSA-style actor-critic over the fixed Top-N candidate surface."""

        def __init__(
            self,
            *,
            n_max: int,
            candidate_feature_dim: int,
            context_feature_dim: int,
            hidden_dim: int = 128,
            layers: int = 5,
            dropout: float = 0.05,
        ) -> None:
            super().__init__()
            self.n_max = int(n_max)
            self.candidate_feature_dim = int(candidate_feature_dim)
            self.context_feature_dim = int(context_feature_dim)
            input_dim = self.n_max * self.candidate_feature_dim + self.context_feature_dim

            def body() -> nn.Sequential:
                modules: list[nn.Module] = []
                current_dim = input_dim
                for _ in range(max(1, int(layers))):
                    modules.extend([nn.Linear(current_dim, hidden_dim), nn.ELU(), nn.Dropout(dropout)])
                    current_dim = hidden_dim
                return nn.Sequential(*modules)

            self.policy_body = body()
            self.value_body = body()
            self.policy_head = nn.Linear(hidden_dim, self.n_max)
            self.value_head = nn.Linear(hidden_dim, 1)

        def forward(self, candidate_features: torch.Tensor, context_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            flat = torch.cat([candidate_features.reshape(candidate_features.shape[0], -1), context_features], dim=1)
            logits = self.policy_head(self.policy_body(flat))
            value = self.value_head(self.value_body(flat)).squeeze(-1)
            return logits, value

else:

    CandidateQNetwork = None
    GnnCnnA3CNetwork = None
    XlronGraphTransformerPpoNetwork = None
    DeepRmsaA3CNetwork = None
