"""Stronger graph encoders for vascular skeleton representation learning.

All encoders expose the same call contract as :class:`GINEEncoder`: they accept
either a padded batch produced by ``angiotda.data.dataset.collate_fn`` or a
flattened PyG batch.  The implementations stay sparse (linear in the number of
edges), which matters because the graph corpus contains a small number of very
large skeletons.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv,
    PNAConv,
    TransformerConv,
    global_max_pool,
    global_mean_pool,
)


# In-degree histogram recomputed over the 2,560 non-control v2 skeleton graphs
# used by GraphDatasetV2. PNA uses it only to calibrate its degree scalers;
# callers training on another corpus should pass that corpus's histogram.
DEFAULT_VASCULAR_DEGREE_HISTOGRAM = (0, 603_661, 1, 408_563, 3_479)


class SparseGraphEncoder(nn.Module, ABC):
    """Shared padded/PyG batching and graph-level readout implementation."""

    def __init__(self, node_feat_dim: int, hidden: int, embed_dim: int):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden = hidden
        self.embed_dim = embed_dim
        self.node_emb = nn.Linear(node_feat_dim, hidden)
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index,
        edge_attr,
        node_mask: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if x.dim() == 3:
            if node_mask is None:
                raise ValueError("node_mask is required for a padded graph batch")
            x, edge_index, edge_attr, batch = self._flatten_padded(
                x, edge_index, edge_attr, node_mask
            )
        elif x.dim() != 2:
            raise ValueError(f"expected x with 2 or 3 dimensions, got {x.shape}")

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return self._forward_pyg(x, edge_index, edge_attr, batch)

    def _flatten_padded(self, x, edge_index_list, edge_attr_list, node_mask):
        batch_size = x.shape[0]
        if len(edge_index_list) != batch_size or len(edge_attr_list) != batch_size:
            raise ValueError("edge lists must contain one tensor per padded graph")

        flat_x, flat_ei, flat_ea, batches = [], [], [], []
        offset = 0
        edge_dim = None
        for graph_id in range(batch_size):
            n_real = int(node_mask[graph_id].sum().item())
            if n_real == 0:
                # Retain an empty graph in the graph-level output using one
                # masked zero node. Real datasets currently have >=14 nodes.
                graph_x = x.new_zeros((1, x.shape[-1]))
                n_packed = 1
            else:
                graph_x = x[graph_id, :n_real]
                n_packed = n_real

            ei = edge_index_list[graph_id].to(x.device)
            ea = edge_attr_list[graph_id].to(x.device)
            if ei.dim() != 2 or ei.shape[0] != 2:
                raise ValueError(f"edge_index must have shape (2, E), got {ei.shape}")
            if ea.dim() != 2 or ea.shape[0] != ei.shape[1]:
                raise ValueError("edge_attr must have shape (E, edge_feat_dim)")
            edge_dim = ea.shape[1] if edge_dim is None else edge_dim
            if ea.shape[1] != edge_dim:
                raise ValueError("all graphs in a batch must share edge feature size")

            valid = (ei[0] < n_real) & (ei[1] < n_real)
            flat_x.append(graph_x)
            flat_ei.append(ei[:, valid] + offset)
            flat_ea.append(ea[valid])
            batches.append(
                torch.full((n_packed,), graph_id, dtype=torch.long, device=x.device)
            )
            offset += n_packed

        return (
            torch.cat(flat_x, dim=0),
            torch.cat(flat_ei, dim=1),
            torch.cat(flat_ea, dim=0),
            torch.cat(batches, dim=0),
        )

    def _forward_pyg(self, x, edge_index, edge_attr, batch):
        h = self.node_emb(x)
        h = self.encode_nodes(h, edge_index, edge_attr)
        pooled = torch.cat(
            [global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1
        )
        return self.readout(pooled)

    @abstractmethod
    def encode_nodes(self, h, edge_index, edge_attr):
        """Return node embeddings after sparse message passing."""


class PNAEncoder(SparseGraphEncoder):
    """Principal Neighbourhood Aggregation with edge-conditioned messages.

    PNA combines mean/min/max/std aggregation and degree-dependent scalers. It
    is well matched to skeletons because tips, paths, and junctions have sharply
    different degrees, while richer aggregation can distinguish neighborhoods
    that a single sum aggregator maps together.
    """

    def __init__(
        self,
        node_feat_dim: int = 7,
        edge_feat_dim: int = 8,
        embed_dim: int = 128,
        n_layers: int = 4,
        hidden: int = 256,
        degree_histogram=None,
        dropout: float = 0.1,
        feature_mode: str = "node_only",
    ):
        super().__init__(node_feat_dim, hidden, embed_dim)
        del feature_mode
        degree_histogram = (
            DEFAULT_VASCULAR_DEGREE_HISTOGRAM
            if degree_histogram is None
            else degree_histogram
        )
        deg = torch.as_tensor(degree_histogram, dtype=torch.long)
        if deg.dim() != 1 or deg.numel() < 2 or int(deg.sum()) == 0:
            raise ValueError("degree_histogram must be a non-empty 1D count vector")
        self.register_buffer("degree_histogram", deg)
        self.dropout = dropout
        self.convs = nn.ModuleList(
            [
                PNAConv(
                    hidden,
                    hidden,
                    aggregators=["mean", "min", "max", "std"],
                    scalers=["identity", "amplification", "attenuation"],
                    deg=self.degree_histogram,
                    edge_dim=edge_feat_dim,
                    towers=4 if hidden % 4 == 0 else 1,
                    pre_layers=2,
                    post_layers=2,
                    divide_input=True if hidden % 4 == 0 else False,
                )
                for _ in range(n_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])

    def encode_nodes(self, h, edge_index, edge_attr):
        for conv, norm in zip(self.convs, self.norms):
            update = F.gelu(norm(conv(h, edge_index, edge_attr)))
            h = h + F.dropout(update, p=self.dropout, training=self.training)
        return h


class GATv2Encoder(SparseGraphEncoder):
    """Edge-aware GATv2 with dynamic, multi-head neighborhood attention.

    GATv2 fixes the static-attention limitation of the original GAT. This lets
    a branch or junction attend differently according to the receiving node's
    morphology and the length/orientation features of each incident segment.
    """

    def __init__(
        self,
        node_feat_dim: int = 7,
        edge_feat_dim: int = 8,
        embed_dim: int = 128,
        n_layers: int = 4,
        hidden: int = 256,
        heads: int = 4,
        dropout: float = 0.1,
        feature_mode: str = "node_only",
    ):
        super().__init__(node_feat_dim, hidden, embed_dim)
        del feature_mode
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.dropout = dropout
        self.convs = nn.ModuleList(
            [
                GATv2Conv(
                    hidden,
                    hidden // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_feat_dim,
                    add_self_loops=True,
                    fill_value="mean",
                    residual=False,
                )
                for _ in range(n_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])

    def encode_nodes(self, h, edge_index, edge_attr):
        for conv, norm in zip(self.convs, self.norms):
            update = F.gelu(norm(conv(h, edge_index, edge_attr)))
            h = h + F.dropout(update, p=self.dropout, training=self.training)
        return h


class GraphTransformerEncoder(SparseGraphEncoder):
    """Sparse edge-aware graph Transformer based on ``TransformerConv``.

    Multi-head query/key attention models longer-range relational patterns as
    layers are stacked, while edge features alter both keys and values. Unlike
    dense GPS attention, computation remains restricted to skeleton edges.
    """

    def __init__(
        self,
        node_feat_dim: int = 7,
        edge_feat_dim: int = 8,
        embed_dim: int = 128,
        n_layers: int = 4,
        hidden: int = 256,
        heads: int = 4,
        dropout: float = 0.1,
        feature_mode: str = "node_only",
    ):
        super().__init__(node_feat_dim, hidden, embed_dim)
        del feature_mode
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.dropout = dropout
        self.convs = nn.ModuleList(
            [
                TransformerConv(
                    hidden,
                    hidden // heads,
                    heads=heads,
                    concat=True,
                    beta=True,
                    dropout=dropout,
                    edge_dim=edge_feat_dim,
                    root_weight=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])

    def encode_nodes(self, h, edge_index, edge_attr):
        for conv, norm in zip(self.convs, self.norms):
            update = F.gelu(norm(conv(h, edge_index, edge_attr)))
            h = h + F.dropout(update, p=self.dropout, training=self.training)
        return h
