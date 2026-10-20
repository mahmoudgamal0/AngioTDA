"""Node-wise Deep Sets baseline with no message passing or attention.

Four shared residual MLP blocks replace the GINE message-passing blocks.
Mean/max pooling and the graph-level readout match the other graph encoders.
Edges and edge attributes are ignored. The seven original node features are
retained, including any features computed from the skeleton during extraction.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_max_pool, global_mean_pool


class DeepSetsEncoder(nn.Module):
    def __init__(self, node_feat_dim=7, edge_feat_dim=8, embed_dim=128,
                 n_layers=4, hidden=256, feature_mode="node_only"):
        super().__init__()
        del edge_feat_dim, feature_mode
        self.node_emb = nn.Linear(node_feat_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.readout = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, embed_dim))

    def forward(self, x, edge_index=None, edge_attr=None, node_mask=None,
                batch=None, **kwargs):
        del edge_index, edge_attr
        if x.dim() == 3:
            if node_mask is None or node_mask.shape != x.shape[:2]:
                raise ValueError("padded node sets require a matching node_mask")
            n_graphs = x.shape[0]
            mask = node_mask.bool()
            batch = torch.arange(n_graphs, device=x.device)[:, None].expand_as(mask)[mask]
            x = x[mask]
        elif x.dim() == 2:
            if batch is None:
                batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            n_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        else:
            raise ValueError("expected flat or padded node features")
        h = self.node_emb(x)
        for block, norm in zip(self.blocks, self.norms):
            h = h + F.gelu(norm(block(h)))
        pooled = torch.cat([global_mean_pool(h, batch, size=n_graphs),
                            global_max_pool(h, batch, size=n_graphs)], dim=-1)
        return self.readout(pooled)
