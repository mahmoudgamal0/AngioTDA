"""
GINE encoder — LOCAL connectivity bias.

Architecture:
  4 GINE convolution layers, hidden 256, LN + GELU + residual
  Edge features projected and incorporated at each message-passing step
  Global mean+max pool readout → Linear → embed_dim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool

HIDDEN = 256
N_LAYERS = 4


class GINEEncoder(nn.Module):
    def __init__(
        self,
        node_feat_dim: int = 7,
        edge_feat_dim: int = 8,
        embed_dim: int = 128,
        n_layers: int = N_LAYERS,
        hidden: int = HIDDEN,
        feature_mode: str = "node_only",  # kept for checkpoint compat, ignored
    ):
        super().__init__()
        self.n_layers = n_layers

        self.node_emb = nn.Linear(node_feat_dim, hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=edge_feat_dim))
            self.norms.append(nn.LayerNorm(hidden))

        self.readout = nn.Sequential(
            nn.Linear(hidden * 2, hidden),  # mean + max
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index,
        edge_attr,
        node_mask: torch.Tensor,
        batch: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if x.dim() == 3:
            return self._forward_padded(x, edge_index, edge_attr, node_mask)
        else:
            return self._forward_pyg(x, edge_index, edge_attr, batch)

    def _forward_pyg(self, x, edge_index, edge_attr, batch):
        h = self.node_emb(x)
        for i in range(self.n_layers):
            h_new = self.convs[i](h, edge_index, edge_attr)
            h_new = self.norms[i](h_new)
            h_new = F.gelu(h_new)
            h = h + h_new

        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        pooled = torch.cat([h_mean, h_max], dim=-1)
        return self.readout(pooled)

    def _forward_padded(self, x, edge_index_list, edge_attr_list, node_mask):
        B, N, F = x.shape
        device = x.device

        flat_x_list, flat_ei_list, flat_ea_list, batch_list = [], [], [], []
        offset = 0
        for b in range(B):
            n_real = node_mask[b].sum().item()
            flat_x_list.append(x[b, :n_real])

            ei = edge_index_list[b].to(device)
            ea = edge_attr_list[b].to(device)

            valid = (ei[0] < n_real) & (ei[1] < n_real)
            ei = ei[:, valid] + offset
            ea = ea[valid]

            flat_ei_list.append(ei)
            flat_ea_list.append(ea)
            batch_list.append(torch.full((n_real,), b, dtype=torch.long, device=device))
            offset += n_real

        flat_x = torch.cat(flat_x_list, dim=0)
        flat_ei = torch.cat(flat_ei_list, dim=1) if flat_ei_list else torch.zeros(2, 0, dtype=torch.long, device=device)
        flat_ea = torch.cat(flat_ea_list, dim=0) if flat_ea_list else torch.zeros(0, 2, device=device)
        flat_batch = torch.cat(batch_list, dim=0)

        return self._forward_pyg(flat_x, flat_ei, flat_ea, flat_batch)
