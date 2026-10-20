"""GPS with edge-aware local GINE and global Performer attention.

The existing coordinates and degree features supply position/structure; no
extra node features are introduced. Size buckets limit attention padding while
retaining every graph and all negatives in the original contrastive batch.
"""

import torch
from torch import nn
from torch_geometric.nn import GPSConv, GINEConv, global_mean_pool, global_max_pool
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.attention.performer import generalized_kernel, linear_attention

from .advanced_graph import SparseGraphEncoder


class MaskedPerformerAttention(PerformerAttention):
    """Mask projected keys as well as values in PyG's linear attention.

    PyG 2.7 masks only values. Its positive kernel offset leaves padded keys
    in the normalization denominator, changing short-graph outputs with padding.
    """

    def forward(self, x, mask=None):
        batch_size, n_nodes, _ = x.shape
        q, k, v = [layer(x).reshape(batch_size, n_nodes, self.heads,
                                  self.head_channels).permute(0, 2, 1, 3)
                   for layer in (self.q, self.k, self.v)]
        q = generalized_kernel(q, self.fast_attn.projection_matrix, self.kernel)
        k = generalized_kernel(k, self.fast_attn.projection_matrix, self.kernel)
        if mask is not None:
            valid = mask[:, None, :, None]
            k = k.masked_fill(~valid, 0)
            v = v.masked_fill(~valid, 0)
        out = linear_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, n_nodes, -1)
        return self.dropout(self.attn_out(out))


class GPSEncoder(SparseGraphEncoder):
    def __init__(self, node_feat_dim=7, edge_feat_dim=8, embed_dim=128,
                 n_layers=4, hidden=256, heads=4, dropout=0.1,
                 feature_mode="node_only", max_padded_nodes=32768):
        super().__init__(node_feat_dim, hidden, embed_dim)
        if hidden % heads or max_padded_nodes < 1:
            raise ValueError("heads must divide hidden; padding budget must be positive")
        self.max_padded_nodes = max_padded_nodes
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            local = GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                          nn.Linear(hidden, hidden)),
                             train_eps=True, edge_dim=edge_feat_dim)
            layer = GPSConv(
                hidden, local, heads=heads, dropout=dropout, act="gelu",
                norm="layer_norm", norm_kwargs={"mode": "node"},
                attn_type="performer", attn_kwargs={"head_channels": hidden // heads},
            )
            # Preserve initialization and checkpoint keys while fixing masking.
            attention = MaskedPerformerAttention(hidden, heads, head_channels=hidden // heads)
            attention.load_state_dict(layer.attn.state_dict())
            layer.attn = attention
            self.layers.append(layer)

    def encode_nodes(self, h, edge_index, edge_attr):
        raise RuntimeError("GPS requires graph batch membership")

    def _forward_pyg(self, x, edge_index, edge_attr, batch):
        h = self.node_emb(x)
        for layer in self.layers:
            h = layer(h, edge_index, batch=batch, edge_attr=edge_attr)
        return self.readout(torch.cat([global_mean_pool(h, batch),
                                      global_max_pool(h, batch)], dim=-1))

    def forward(self, x, edge_index, edge_attr, node_mask=None, batch=None, **kwargs):
        if x.dim() != 3:
            return super().forward(x, edge_index, edge_attr, node_mask, batch, **kwargs)
        sizes = node_mask.sum(dim=1).tolist()
        buckets, current = [], []
        for index in sorted(range(len(sizes)), key=lambda i: sizes[i]):
            if current and max(1, sizes[index]) * (len(current) + 1) > self.max_padded_nodes:
                buckets.append(current)
                current = []
            current.append(index)
        if current:
            buckets.append(current)
        outputs, order = [], []
        for indices in buckets:
            length = max(1, max(sizes[i] for i in indices))
            group = torch.as_tensor(indices, device=x.device)
            flat = self._flatten_padded(
                x[group, :length], [edge_index[i] for i in indices],
                [edge_attr[i] for i in indices], node_mask[group, :length])
            outputs.append(self._forward_pyg(*flat))
            order.extend(indices)
        inverse = torch.argsort(torch.tensor(order, device=x.device))
        return torch.cat(outputs, dim=0)[inverse]
