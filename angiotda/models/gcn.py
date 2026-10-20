"""Simple GCN baseline using node features and unweighted connectivity only."""

from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

from .advanced_graph import SparseGraphEncoder


class GCNEncoder(SparseGraphEncoder):
    """Standard GCN convolutions with the shared mean/max graph readout.

    Each layer uses fixed symmetric degree normalization and self-loops.
    There is no attention or learned edge weighting, and the eight edge
    attributes are ignored. Width, layer count, residual normalization, and
    graph readout follow the existing graph-encoder comparison.
    """

    def __init__(self, node_feat_dim=7, edge_feat_dim=8, embed_dim=128,
                 n_layers=4, hidden=256, feature_mode="node_only"):
        super().__init__(node_feat_dim, hidden, embed_dim)
        del edge_feat_dim, feature_mode
        self.convs = nn.ModuleList([
            GCNConv(hidden, hidden, normalize=True, add_self_loops=True, cached=False)
            for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])

    def encode_nodes(self, h, edge_index, edge_attr):
        del edge_attr
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.gelu(norm(conv(h, edge_index)))
        return h
