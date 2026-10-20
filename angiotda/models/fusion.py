"""Late-fusion image+graph encoder: DINOv3 (fluorescence) + GIN-E (skeleton graph).

The two branches are L2-normalised and combined by a *fixed* convex weight into
a single vector, which is what the training loss sees. This is score-level late
fusion, not CLIP: no term in the objective compares the image branch against the
graph branch, so nothing pushes the two spaces into alignment. For the
cross-modal contrastive model see models/clip.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gine import GINEEncoder
from .image_encoder import ImageEncoder


class LateFusionEncoder(nn.Module):
    """DINOv3 image branch + GIN-E graph branch, fused by weighted sum.

    Args:
        graph_weight: weight w on the graph branch; the image branch gets 1-w.
                      0.5 = equal, 0.8 = graph-dominant.
    """

    def __init__(self, embed_dim=128, model_type="dinov3_vits16",
                 node_feat_dim=7, edge_feat_dim=8, n_layers=4, hidden_dim=256,
                 graph_weight=0.5, input_mode="learned_adapter"):
        super().__init__()
        self.graph_weight = graph_weight
        self.image_encoder = ImageEncoder(
            output_dim=embed_dim, model_type=model_type, freeze_backbone=False,
            input_mode=input_mode)
        self.graph_encoder = GINEEncoder(
            node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim,
            embed_dim=embed_dim, n_layers=n_layers, hidden=hidden_dim)

    def forward(self, images, x, edge_index, edge_attr, node_mask):
        z_img = self.image_encoder(images)                                # (B,D), unit norm
        z_graph = F.normalize(self.graph_encoder(x, edge_index, edge_attr, node_mask), dim=-1)
        w = self.graph_weight
        return F.normalize(w * z_graph + (1 - w) * z_img, dim=-1)

    def forward_components(self, images, x, edge_index, edge_attr, node_mask):
        """Return the fused vector and both normalized branch vectors."""
        z_img = self.encode_image_only(images)
        z_graph = self.encode_graph_only(x, edge_index, edge_attr, node_mask)
        w = self.graph_weight
        z_fused = F.normalize(w * z_graph + (1 - w) * z_img, dim=-1)
        return z_fused, z_img, z_graph

    # Alias so fusion and CLIP encoders share one downstream-eval entry point.
    embed = forward

    def encode_graph_only(self, x, edge_index, edge_attr, node_mask):
        return F.normalize(self.graph_encoder(x, edge_index, edge_attr, node_mask), dim=-1)

    def encode_image_only(self, images):
        return self.image_encoder(images)

    @classmethod
    def from_checkpoint(cls, ckpt, embed_dim=128, node_feat_dim=7, edge_feat_dim=8,
                        n_layers=4, hidden_dim=256):
        """Rebuild from a checkpoint dict, honouring its stored hyperparameters.

        `graph_weight` is read from the checkpoint rather than assumed, so eval
        cannot silently disagree with how the model was trained.
        """
        if "graph_weight" not in ckpt:
            raise KeyError(
                "checkpoint has no 'graph_weight'; it predates this loader and its "
                "fusion weight is unknown — retrain or pass the weight explicitly")
        model = cls(
            embed_dim=embed_dim, model_type=ckpt["model_type"],
            node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim,
            n_layers=n_layers, hidden_dim=hidden_dim,
            graph_weight=ckpt["graph_weight"],
        )
        model.load_state_dict(ckpt["encoder_state"])
        return model
