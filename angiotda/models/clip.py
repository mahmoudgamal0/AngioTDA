"""True CLIP-style cross-modal encoder: DINOv3 (fluorescence) ↔ GIN-E (skeleton graph).

Unlike models/fusion.py, the two branches are kept *separate* and the objective
is the symmetric InfoNCE of Radford et al. (2021) over the image×graph
similarity matrix, with a learned temperature. Positives are the image and the
skeleton graph of the *same well*; every other well in the batch supplies a
negative in both directions. That is what forces the two encoders into a shared
space — the property fixed-weight fusion never establishes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gine import GINEEncoder
from .image_encoder import ImageEncoder

# CLIP clamps the learned temperature so logits cannot blow up (1/0.01 = 100).
MAX_LOGIT_SCALE = math.log(100.0)


class CLIPEncoder(nn.Module):
    """Dual-tower image/graph encoder with a learned logit scale.

    Args:
        init_temperature: initial softmax temperature (CLIP uses 0.07).
    """

    def __init__(self, embed_dim=128, model_type="dinov3_vits16",
                 node_feat_dim=7, edge_feat_dim=8, n_layers=4, hidden_dim=256,
                 init_temperature=0.07):
        super().__init__()
        self.image_encoder = ImageEncoder(
            output_dim=embed_dim, model_type=model_type, freeze_backbone=False)
        self.graph_encoder = GINEEncoder(
            node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim,
            embed_dim=embed_dim, n_layers=n_layers, hidden=hidden_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))

    def encode_image(self, images):
        return self.image_encoder(images)  # already L2-normalised

    def encode_graph(self, x, edge_index, edge_attr, node_mask):
        return F.normalize(self.graph_encoder(x, edge_index, edge_attr, node_mask), dim=-1)

    def forward(self, images, x, edge_index, edge_attr, node_mask):
        """Returns (z_img, z_graph, logit_scale) — the branches stay separate."""
        z_img = self.encode_image(images)
        z_graph = self.encode_graph(x, edge_index, edge_attr, node_mask)
        return z_img, z_graph, self.logit_scale.clamp(max=MAX_LOGIT_SCALE).exp()

    def embed(self, images, x, edge_index, edge_attr, node_mask):
        """Single vector per well for downstream eval.

        Averaging the towers is only meaningful *because* the CLIP objective has
        aligned them; the same operation on an unaligned pair of encoders is
        what late fusion does.
        """
        z_img, z_graph, _ = self.forward(images, x, edge_index, edge_attr, node_mask)
        return F.normalize(z_img + z_graph, dim=-1)

    @classmethod
    def from_checkpoint(cls, ckpt, embed_dim=128, node_feat_dim=7, edge_feat_dim=8,
                        n_layers=4, hidden_dim=256):
        if ckpt.get("arch") != "clip" or "logit_scale" not in ckpt.get("encoder_state", {}):
            raise ValueError(
                "checkpoint is not a true CLIP checkpoint: expected arch='clip' and "
                "a learned 'logit_scale' parameter in encoder_state"
            )
        model = cls(
            embed_dim=embed_dim, model_type=ckpt["model_type"],
            node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim,
            n_layers=n_layers, hidden_dim=hidden_dim,
        )
        model.load_state_dict(ckpt["encoder_state"])
        return model
