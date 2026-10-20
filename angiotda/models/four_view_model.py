"""Dual-tower encoder used by the four-view objective."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from angiotda.models.gine import GINEEncoder
from angiotda.models.image_encoder import ImageEncoder


class FourViewEncoder(nn.Module):
    """Image and graph towers without CLIP's unused learned temperature."""

    def __init__(self, embed_dim=128, model_type="dinov3_vits16",
                 node_feat_dim=7, edge_feat_dim=8, n_layers=4, hidden_dim=256,
                 init_temperature=0.07, input_mode="learned_adapter"):
        super().__init__()
        # init_temperature is accepted to keep experiment construction explicit;
        # temperature belongs to FourViewObjective, not the encoder.
        self.init_temperature = init_temperature
        self.image_encoder = ImageEncoder(
            output_dim=embed_dim, model_type=model_type, freeze_backbone=False,
            input_mode=input_mode,
        )
        self.graph_encoder = GINEEncoder(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            embed_dim=embed_dim,
            n_layers=n_layers,
            hidden=hidden_dim,
        )

    def forward(self, images, x, edge_index, edge_attr, node_mask):
        z_image = self.image_encoder(images)
        z_graph = F.normalize(
            self.graph_encoder(x, edge_index, edge_attr, node_mask), dim=-1
        )
        return z_image, z_graph

    def embed(self, images, x, edge_index, edge_attr, node_mask):
        z_image, z_graph = self.forward(
            images, x, edge_index, edge_attr, node_mask
        )
        return F.normalize(z_image + z_graph, dim=-1)

    @classmethod
    def from_checkpoint(cls, ckpt, embed_dim=128, node_feat_dim=7,
                        edge_feat_dim=8, n_layers=4, hidden_dim=256):
        if ckpt.get("arch") != "four_view_multimodal":
            raise ValueError("expected arch='four_view_multimodal'")
        model = cls(
            embed_dim=embed_dim,
            model_type=ckpt["model_type"],
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            init_temperature=ckpt.get("temperature", 0.07),
            input_mode=ckpt.get("input_mode", "learned_adapter"),
        )
        model.load_state_dict(ckpt["encoder_state"])
        return model
