"""Model architectures.

Core encoders and graph-architecture candidates:

  GINEEncoder        graph-only, message passing over the skeleton graph
  ImageEncoder       DINOv3 ViT on the fluorescence image
  LateFusionEncoder  fixed-weight sum of the two above
  CLIPEncoder        dual tower, cross-modal contrastive
  CellDINOEncoder    frozen Cell-Painting ViT-S/8 baseline (no checkpoint)
  PNAEncoder         degree-scaled multi-aggregation graph encoder
  GATv2Encoder       dynamic edge-aware graph attention
  GraphTransformerEncoder  sparse edge-conditioned TransformerConv stack
  GPSEncoder         local GINE and whole-graph Performer attention
"""

from angiotda.models.gine import GINEEncoder
from angiotda.models.image_encoder import ImageEncoder
from angiotda.models.fusion import LateFusionEncoder
from angiotda.models.clip import CLIPEncoder
from angiotda.models.cell_dino import CellDINOEncoder
from angiotda.models.projection_head import ProjectionHead
from angiotda.models.advanced_graph import (
    GATv2Encoder,
    GraphTransformerEncoder,
    PNAEncoder,
)
from angiotda.models.graph_factory import build_graph_encoder
from angiotda.models.gps import GPSEncoder

__all__ = [
    "GINEEncoder", "ImageEncoder", "LateFusionEncoder", "CLIPEncoder",
    "CellDINOEncoder", "ProjectionHead",
    "PNAEncoder", "GATv2Encoder", "GraphTransformerEncoder", "GPSEncoder",
    "build_graph_encoder",
]
