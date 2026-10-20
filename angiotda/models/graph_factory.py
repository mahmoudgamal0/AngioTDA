"""Factory for graph encoders sharing the GINE-compatible call interface."""

from __future__ import annotations

from .advanced_graph import GATv2Encoder, GraphTransformerEncoder, PNAEncoder
from .gine import GINEEncoder
from .gps import GPSEncoder
from .deep_sets import DeepSetsEncoder
from .gcn import GCNEncoder


GRAPH_ENCODERS = {
    "gine": GINEEncoder,
    "pna": PNAEncoder,
    "gatv2": GATv2Encoder,
    "transformer": GraphTransformerEncoder,
    "gps": GPSEncoder,
    "deepsets": DeepSetsEncoder,
    "gcn": GCNEncoder,
}


def build_graph_encoder(architecture: str = "gine", **kwargs):
    """Construct an encoder by stable CLI/configuration name."""
    architecture = architecture.lower()
    try:
        encoder_cls = GRAPH_ENCODERS[architecture]
    except KeyError as exc:
        choices = ", ".join(sorted(GRAPH_ENCODERS))
        raise ValueError(f"unknown graph architecture {architecture!r}; choose {choices}") from exc
    # ``heads`` is meaningful only for the attention architectures. Keeping
    # this filtering in the factory prevents shared CLIs from accidentally
    # passing an unsupported argument to PNA or GINE.
    if architecture not in {"gatv2", "transformer", "gps"}:
        kwargs.pop("heads", None)
    return encoder_cls(**kwargs)
