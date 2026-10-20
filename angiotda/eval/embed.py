"""Canonical embedding functions — one convention for every evaluation.

Every embedder returns ``(compound_embs, well_embs)``:

  compound_embs : {compound -> unit vector}, the mean over that compound's wells
  well_embs     : {(compound, well_index) -> unit vector}, **one entry per well**

All physical wells remain separate. A compound with four wells contributes
four well vectors; its compound vector is their normalized mean.

All embedders take a checkpoint path and return the same shape, so a caller can
swap architectures without special-casing.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from angiotda.data.dataset import GraphDatasetV2, collate_fn
from angiotda.data.graph_ablation import degree_preserving_rewire
from angiotda.data.paths import DATASET
from angiotda import CONTROLS

__all__ = [
    "pool", "embed_gine", "embed_raw_26d", "embed_dino",
    "embed_dino_repeat_frozen", "embed_dino_repeat_frozen_400",
    "embed_fusion", "embed_clip", "embed_cell_dino",
    "embed_advanced_graph", "embed_four_view",
]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-8)


def pool(per_well: list[tuple[str, np.ndarray]]) -> tuple[dict, dict]:
    """(compound, vector) pairs -> (compound_embs, well_embs). Keeps every well."""
    accum: dict[str, list[np.ndarray]] = defaultdict(list)
    well_embs: dict[tuple[str, int], np.ndarray] = {}
    for compound, vec in per_well:
        well_embs[(compound, len(accum[compound]))] = _unit(vec)
        accum[compound].append(vec)
    compound_embs = {c: _unit(np.mean(v, axis=0)) for c, v in accum.items()}
    return compound_embs, well_embs


def _graph_loader(batch_size: int = 16):
    ds = GraphDatasetV2(DATASET.sample_index, DATASET.graphs_dir, controls=CONTROLS)
    return ds, torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# Preserve the old private import while using the shared training/evaluation code.
_degree_preserving_rewire = degree_preserving_rewire


def embed_gine(ckpt_path: Path, condition: str = "real", seed: int = 42,
               drop_node_features: bool = False):
    """GIN-E graph encoder.

    condition: 'real' | 'scrambled' (edges rewired at random) | 'empty' (no edges)
    drop_node_features: zero the node features, leaving structure only
    """
    from angiotda.models.gine import GINEEncoder
    dev = _device()
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    cfg = ckpt.get("cfg", {})
    enc = GINEEncoder(
        node_feat_dim=cfg.get("node_feat_dim", 7), edge_feat_dim=cfg.get("edge_feat_dim", 8),
        embed_dim=cfg.get("embed_dim", 128), n_layers=cfg.get("n_layers", 4),
        hidden=cfg.get("hidden_dim", 256)).to(dev)
    enc.load_state_dict(ckpt["encoder_state"])
    enc.eval()

    _, loader = _graph_loader()
    gen = torch.Generator().manual_seed(seed)
    per_well = []
    with torch.inference_mode():
        for batch in loader:
            x = batch["x"].to(dev)
            if drop_node_features:
                x = torch.zeros_like(x)
            node_mask = batch["node_mask"].to(dev)
            if condition == "scrambled":
                rewired = [_degree_preserving_rewire(ei, ea, gen)
                           for ei, ea in zip(batch["edge_index"], batch["edge_attr"])]
                ei = [pair[0].to(dev) for pair in rewired]
                ea = [pair[1].to(dev) for pair in rewired]
            elif condition == "empty":
                ei = [torch.zeros((2, 0), dtype=torch.long, device=dev)
                      for _ in batch["edge_index"]]
                ea = [torch.zeros((0, e.shape[1] if e.ndim > 1 else 1), device=dev)
                      for e in batch["edge_attr"]]
            else:
                ei = [e.to(dev) for e in batch["edge_index"]]
                ea = [e.to(dev) for e in batch["edge_attr"]]
            emb = enc(x, ei, ea, node_mask)
            per_well += [(c, emb[i].float().cpu().numpy())
                         for i, c in enumerate(batch["compound"])]
    return pool(per_well)


def embed_raw_26d():
    """Raw graph statistics, min-max normalised across the dataset."""
    ds, _ = _graph_loader()
    stats, compounds = [], []
    for i in range(len(ds)):
        item = ds[i]
        stats.append(item["graph_stats"].numpy())
        compounds.append(item["compound"])
    stats = np.asarray(stats)
    rng = stats.max(axis=0) - stats.min(axis=0)
    rng[rng == 0] = 1
    normed = (stats - stats.min(axis=0)) / rng
    return pool(list(zip(compounds, normed)))


def _image_samples():
    """Screen wells for the image-only encoders — the same set GraphDatasetV2 keeps.

    Only the three on-plate controls are dropped. "Suramin sodium salt" was
    dropped here as well, which was wrong: it is the LOPAC library entry
    (P2X/P2Y antagonist), sitting at a single library position P4/R014C005 with
    the usual two replicate wells, and is a different thing from the "Suramin"
    reference control that occupies columns 2 and 23 of every plate. Dropping it
    left the image rows scoring 2,558 wells / 1,277 compounds while every graph
    and multimodal row scored 2,560 / 1,278.
    """
    import json
    payload = json.loads(DATASET.sample_index.read_text())
    samples = payload["samples"] if isinstance(payload, dict) else payload
    drop = {c.lower() for c in CONTROLS}
    return [s for s in samples if s["compound"].lower() not in drop]


def _embed_images(encode, img_size: int, batch_size: int):
    from angiotda.data.multimodal import load_image
    from angiotda import PROJECT_ROOT
    dev = _device()
    samples = _image_samples()
    per_well = []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start:start + batch_size]
            imgs = torch.stack([load_image(PROJECT_ROOT / s["image_path"], img_size)
                                for s in chunk]).to(dev)
            z = encode(imgs).float().cpu().numpy()
            per_well += [(s["compound"], z[i]) for i, s in enumerate(chunk)]
    return pool(per_well)


def embed_dino(ckpt_path: Path, batch_size: int = 32):
    """Image-only DINOv3 encoder."""
    from angiotda.models.image_encoder import ImageEncoder
    dev = _device()
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    enc = ImageEncoder(output_dim=128, model_type=ckpt["model_type"],
                       freeze_backbone=ckpt.get("freeze_backbone", False),
                       input_mode=ckpt.get("input_mode", "learned_adapter")).to(dev)
    enc.load_state_dict(ckpt["encoder_state"])
    enc.eval()
    return _embed_images(enc, ckpt.get("img_size", 224), batch_size)


def embed_dino_repeat_frozen(img_size: int = 224, batch_size: int = 32):
    """Direct pretrained DINOv3 features from grayscale repeated to RGB.

    This is deliberately checkpoint-free: neither an input adapter nor a
    projection head is trained. ``encode_features`` bypasses the randomly
    initialised projection module entirely.
    """
    from angiotda.models.image_encoder import ImageEncoder
    enc = ImageEncoder(
        output_dim=128,
        model_type="dinov3_vits16",
        freeze_backbone=True,
        input_mode="repeat",
    ).to(_device())
    enc.eval()
    return _embed_images(enc.encode_features, img_size, batch_size)


def embed_dino_repeat_frozen_400(batch_size: int = 8):
    """Frozen direct DINOv3 features at the native 400x400 TIFF resolution."""
    return embed_dino_repeat_frozen(img_size=400, batch_size=batch_size)


def embed_cell_dino(pretrained_path=None, img_size: int = 224, batch_size: int = 16):
    """Frozen Cell-DINO baseline. No checkpoint: the weights are the release."""
    from angiotda.models.cell_dino import CellDINOEncoder
    enc = CellDINOEncoder(pretrained_path=pretrained_path, img_size=img_size).to(_device())
    enc.eval()
    return _embed_images(enc.encode_features, img_size, batch_size)


def _embed_multimodal(encoder, img_size: int, batch_size: int):
    """Shared graph+image walk for the fusion and CLIP encoders."""
    from angiotda.data.multimodal import build_dataset, build_eval_loader, batch_to_device
    from angiotda import PROJECT_ROOT
    dev = _device()
    ds = build_dataset(PROJECT_ROOT, img_size=img_size)
    # num_workers=0: worker processes deadlocked here on repeated
    # loader construction within one process, and the IO is not the
    # bottleneck for a ViT forward pass.
    loader = build_eval_loader(ds, batch_size=batch_size, num_workers=0)
    per_well = []
    with torch.inference_mode():
        for batch in loader:
            images, x, ei, ea, nm = batch_to_device(batch, dev)
            z = encoder.embed(images, x, ei, ea, nm).float().cpu().numpy()
            per_well += [(c, z[i]) for i, c in enumerate(batch["compound"])]
    return pool(per_well)


def embed_fusion(ckpt_path: Path, batch_size: int = 16):
    """Late-fusion encoder; the fusion weight is read from the checkpoint."""
    from angiotda.models.fusion import LateFusionEncoder
    dev = _device()
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    enc = LateFusionEncoder.from_checkpoint(ckpt).to(dev)
    enc.eval()
    return _embed_multimodal(enc, ckpt.get("img_size", 224), batch_size)


def embed_clip(ckpt_path: Path, batch_size: int = 16):
    """Cross-modal CLIP encoder.

    Uses the same summed-tower embedding as fusion. The two give very different
    replicate consistency, and that difference is the finding, not an artefact:
    CLIP aligns the two towers *per well*, so the summed vector encodes well
    identity rather than compound identity.
    """
    from angiotda.models.clip import CLIPEncoder
    dev = _device()
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    enc = CLIPEncoder.from_checkpoint(ckpt).to(dev)
    enc.eval()
    return _embed_multimodal(enc, ckpt.get("img_size", 224), batch_size)


def embed_advanced_graph(ckpt_path: Path, batch_size: int = 16):
    """PNA, GATv2, or sparse Transformer checkpoint from the new battery."""
    from angiotda.models.graph_factory import build_graph_encoder

    dev = _device()
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    if ckpt.get("arch") != "advanced_graph_ssl":
        raise ValueError(f"not an advanced graph checkpoint: {ckpt_path}")
    cfg = ckpt.get("cfg", {})
    kwargs = {
        "node_feat_dim": cfg.get("node_feat_dim", 7),
        "edge_feat_dim": cfg.get("edge_feat_dim", 8),
        "embed_dim": cfg.get("embed_dim", 128),
        "n_layers": cfg.get("layers", 4),
        "hidden": cfg.get("hidden", 256),
        "heads": cfg.get("heads", 4),
        "dropout": cfg.get("dropout", 0.1),
    }
    enc = build_graph_encoder(ckpt["architecture"], **kwargs).to(dev)
    enc.load_state_dict(ckpt["encoder_state"])
    enc.eval()

    _, loader = _graph_loader(batch_size=batch_size)
    per_well = []
    with torch.inference_mode():
        for batch in loader:
            emb = enc(
                batch["x"].to(dev),
                [edge.to(dev) for edge in batch["edge_index"]],
                [edge.to(dev) for edge in batch["edge_attr"]],
                batch["node_mask"].to(dev),
            )
            per_well += [
                (compound, emb[i].float().cpu().numpy())
                for i, compound in enumerate(batch["compound"])
            ]
    return pool(per_well)


def embed_four_view(ckpt_path: Path, batch_size: int = 16):
    """Fused representation from the four-view DINO+GIN-E checkpoint."""
    from angiotda.models.four_view_model import FourViewEncoder

    dev = _device()
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    enc = FourViewEncoder.from_checkpoint(
        ckpt,
        embed_dim=ckpt.get("embed_dim", 128),
        n_layers=ckpt.get("graph_layers", 4),
        hidden_dim=ckpt.get("graph_hidden", 256),
    ).to(dev)
    enc.eval()
    return _embed_multimodal(enc, ckpt.get("img_size", 224), batch_size)
