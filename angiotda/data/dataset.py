"""
Dataset for self-supervised training on compound replicates.

Each compound has exactly 2 wells. The dataset returns individual wells.
A custom sampler ensures each batch has B compounds × 2 wells = 2B samples,
so every well in the batch has its positive pair present.
"""

import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


_WELL_RE = re.compile(r"^(P\d+)R\d+_(R\d+C\d+)$")


def replicate_pair_key(image_name: str) -> str:
    """Return the physical-well key shared by its replicate images.

    For example, ``P1R1_R002C016`` and ``P1R2_R002C016`` both map to
    ``P1_R002C016``.  Keeping the plate and location in the key matters for the
    two LOPAC compounds that occur at more than one physical well location.
    """
    match = _WELL_RE.fullmatch(image_name)
    return f"{match.group(1)}_{match.group(2)}" if match else image_name


def _replicate_pairs(samples: list[dict], compound_to_indices: dict) -> dict:
    """Build matched replicate pairs after missing graph files are filtered."""
    result = {}
    for compound, indices in compound_to_indices.items():
        by_well = {}
        for idx in indices:
            key = replicate_pair_key(samples[idx]["image_name"])
            by_well.setdefault(key, []).append(idx)
        pairs = []
        for well_indices in by_well.values():
            if len(well_indices) == 2:
                pairs.append(tuple(well_indices))
        # Backwards-compatible fallback for synthetic/legacy indices whose two
        # image names do not encode a shared physical-well identifier.
        if not pairs and len(indices) == 2:
            pairs.append(tuple(indices))
        if pairs:
            result[compound] = pairs
    return result


class GraphDatasetV2(Dataset):
    """Loads v2 graphs (.npz with 7D node + 8D edge features)."""

    def __init__(self, sample_index_path: Path, graphs_dir: Path,
                 controls: list[str] = None):
        with open(sample_index_path) as f:
            si = json.load(f)

        controls = set(controls or ["CTRL", "DMSO", "Suramin"])

        self.samples = []
        self.compound_to_indices = {}

        for s in si["samples"]:
            compound = s["compound"]
            if compound in controls:
                continue
            image_name = s["image_name"]
            graph_path = graphs_dir / f"{image_name}.npz"
            if not graph_path.exists():
                continue
            idx = len(self.samples)
            self.samples.append({
                "sample_id": s["sample_id"],
                "image_name": image_name,
                "compound": compound,
                "graph_path": str(graph_path),
                "moa": s.get("moa", []),
            })
            self.compound_to_indices.setdefault(compound, []).append(idx)

        self.compounds = sorted(self.compound_to_indices.keys())
        self.compound_to_replicate_pairs = _replicate_pairs(
            self.samples, self.compound_to_indices)
        print(f"GraphDatasetV2: {len(self.samples)} wells, "
              f"{len(self.compounds)} compounds")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        with np.load(s["graph_path"]) as data:
            nf_key = "node_feat" if "node_feat" in data else "node_feat_slim"
            node_feat = data[nf_key].astype(np.float32)
            edge_index = data["edge_index"].astype(np.int64)
            edge_attr = data["edge_attr"].astype(np.float32)
            graph_stats = data["graph_stats"].astype(np.float32)
            n_nodes = int(data["n_nodes"])

        if node_feat.ndim != 2 or node_feat.shape[0] != n_nodes:
            raise ValueError(
                f"{s['graph_path']}: node features {node_feat.shape} do not "
                f"match n_nodes={n_nodes}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"{s['graph_path']}: edge_index must have shape (2, E), got "
                f"{edge_index.shape}")
        if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"{s['graph_path']}: edge_attr rows must equal edge count; got "
                f"{edge_attr.shape} for {edge_index.shape[1]} edges")
        if edge_index.size and (
                edge_index.min() < 0 or edge_index.max() >= n_nodes):
            raise ValueError(
                f"{s['graph_path']}: edge index outside [0, {n_nodes})")
        return {
            "node_feat": torch.from_numpy(node_feat),
            "edge_index": torch.from_numpy(edge_index),
            "edge_attr": torch.from_numpy(edge_attr),
            "graph_stats": torch.from_numpy(graph_stats),
            "n_nodes": n_nodes,
            "compound": s["compound"],
            "image_name": s["image_name"],
        }


class CompoundPairSampler(Sampler):
    """Samples B compounds per batch, yielding two wells for each compound.

    Each iteration yields indices for B*2 wells (B compounds × 2 wells each).
    Compounds with fewer than 2 wells are skipped.
    """

    def __init__(self, dataset: GraphDatasetV2, batch_compounds: int = 64,
                 seed: int = 0):
        self.dataset = dataset
        self.batch_compounds = batch_compounds
        self.seed = seed
        # One RNG for the sampler's whole lifetime: its state carries across
        # epochs, so every epoch draws a *different* compound grouping (and a
        # different pair of wells where a compound has more than two), while the
        # whole sequence stays reproducible from `seed` alone. Re-seeding inside
        # __iter__ instead would replay the identical batches every epoch.
        self._rng = random.Random(seed)

        if batch_compounds < 2:
            raise ValueError("batch_compounds must be at least 2 for negatives")

        pair_map = getattr(dataset, "compound_to_replicate_pairs", None)
        if pair_map is None:
            pair_map = {
                c: [tuple(idxs)] for c, idxs in dataset.compound_to_indices.items()
                if len(idxs) == 2
            }
        self.compound_to_pairs = pair_map
        self.valid_compounds = sorted(
            c for c, pairs in self.compound_to_pairs.items() if pairs)
        full, remainder = divmod(len(self.valid_compounds), batch_compounds)
        # A one-compound tail has no negatives and is therefore dropped. Any
        # larger tail is a valid contrastive batch and must count toward the LR
        # scheduler's total number of steps.
        self.n_batches = full + int(remainder >= 2)

    def __iter__(self):
        rng = self._rng
        compounds = self.valid_compounds.copy()
        rng.shuffle(compounds)

        for batch_start in range(0, len(compounds), self.batch_compounds):
            batch_compounds = compounds[batch_start:batch_start + self.batch_compounds]
            if len(batch_compounds) < 2:
                continue
            indices = []
            for c in batch_compounds:
                chosen = rng.choice(self.compound_to_pairs[c])
                indices.extend(chosen)
            yield indices

    def __len__(self):
        return self.n_batches


def collate_fn(batch: list[dict]) -> dict:
    """Collate variable-size graphs into padded batch.

    Returns:
        x: (B, max_N, 7) padded node features
        node_mask: (B, max_N) bool
        edge_index: list of B tensors, each (2, E_i)
        edge_attr: list of B tensors, each (E_i, 8)
        graph_stats: (B, 10)
        compound: list of B strings
        image_name: list of B strings
    """
    if not batch:
        raise ValueError("cannot collate an empty graph batch")
    max_n = max(item["n_nodes"] for item in batch)
    max_n = max(max_n, 1)  # at least 1 to avoid empty tensors

    B = len(batch)
    x = torch.zeros(B, max_n, batch[0]["node_feat"].shape[-1])
    node_mask = torch.zeros(B, max_n, dtype=torch.bool)
    edge_index_list = []
    edge_attr_list = []
    graph_stats = torch.stack([item["graph_stats"] for item in batch])
    compounds = [item["compound"] for item in batch]
    image_names = [item["image_name"] for item in batch]

    for i, item in enumerate(batch):
        n = item["n_nodes"]
        if item["node_feat"].ndim != 2 or item["node_feat"].shape[0] != n:
            raise ValueError(
                f"sample {item.get('image_name', i)} has n_nodes={n} but "
                f"node_feat shape {tuple(item['node_feat'].shape)}")
        if item["edge_index"].ndim != 2 or item["edge_index"].shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        if (item["edge_attr"].ndim != 2 or
                item["edge_attr"].shape[0] != item["edge_index"].shape[1]):
            raise ValueError("edge_attr rows must equal edge_index columns")
        if item["edge_index"].numel() and (
                item["edge_index"].min() < 0 or item["edge_index"].max() >= n):
            raise ValueError("edge_index contains a node outside this graph")
        if n > 0:
            x[i, :n] = item["node_feat"]
            node_mask[i, :n] = True
        edge_index_list.append(item["edge_index"])
        edge_attr_list.append(item["edge_attr"])

    return {
        "x": x,
        "node_mask": node_mask,
        "edge_index": edge_index_list,
        "edge_attr": edge_attr_list,
        "graph_stats": graph_stats,
        "compound": compounds,
        "image_name": image_names,
    }
