"""Shared datasets, audits, embedding, and checkpoint helpers for cross-fit runs."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from angiotda.data.dataset import collate_fn, replicate_pair_key, CompoundPairSampler
from angiotda.data.graph_ablation import ablate_graph_batch
from angiotda.eval.training_monitor import evaluate_representation
from angiotda.models.graph_factory import build_graph_encoder


_REPLICATE_RE = re.compile(r"^P\d+R([12])_R\d+C\d+$")


class CompoundSubsetDataset(torch.utils.data.Dataset):
    """Index-preserving compound subset with correctly remapped pair indices."""

    def __init__(self, base_dataset, compounds):
        requested = set(compounds)
        self._base = base_dataset
        self.base_indices = [
            index for index, sample in enumerate(base_dataset.samples)
            if sample["compound"] in requested
        ]
        self.samples = [base_dataset.samples[index] for index in self.base_indices]
        self.compound_to_indices = {}
        for local_index, sample in enumerate(self.samples):
            self.compound_to_indices.setdefault(sample["compound"], []).append(local_index)
        self.compounds = sorted(self.compound_to_indices)
        missing = sorted(requested - set(self.compounds))
        if missing:
            raise ValueError(f"subset compounds absent from graph dataset: {missing[:5]}")

        base_to_local = {
            base_index: local_index
            for local_index, base_index in enumerate(self.base_indices)
        }
        self.compound_to_replicate_pairs = {}
        for compound, pairs in base_dataset.compound_to_replicate_pairs.items():
            if compound not in requested:
                continue
            remapped = [
                (base_to_local[left], base_to_local[right])
                for left, right in pairs
                if left in base_to_local and right in base_to_local
            ]
            if remapped:
                self.compound_to_replicate_pairs[compound] = remapped

    def __len__(self):
        return len(self.base_indices)

    def __getitem__(self, index):
        return self._base[self.base_indices[index]]


def audit_physical_pairs(dataset: CompoundSubsetDataset) -> dict:
    """Fail unless every training pair is a distinct matched physical R1/R2 pair."""
    n_pairs = 0
    locations = Counter()
    for compound, pairs in dataset.compound_to_replicate_pairs.items():
        for left, right in pairs:
            if left == right:
                raise AssertionError(f"self-pair for {compound}")
            first, second = dataset.samples[left], dataset.samples[right]
            if first["compound"] != compound or second["compound"] != compound:
                raise AssertionError(f"cross-compound positive pair for {compound}")
            if replicate_pair_key(first["image_name"]) != replicate_pair_key(second["image_name"]):
                raise AssertionError(f"unmatched physical location for {compound}")
            reps = {
                int(_REPLICATE_RE.fullmatch(first["image_name"]).group(1))
                if _REPLICATE_RE.fullmatch(first["image_name"]) else None,
                int(_REPLICATE_RE.fullmatch(second["image_name"]).group(1))
                if _REPLICATE_RE.fullmatch(second["image_name"]) else None,
            }
            if reps != {1, 2}:
                raise AssertionError(f"positive pair is not physical R1/R2 for {compound}")
            locations[compound] += 1
            n_pairs += 1
    missing = sorted(set(dataset.compounds) - set(dataset.compound_to_replicate_pairs))
    return {
        "n_subset_compounds": len(dataset.compounds),
        "n_pairable_compounds": len(dataset.compound_to_replicate_pairs),
        "n_physical_r1_r2_pairs": n_pairs,
        "unpairable_compounds": missing,
        "compounds_with_multiple_locations": sorted(
            compound for compound, count in locations.items() if count > 1
        ),
    }


def audit_training_batch(batch: dict) -> None:
    grouped = {}
    for compound, image_name in zip(batch["compound"], batch["image_name"]):
        grouped.setdefault(compound, []).append(image_name)
    for compound, names in grouped.items():
        if len(names) != 2 or len(set(names)) != 2:
            raise AssertionError(f"batch does not contain exactly two distinct wells for {compound}")
        if replicate_pair_key(names[0]) != replicate_pair_key(names[1]):
            raise AssertionError(f"batch wells do not share a physical location for {compound}")
        matches = [_REPLICATE_RE.fullmatch(name) for name in names]
        if any(match is None for match in matches) or {int(match.group(1)) for match in matches} != {1, 2}:
            raise AssertionError(f"batch does not contain physical R1/R2 for {compound}")


def training_degree_histogram(dataset: CompoundSubsetDataset) -> list[int]:
    histogram = np.zeros(1, dtype=np.int64)
    for base_index in dataset.base_indices:
        path = dataset._base.samples[base_index]["graph_path"]
        with np.load(path) as graph:
            n_nodes = int(graph["n_nodes"])
            edge_index = graph["edge_index"]
        degree = np.bincount(edge_index[1], minlength=n_nodes)
        counts = np.bincount(degree)
        if counts.size > histogram.size:
            histogram = np.pad(histogram, (0, counts.size - histogram.size))
        histogram[:counts.size] += counts
    return histogram.tolist()


def build_crossfit_encoder(architecture: str, config: dict):
    kwargs = {
        "node_feat_dim": 7,
        "edge_feat_dim": 8,
        "embed_dim": config["embed_dim"],
        "n_layers": config["layers"],
        "hidden": config["hidden"],
    }
    if architecture == "pna":
        kwargs.update(
            degree_histogram=config["degree_histogram"],
            dropout=config["dropout"],
        )
    elif architecture in {"gatv2", "transformer", "gps"}:
        kwargs.update(
            heads=config["heads"],
            dropout=config["dropout"],
        )
    return build_graph_encoder(architecture, **kwargs)


def collect_embeddings(encoder, loader, device, graph_ablation="none", ablation_seed=0):
    was_training = encoder.training
    encoder.eval()
    compounds, image_names, vectors = [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = ablate_graph_batch(batch, graph_ablation, ablation_seed)
            output = encoder(
                batch["x"].to(device),
                [value.to(device) for value in batch["edge_index"]],
                [value.to(device) for value in batch["edge_attr"]],
                batch["node_mask"].to(device),
            ).float().cpu().numpy()
            compounds.extend(batch["compound"])
            image_names.extend(batch["image_name"])
            vectors.extend(output)
    encoder.train(was_training)
    return compounds, image_names, np.asarray(vectors, dtype=np.float32)


def save_embedding_artifact(path: Path, compounds, image_names, embeddings, metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        compounds=np.asarray(compounds, dtype=str),
        image_names=np.asarray(image_names, dtype=str),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def evaluate_and_save(path: Path, compounds, embeddings, label_map, n_permutations, metadata):
    result = evaluate_representation(
        list(zip(compounds, embeddings)),
        label_map=label_map,
        n_permutations=n_permutations,
    )
    payload = {**metadata, "metrics": result}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch without importing another training program."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AuditedPairSampler(CompoundPairSampler):
    """Shared physical-replicate sampler plus repeated-epoch detection."""

    def __init__(self, dataset, batch_compounds=32, seed=0):
        super().__init__(dataset, batch_compounds=batch_compounds, seed=seed)
        self.previous_fingerprint = None
        self.epoch = 0

    def __iter__(self):
        epoch_batches = []
        for indices in super().__iter__():
            epoch_batches.append(tuple(indices))
            yield indices
        fingerprint = tuple(epoch_batches)
        if self.previous_fingerprint == fingerprint:
            raise RuntimeError(
                "sampler replayed the exact previous epoch; refusing to train on "
                "repeated batches"
            )
        self.previous_fingerprint = fingerprint
        self.epoch += 1


def augment_batch(batch: dict, node_dropout: float, edge_dropout: float) -> dict:
    """Return an augmented copy without mutating DataLoader-owned tensors."""
    output = dict(batch)
    x = batch["x"].clone()
    if node_dropout:
        x *= torch.bernoulli(torch.full_like(x, 1.0 - node_dropout))
    output["x"] = x
    if edge_dropout:
        edge_indices, edge_attributes = [], []
        for edge_index, edge_attr in zip(batch["edge_index"], batch["edge_attr"]):
            keep = torch.rand(edge_index.shape[1]) > edge_dropout
            edge_indices.append(edge_index[:, keep])
            edge_attributes.append(edge_attr[keep])
        output["edge_index"] = edge_indices
        output["edge_attr"] = edge_attributes
    return output


def format_optional(value):
    """Format a diagnostic metric that can be unavailable for a small fold."""
    return "n/a" if value is None else f"{value:.4f}"
