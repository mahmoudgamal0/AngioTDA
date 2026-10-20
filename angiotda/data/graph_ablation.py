"""Controlled graph ablations shared by training and evaluation."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

import torch


GRAPH_ABLATIONS = (
    "none", "rewire_edges", "no_edges", "no_node_features",
    "no_edge_features", "no_intensity_features", "no_position_features",
)
_REWIRE_CACHE: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def degree_preserving_rewire(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    generator: torch.Generator,
    swaps_per_edge: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomise topology with double-edge swaps while preserving node degree."""
    if edge_index.shape[1] < 2:
        return edge_index.clone(), edge_attr.clone()

    device = edge_index.device
    edges = [(int(u), int(v)) for u, v in edge_index.t().cpu().tolist()]
    edge_counts = Counter(edges)
    symmetric = all(
        source == target
        or count == edge_counts[(target, source)]
        for (source, target), count in edge_counts.items()
    )

    if symmetric:
        edge_positions = defaultdict(list)
        for index, edge in enumerate(edges):
            edge_positions[edge].append(index)
        undirected_keys = sorted({
            (min(source, target), max(source, target))
            for source, target in edges if source != target
        })
        loops = []
        for index, (source, target) in enumerate(edges):
            if source == target:
                loops.append(((source, target), edge_attr[index].clone()))
        work, attrs = [], []
        for source, target in undirected_keys:
            forward = edge_positions[(source, target)]
            reverse = edge_positions[(target, source)]
            for forward_index, reverse_index in zip(forward, reverse):
                work.append((source, target))
                attrs.append((edge_attr[forward_index].clone(),
                              edge_attr[reverse_index].clone()))
        occupied = Counter(work)
        target_swaps = swaps_per_edge * len(work)
        accepted = attempts = 0
        max_attempts = max(100, target_swaps * 20)
        while accepted < target_swaps and attempts < max_attempts and len(work) >= 2:
            attempts += 1
            left, right = torch.randint(
                0, len(work), (2,), generator=generator).tolist()
            if left == right:
                continue
            a, b = work[left]
            c, d = work[right]
            if len({a, b, c, d}) < 4:
                continue
            candidates = ((a, d), (c, b)) if int(torch.randint(
                0, 2, (1,), generator=generator)) else ((a, c), (b, d))
            edge_1, edge_2 = (tuple(sorted(edge)) for edge in candidates)
            old_1, old_2 = work[left], work[right]
            remaining_1 = (occupied[edge_1] - int(old_1 == edge_1)
                           - int(old_2 == edge_1))
            remaining_2 = (occupied[edge_2] - int(old_1 == edge_2)
                           - int(old_2 == edge_2))
            if edge_1 == edge_2 or remaining_1 > 0 or remaining_2 > 0:
                continue
            occupied[old_1] -= 1
            occupied[old_2] -= 1
            occupied[edge_1] += 1
            occupied[edge_2] += 1
            work[left], work[right] = edge_1, edge_2
            accepted += 1

        out_edges, out_attrs = [], []
        for (source, target), (forward_attr, reverse_attr) in zip(work, attrs):
            out_edges.extend(((source, target), (target, source)))
            out_attrs.extend((forward_attr, reverse_attr))
        for edge, attr in loops:
            out_edges.append(edge)
            out_attrs.append(attr)
        if not out_edges:
            return (edge_index.new_zeros((2, 0)),
                    edge_attr.new_zeros((0, edge_attr.shape[1])))
        return (
            torch.tensor(out_edges, dtype=edge_index.dtype,
                         device=device).t().contiguous(),
            torch.stack(out_attrs).to(edge_attr.device),
        )

    work = list(edges)
    occupied = set(work)
    target_swaps = swaps_per_edge * len(work)
    accepted = attempts = 0
    max_attempts = max(100, target_swaps * 20)
    while accepted < target_swaps and attempts < max_attempts:
        attempts += 1
        left, right = torch.randint(
            0, len(work), (2,), generator=generator).tolist()
        if left == right:
            continue
        a, b = work[left]
        c, d = work[right]
        edge_1, edge_2 = (a, d), (c, b)
        old_1, old_2 = work[left], work[right]
        remaining = occupied - {old_1, old_2}
        if (a == d or c == b or edge_1 == edge_2
                or edge_1 in remaining or edge_2 in remaining):
            continue
        occupied.remove(old_1)
        occupied.remove(old_2)
        occupied.add(edge_1)
        occupied.add(edge_2)
        work[left], work[right] = edge_1, edge_2
        accepted += 1
    return (
        torch.tensor(work, dtype=edge_index.dtype,
                     device=device).t().contiguous(),
        edge_attr.clone(),
    )


def _stable_seed(base_seed: int, image_name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{image_name}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def ablate_graph_batch(batch: dict, mode: str, seed: int = 0) -> dict:
    """Return a non-mutating, deterministic ablation of a collated graph batch."""
    if mode not in GRAPH_ABLATIONS:
        raise ValueError(f"unknown graph ablation {mode!r}")
    if mode == "none":
        return batch

    output = dict(batch)
    if mode == "no_position_features":
        output["x"] = batch["x"].clone()
        output["x"][..., :2] = 0
        return output

    if mode == "no_intensity_features":
        output["edge_attr"] = [attr.clone() for attr in batch["edge_attr"]]
        for attr in output["edge_attr"]:
            attr[:, 6:8] = 0
        return output

    if mode == "no_node_features":
        output["x"] = torch.zeros_like(batch["x"])
        return output

    if mode == "no_edge_features":
        output["edge_attr"] = [torch.zeros_like(attr) for attr in batch["edge_attr"]]
        return output

    if mode == "no_edges":
        output["edge_index"] = [edge.new_zeros((2, 0))
                                for edge in batch["edge_index"]]
        output["edge_attr"] = [attr.new_zeros((0, attr.shape[1]))
                               for attr in batch["edge_attr"]]
        return output

    rewired = []
    for image_name, edge_index, edge_attr in zip(
            batch["image_name"], batch["edge_index"], batch["edge_attr"]):
        key = (seed, str(image_name))
        if key not in _REWIRE_CACHE:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_stable_seed(seed, str(image_name)))
            _REWIRE_CACHE[key] = degree_preserving_rewire(
                edge_index, edge_attr, generator)
        rewired.append(_REWIRE_CACHE[key])
    output["edge_index"] = [pair[0] for pair in rewired]
    output["edge_attr"] = [pair[1] for pair in rewired]
    return output
