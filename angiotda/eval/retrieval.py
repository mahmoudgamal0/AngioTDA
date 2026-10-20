"""Cross-replicate biological-neighbour retrieval metrics.

The query and gallery contain different physical replicates.  The identical
compound is always removed before ranking, so this measures biological
neighbourhood structure rather than compound identity retrieval.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable

import numpy as np


def _labels(value) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value} if value else set()
    return {str(item) for item in value if str(item)}


def label_jaccard(left, right) -> float:
    """Jaccard similarity between two possibly multi-label annotations."""
    a, b = _labels(left), _labels(right)
    union = a | b
    return float(len(a & b) / len(union)) if union else 0.0


def tfidf_profile_vectors(profile_map: dict[str, list[str]],
                          compounds=None) -> dict[str, np.ndarray]:
    """Convert sparse categorical profiles to unit TF-IDF vectors.

    Document frequency and vocabulary are fitted only on ``compounds`` (or all
    nonempty profiles when omitted). This is appropriate for ChEMBL target
    lists, whose common promiscuous targets should contribute less relevance
    than rarer targets. Empty profiles are deliberately omitted rather than
    being interpreted as confirmed biological negatives.
    """
    selected = sorted(
        compound for compound in (compounds if compounds is not None else profile_map)
        if _labels(profile_map.get(compound)))
    if not selected:
        return {}
    document_frequency = Counter(
        item for compound in selected for item in _labels(profile_map[compound]))
    vocabulary = sorted(document_frequency)
    index = {item: position for position, item in enumerate(vocabulary)}
    n_documents = len(selected)
    idf = np.asarray([
        np.log(n_documents / document_frequency[item])
        for item in vocabulary
    ], dtype=np.float64)
    vectors = {}
    for compound in selected:
        vector = np.zeros(len(vocabulary), dtype=np.float64)
        for item in _labels(profile_map[compound]):
            vector[index[item]] = idf[index[item]]
        norm = float(np.linalg.norm(vector))
        # A target present in every profile has IDF zero. If that is the only
        # feature, the profile contains no discriminative retrieval signal.
        if norm > 0:
            vectors[compound] = vector / norm
    return vectors


def profile_cosine(left, right) -> float:
    """Cosine relevance between unit profile vectors."""
    a, b = np.asarray(left), np.asarray(right)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("profile vectors must be one-dimensional with equal size")
    return float(np.clip(a @ b, 0.0, 1.0))


def _has_annotation(value) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim == 1 and bool(np.any(value != 0))
    return bool(_labels(value))


def _average_precision(binary_relevance: np.ndarray) -> float:
    positive = int(binary_relevance.sum())
    if positive == 0:
        return float("nan")
    precision = np.cumsum(binary_relevance) / np.arange(
        1, len(binary_relevance) + 1)
    return float((precision * binary_relevance).sum() / positive)


def _ndcg(relevance: np.ndarray, k: int) -> float:
    cutoff = min(k, len(relevance))
    if cutoff == 0:
        return float("nan")
    discount = np.log2(np.arange(2, cutoff + 2))
    dcg = float((relevance[:cutoff] / discount).sum())
    ideal = np.sort(relevance)[::-1][:cutoff]
    idcg = float((ideal / discount).sum())
    return dcg / idcg if idcg > 0 else float("nan")


def evaluate_direction(query_embeddings: dict[str, np.ndarray],
                       gallery_embeddings: dict[str, np.ndarray],
                       label_map: dict,
                       direction: str,
                       k_values: tuple[int, ...] = (5, 10),
                       relevance_fn: Callable = label_jaccard) -> list[dict]:
    """Evaluate one replicate direction and return one row per eligible drug.

    Only annotated gallery compounds are scored.  Queries with no other drug
    sharing any label are not evaluable and are omitted.  This avoids treating
    missing annotations as confirmed biological negatives.
    """
    gallery_compounds = sorted(
        compound for compound in gallery_embeddings
        if _has_annotation(label_map.get(compound)))
    rows = []
    for compound in sorted(query_embeddings):
        query_profile = label_map.get(compound)
        if not _has_annotation(query_profile):
            continue
        candidates = [candidate for candidate in gallery_compounds
                      if candidate != compound]
        relevance = np.asarray([
            relevance_fn(query_profile, label_map[candidate])
            for candidate in candidates
        ], dtype=float)
        binary = relevance > 0
        if not binary.any():
            continue

        gallery = np.stack([gallery_embeddings[candidate]
                            for candidate in candidates])
        similarity = gallery @ query_embeddings[compound]
        order = np.argsort(-similarity, kind="stable")
        ranked_relevance = relevance[order]
        ranked_binary = binary[order].astype(float)
        row = {
            "compound": compound,
            "direction": direction,
            "n_candidates": len(candidates),
            "n_relevant": int(binary.sum()),
            "average_precision": _average_precision(ranked_binary),
        }
        for k in k_values:
            cutoff = min(k, len(candidates))
            row[f"ndcg_at_{k}"] = _ndcg(ranked_relevance, k)
            row[f"recall_at_{k}"] = float(
                ranked_binary[:cutoff].sum() / binary.sum())
        rows.append(row)
    return rows


def cross_replicate_retrieval(replicate_1: dict[str, np.ndarray],
                              replicate_2: dict[str, np.ndarray],
                              label_map: dict,
                              k_values: tuple[int, ...] = (5, 10),
                              relevance_fn: Callable = label_jaccard) -> list[dict]:
    """Score R1->R2 and R2->R1 biological-neighbour retrieval."""
    return (
        evaluate_direction(
            replicate_1, replicate_2, label_map, "R1_to_R2", k_values,
            relevance_fn)
        + evaluate_direction(
            replicate_2, replicate_1, label_map, "R2_to_R1", k_values,
            relevance_fn)
    )


def compound_metric_means(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Average directions first, leaving compounds as independent units."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["compound"]].append(row)
    metric_keys = sorted(
        key for key in rows[0]
        if key == "average_precision" or key.startswith(("ndcg_at_", "recall_at_"))
    ) if rows else []
    return {
        compound: {
            metric: float(np.mean([row[metric] for row in compound_rows]))
            for metric in metric_keys
        }
        for compound, compound_rows in grouped.items()
    }


def summarize_retrieval(rows: list[dict], n_bootstrap: int = 5000,
                        seed: int = 42) -> dict:
    """Compound-level means and percentile bootstrap confidence intervals."""
    per_compound = compound_metric_means(rows)
    if not per_compound:
        return {"n_query_compounds": 0, "metrics": {}}
    compounds = sorted(per_compound)
    metrics = sorted(per_compound[compounds[0]])
    matrix = np.asarray([
        [per_compound[compound][metric] for metric in metrics]
        for compound in compounds
    ])
    observed = matrix.mean(axis=0)
    rng = np.random.RandomState(seed)
    boot = np.empty((n_bootstrap, len(metrics)), dtype=float)
    for index in range(n_bootstrap):
        sampled = rng.randint(0, len(compounds), size=len(compounds))
        boot[index] = matrix[sampled].mean(axis=0)
    return {
        "n_query_compounds": len(compounds),
        "n_directional_queries": len(rows),
        "bootstrap_unit": "compound",
        "n_bootstrap": n_bootstrap,
        "metrics": {
            metric: {
                "mean": float(observed[index]),
                "ci_low": (float(np.percentile(boot[:, index], 2.5))
                           if n_bootstrap else float(observed[index])),
                "ci_high": (float(np.percentile(boot[:, index], 97.5))
                            if n_bootstrap else float(observed[index])),
            }
            for index, metric in enumerate(metrics)
        },
    }
