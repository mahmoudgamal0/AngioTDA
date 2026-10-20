"""Training-fold-only biological relevance and weighted contrastive loss."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedMultiPositiveContrastiveLoss(nn.Module):
    """Weighted supervised contrastive loss with explicitly unknown pairs.

    ``relevance`` is an ``(N,N)`` matrix. Positive finite values are weighted
    positives, finite zeros are known negatives, and NaN means unknown. Unknown
    pairs never enter either the numerator or denominator, so missing biological
    annotations are not silently converted to negatives. Same-compound pairs
    are excluded because replicate NT-Xent handles them separately.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, z: torch.Tensor, compounds: list[str],
                relevance: torch.Tensor) -> dict:
        n = z.shape[0]
        if relevance.shape != (n, n):
            raise ValueError(f"relevance must have shape {(n, n)}, got {relevance.shape}")
        if len(compounds) != n:
            raise ValueError("one compound identifier is required per embedding")
        relevance = relevance.to(device=z.device, dtype=z.dtype)
        same_compound = torch.tensor(
            [[left == right for right in compounds] for left in compounds],
            dtype=torch.bool, device=z.device,
        )
        self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
        eligible = torch.isfinite(relevance) & ~same_compound & ~self_mask
        positive = eligible & (relevance > 0)
        valid_anchor = positive.any(dim=1)
        n_anchors = int(valid_anchor.sum().item())
        if n_anchors == 0:
            return {
                "loss": z.sum() * 0.0,
                "n_anchors": 0,
                "n_positive_pairs": 0,
                "n_known_pairs": int(eligible.sum().item()),
            }

        logits = F.normalize(z, dim=-1) @ F.normalize(z, dim=-1).t()
        logits = logits / self.temperature
        logits = logits.masked_fill(~eligible, -torch.inf)
        log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        weights = torch.where(positive, relevance.clamp_min(0), torch.zeros_like(relevance))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        safe_log_probability = torch.where(
            positive, log_probability, torch.zeros_like(log_probability))
        per_anchor = -(weights * safe_log_probability).sum(dim=1)
        return {
            "loss": per_anchor[valid_anchor].mean(),
            "n_anchors": n_anchors,
            "n_positive_pairs": int(positive.sum().item()),
            "n_known_pairs": int(eligible.sum().item()),
        }


@dataclass
class TrainingFoldRelevance:
    """Biological relevance fitted exclusively on training compounds."""

    mode: str
    training_compounds: frozenset[str]
    profiles: dict[str, object]
    audit: dict

    def pair(self, left: str, right: str) -> float:
        if left not in self.profiles or right not in self.profiles:
            return float("nan")
        if self.mode == "moa":
            left_terms, right_terms = self.profiles[left], self.profiles[right]
            union = left_terms | right_terms
            return len(left_terms & right_terms) / len(union) if union else float("nan")
        return float(np.dot(self.profiles[left], self.profiles[right]))

    def batch_matrix(self, compounds: list[str], device=None) -> torch.Tensor:
        # A held-out identifier reaching the optimizer is a hard leakage error,
        # even if it happens to have no annotation.
        unexpected = sorted(set(compounds) - self.training_compounds)
        if unexpected:
            raise AssertionError(f"alignment batch contains non-training compounds: {unexpected[:5]}")
        matrix = np.full((len(compounds), len(compounds)), np.nan, dtype=np.float32)
        for i, left in enumerate(compounds):
            for j, right in enumerate(compounds):
                if i != j:
                    matrix[i, j] = self.pair(left, right)
        return torch.as_tensor(matrix, device=device)


def build_moa_relevance(training_compounds, label_map) -> TrainingFoldRelevance:
    training = frozenset(training_compounds)
    profiles = {
        compound: frozenset(label_map[compound])
        for compound in training
        if label_map.get(compound)
    }
    positive_pairs = sum(
        bool(profiles[left] & profiles[right]) for left, right in combinations(profiles, 2))
    audit = {
        "mode": "moa",
        "fit_scope": "training_fold_only",
        "relevance": "full_multilabel_jaccard",
        "n_training_compounds": len(training),
        "n_annotated_training_compounds": len(profiles),
        "annotation_coverage": len(profiles) / max(len(training), 1),
        "n_positive_compound_pairs": positive_pairs,
        "missing_annotations_policy": "unknown_excluded_not_negative",
        "profile_vocabulary_size": len(set().union(*profiles.values())) if profiles else 0,
    }
    return TrainingFoldRelevance("moa", training, profiles, audit)


def build_target_relevance(training_compounds, target_map) -> TrainingFoldRelevance:
    training = frozenset(training_compounds)
    annotated = {
        compound: sorted(set(target_map[compound]))
        for compound in training
        if target_map.get(compound)
    }
    vocabulary = sorted({target for values in annotated.values() for target in values})
    index = {target: position for position, target in enumerate(vocabulary)}
    n_documents = len(annotated)
    document_frequency = np.zeros(len(vocabulary), dtype=np.float32)
    for values in annotated.values():
        for target in values:
            document_frequency[index[target]] += 1
    idf = np.log(n_documents / np.maximum(document_frequency, 1)) if n_documents else document_frequency
    profiles = {}
    for compound, values in annotated.items():
        vector = np.zeros(len(vocabulary), dtype=np.float32)
        for target in values:
            vector[index[target]] = idf[index[target]]
        norm = float(np.linalg.norm(vector))
        # An all-zero TF-IDF vector has no usable profile and remains unknown.
        if norm > 0:
            profiles[compound] = vector / norm
    positive_pairs = sum(
        float(np.dot(profiles[left], profiles[right])) > 0
        for left, right in combinations(profiles, 2)
    )
    audit = {
        "mode": "targets",
        "fit_scope": "training_fold_only",
        "relevance": "tfidf_profile_cosine",
        "idf_fit_compounds": n_documents,
        "n_training_compounds": len(training),
        "n_annotated_training_compounds": len(profiles),
        "n_raw_annotated_training_compounds": len(annotated),
        "annotation_coverage": len(profiles) / max(len(training), 1),
        "n_positive_compound_pairs": positive_pairs,
        "missing_annotations_policy": "unknown_excluded_not_negative",
        "profile_vocabulary_size": len(vocabulary),
        "vocabulary": vocabulary,
    }
    return TrainingFoldRelevance("targets", training, profiles, audit)
