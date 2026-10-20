"""Losses for two image wells + two graph wells per compound.

The joint term is the supervised-contrastive form of InfoNCE: each of the four
views is an anchor and its other three same-compound views are positives.  This
is deliberately identity-driven rather than position-driven, so a shuffled
batch cannot silently corrupt the positive map.
"""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F


def _validate_two_wells(compounds: list[str], n: int) -> None:
    if len(compounds) != n:
        raise ValueError(f"got {len(compounds)} compound labels for {n} embeddings")
    bad = {c: k for c, k in Counter(compounds).items() if k != 2}
    if bad:
        raise ValueError(
            "four-view training requires exactly two wells per compound; "
            f"invalid counts: {bad}"
        )


def positive_mask(labels: list[str], device=None) -> torch.Tensor:
    """Return the non-diagonal same-identity positive mask."""
    same = torch.tensor(
        [[a == b for b in labels] for a in labels],
        dtype=torch.bool,
        device=device,
    )
    return same & ~torch.eye(len(labels), dtype=torch.bool, device=device)


def multi_positive_info_nce(
    embeddings: torch.Tensor,
    labels: list[str],
    temperature: float,
) -> torch.Tensor:
    """Multi-positive InfoNCE averaged over positives and anchors."""
    if embeddings.ndim != 2 or embeddings.shape[0] != len(labels):
        raise ValueError("embeddings must be (N,D) with one label per row")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    z = F.normalize(embeddings, dim=-1)
    logits = z @ z.t() / temperature
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positives = positive_mask(labels, z.device)
    counts = positives.sum(dim=1)
    if bool((counts == 0).any()):
        missing = [labels[i] for i in torch.where(counts == 0)[0].tolist()]
        raise ValueError(f"anchors without a positive: {missing}")
    return -((log_prob.masked_fill(~positives, 0.0).sum(dim=1) / counts).mean())


class FourViewObjective(torch.nn.Module):
    """Image NT-Xent + graph NT-Xent + joint four-view InfoNCE."""

    def __init__(
        self,
        temperature: float = 0.07,
        image_weight: float = 1.0,
        graph_weight: float = 1.0,
        cross_modal_weight: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.image_weight = image_weight
        self.graph_weight = graph_weight
        self.cross_modal_weight = cross_modal_weight

    def forward(
        self,
        z_image: torch.Tensor,
        z_graph: torch.Tensor,
        compounds: list[str],
    ) -> dict[str, torch.Tensor | int]:
        if z_image.shape != z_graph.shape:
            raise ValueError("image and graph embeddings must have the same shape")
        _validate_two_wells(compounds, z_image.shape[0])

        image_loss = multi_positive_info_nce(z_image, compounds, self.temperature)
        graph_loss = multi_positive_info_nce(z_graph, compounds, self.temperature)

        # [all images, all graphs]; every anchor has exactly three positives:
        # the replicate in its modality and both views in the other modality.
        joint_z = torch.cat((z_image, z_graph), dim=0)
        joint_labels = compounds + compounds
        joint_loss = multi_positive_info_nce(joint_z, joint_labels, self.temperature)
        total = (
            self.image_weight * image_loss
            + self.graph_weight * graph_loss
            + self.cross_modal_weight * joint_loss
        )
        return {
            "loss": total,
            "image_ntxent": image_loss.detach(),
            "graph_ntxent": graph_loss.detach(),
            "four_view_infonce": joint_loss.detach(),
            "n_compounds": len(set(compounds)),
        }

