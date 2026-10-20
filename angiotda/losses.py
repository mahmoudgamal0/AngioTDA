"""Contrastive losses: NT-Xent over replicates, and cross-modal CLIP."""

import torch
import torch.nn.functional as F


class NTXentLoss(torch.nn.Module):
    """NT-Xent loss over compound replicates.

    Given a batch of 2B embeddings (B compounds × 2 wells each),
    positive pairs are the two wells of the same compound.
    All other 2(B-1) wells are negatives — never a well of the same compound:
    any same-compound well that is not the designated positive is removed from
    the denominator rather than being scored as a negative.

    Args:
        temperature: scaling factor for cosine similarities (default 0.07)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor, compounds: list[str]) -> dict:
        """
        Args:
            z: (2B, D) L2-normalized embeddings
            compounds: list of 2B compound names

        Returns:
            dict with 'loss' tensor and 'n_pairs' int
        """
        N = z.shape[0]
        device = z.device

        # Build positive pair mask: (N, N) where mask[i,j] = True if same compound and i != j
        compound_ids = {}
        for i, c in enumerate(compounds):
            compound_ids.setdefault(c, []).append(i)

        # Cosine similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # (N, N)

        # Mask out self-similarity
        mask_self = torch.eye(N, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask_self, -1e9)

        # For each sample i, find its positive j (same compound, other well)
        positive_idx = torch.zeros(N, dtype=torch.long, device=device)
        valid = torch.ones(N, dtype=torch.bool, device=device)

        for c, indices in compound_ids.items():
            if len(indices) == 2:
                i, j = indices
                positive_idx[i] = j
                positive_idx[j] = i
            elif len(indices) > 2:
                # More than 2 wells: pick one random positive (shouldn't happen normally)
                for idx in indices:
                    others = [o for o in indices if o != idx]
                    positive_idx[idx] = others[0]
            else:
                # Only 1 well: skip
                valid[indices[0]] = False

        if valid.sum() == 0:
            return {"loss": torch.tensor(0.0, device=device), "n_pairs": 0}

        # A well of the same compound is never a negative. With exactly 2 wells
        # per compound the only same-compound entry is the positive itself, so
        # this is a no-op there; it matters when a compound contributes >2 wells,
        # where the wells not chosen as the positive would otherwise be pushed
        # apart from their own replicate.
        same = torch.zeros(N, N, dtype=torch.bool, device=device)
        for indices in compound_ids.values():
            if len(indices) > 1:
                idx = torch.tensor(indices, device=device)
                same[idx.unsqueeze(1), idx.unsqueeze(0)] = True
        drop = same & ~mask_self
        drop[torch.arange(N, device=device), positive_idx] = False
        sim = sim.masked_fill(drop, -1e9)

        # InfoNCE: -log(exp(sim(i, pos)) / sum_j!=i exp(sim(i, j)))
        sim_valid = sim[valid]
        pos_idx_valid = positive_idx[valid]

        log_softmax = F.log_softmax(sim_valid, dim=1)  # (n_valid, N)
        loss = -log_softmax[torch.arange(len(pos_idx_valid), device=device), pos_idx_valid]

        return {"loss": loss.mean(), "n_pairs": int(valid.sum()) // 2}


class CLIPLoss(torch.nn.Module):
    """Symmetric cross-modal InfoNCE (Radford et al., 2021).

    Given per-well image and graph embeddings, the positive for image i is
    graph i — the *same well* — and every other well in the batch is a negative,
    scored in both directions. This is the term that aligns the two encoders;
    without it, two towers summed before the loss are just late fusion.

    Args:
        mask_same_compound: also exclude the replicate well of the same compound
            from the negatives. The class default is off (faithful CLIP, where a
            replicate is a genuinely different well), but scripts/train/train_clip.py
            turns it on: a replicate is not a meaningful negative here, so
            scoring it as one fights the replicate-consistency objective.
    """

    def __init__(self, mask_same_compound: bool = False):
        super().__init__()
        self.mask_same_compound = mask_same_compound

    def forward(self, z_img: torch.Tensor, z_graph: torch.Tensor,
                logit_scale: torch.Tensor, compounds: list[str] | None = None) -> dict:
        """
        Args:
            z_img:   (B, D) L2-normalized image embeddings
            z_graph: (B, D) L2-normalized graph embeddings
            logit_scale: scalar tensor (already exp'd and clamped)
            compounds: list of B compound names, required if mask_same_compound

        Returns:
            dict with 'loss', and top-1 retrieval accuracy in each direction
        """
        B = z_img.shape[0]
        device = z_img.device

        logits = logit_scale * z_img @ z_graph.t()  # (B, B): rows images, cols graphs
        labels = torch.arange(B, device=device)

        if self.mask_same_compound:
            if compounds is None:
                raise ValueError("mask_same_compound=True requires `compounds`")
            same = torch.tensor(
                [[a == b for b in compounds] for a in compounds], device=device)
            off_diag = same & ~torch.eye(B, dtype=torch.bool, device=device)
            logits = logits.masked_fill(off_diag, -1e4)

        loss_i2g = F.cross_entropy(logits, labels)
        loss_g2i = F.cross_entropy(logits.t(), labels)
        loss = 0.5 * (loss_i2g + loss_g2i)

        with torch.no_grad():
            acc_i2g = (logits.argmax(dim=1) == labels).float().mean().item()
            acc_g2i = (logits.t().argmax(dim=1) == labels).float().mean().item()

        return {
            "loss": loss,
            "loss_i2g": loss_i2g.detach(),
            "loss_g2i": loss_g2i.detach(),
            "acc_i2g": acc_i2g,
            "acc_g2i": acc_g2i,
            "n_pairs": B,
        }
