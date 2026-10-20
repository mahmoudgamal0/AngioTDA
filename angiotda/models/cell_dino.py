"""Cell-DINO baseline: the official Cell-Painting ViT-S/8, frozen.

This is a *pretrained* baseline, not a model we train. It carries no
checkpoint of its own: the weights are the published Cell-DINO Cell-Painting
release, loaded as-is, and the only thing between the assay image and the
backbone is a fixed 1->5 channel replication. Evaluating it therefore answers
"how far does an off-the-shelf Cell-Painting representation get on tube
formation?", with no fitting to this dataset at all.

Set `CELL_DINO_REPO` to the upstream local repository and `CELL_DINO_WEIGHTS`
to its released Cell-Painting weights. An explicit `pretrained_path` argument
can override the weights path.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_DIM = 384
INPUT_CHANNELS = 5
# The `cell_dino_cp_vits8` entrypoint hardcodes img_size=128 and does not accept
# an override, so the released weights carry a 16x16 positional-embedding grid.
# `img_size` below therefore sets the *input* edge, not the backbone geometry.
NATIVE_IMG_SIZE = 128


def _resolve_weights(pretrained_path: str | Path | None) -> Path:
    if pretrained_path is not None:
        return Path(pretrained_path)
    env = os.environ.get("CELL_DINO_WEIGHTS")
    if env:
        return Path(env)
    raise FileNotFoundError(
        "Cell-DINO weights not found. Pass pretrained_path or set CELL_DINO_WEIGHTS."
    )


def _resolve_repo() -> str:
    repo = os.environ.get("CELL_DINO_REPO")
    if repo is None or not os.path.isdir(repo):
        raise FileNotFoundError("Set CELL_DINO_REPO to the upstream Cell-DINO repository.")
    return repo



class CellDINOEncoder(nn.Module):
    """Frozen Cell-DINO ViT-S/8 with a fixed 1->5 channel adapter.

    Args:
        pretrained_path: Cell-DINO Cell-Painting weights (.pth).
        img_size:        square edge the input is resized to before the forward
                         pass. This does **not** rebuild the backbone: the
                         entrypoint fixes it at 128 (see NATIVE_IMG_SIZE), so
                         any other value goes through DINOv2's positional-
                         embedding interpolation. We evaluate at 224 to match
                         the DINOv3 rows' input size, which means a 28x28 patch
                         grid interpolated up from the pretrained 16x16. The public configuration uses 224-pixel inputs.

    The adapter replicates the single assay channel into the five Cell-Painting
    channels the backbone expects. It is deliberately fixed rather than learned:
    a learned adapter would be trained parameters, which is exactly what this
    baseline is meant not to have.
    """

    def __init__(self, pretrained_path: str | Path | None = None, img_size: int = 224):
        super().__init__()
        self.pretrained_path = str(_resolve_weights(pretrained_path))
        self.img_size = img_size

        self.encoder = torch.hub.load(
            _resolve_repo(), "cell_dino_cp_vits8", source="local",
            pretrained_path=self.pretrained_path, in_channels=INPUT_CHANNELS,
        )
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    @property
    def backbone_dim(self) -> int:
        return FEATURE_DIM

    def train(self, mode: bool = True):
        """Stay in eval mode: nothing here is trainable."""
        return super().train(False)

    def _to_five_channels(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == INPUT_CHANNELS:
            return x
        if x.shape[1] != 1:
            raise ValueError(f"expected 1 or {INPUT_CHANNELS} channels, got {x.shape[1]}")
        return x.expand(-1, INPUT_CHANNELS, -1, -1)

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """(B,1,H,W) -> (B, 384) L2-normalised backbone features."""
        with torch.no_grad():
            feats = self.encoder(self._to_five_channels(x))
        return F.normalize(feats, dim=-1)

    forward = encode_features

    def provenance(self) -> dict:
        """Everything needed to reproduce this baseline — there is no checkpoint."""
        return {
            "family": "cell_dino_frozen",
            "model_type": "cell_dino_cp_vits8",
            "pretrained_path": self.pretrained_path,
            "in_channels": INPUT_CHANNELS,
            "input_adapter": "single channel replicated to five (fixed, not learned)",
            "trainable_parameters": 0,
            "img_size": self.img_size,
            "native_img_size": NATIVE_IMG_SIZE,
            "pos_embed_interpolated": self.img_size != NATIVE_IMG_SIZE,
            "patch_grid": self.img_size // 8,
        }
