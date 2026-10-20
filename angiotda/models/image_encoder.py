"""DinoV3 and Cell-DINO image encoders for single-channel microscopy images."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """Image encoder with configurable grayscale handling and projection head.

    ``cell_dino_cp_vits8`` loads the official five-channel Cell-DINO
    Cell-Painting checkpoint. ``input_mode="learned_adapter"`` uses the legacy
    learnable 1x1 convolution. ``input_mode="repeat"`` has no adapter
    parameters and copies the grayscale plane into every backbone channel.
    """

    _DIMS = {
        "dinov3_vits16": 384,
        "dinov3_vitb16": 768,
        "dinov3_vitl16": 1024,
        "dinov3_vith16plus": 1280,
        "cell_dino_cp_vits8": 384,
    }

    def __init__(
        self,
        output_dim: int = 256,
        model_type: str = "dinov3_vits16",
        freeze_backbone: bool = True,
        unfreeze_last_n: int = 0,
        pretrained_path: str | None = None,
        input_mode: str = "learned_adapter",
    ):
        super().__init__()
        if model_type not in self._DIMS:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose from {list(self._DIMS)}")
        if input_mode not in {"learned_adapter", "repeat"}:
            raise ValueError(
                f"Unknown input_mode '{input_mode}'. Choose 'learned_adapter' or 'repeat'.")
        self._feat_dim = self._DIMS[model_type]
        self.input_mode = input_mode

        import os
        if model_type == "cell_dino_cp_vits8":
            repo = os.environ.get(
                "CELL_DINO_REPO",
                "",
            )
            if not os.path.isdir(repo):
                raise FileNotFoundError(f"Cell-DINO repository not found: {repo}")
            if pretrained_path is None:
                raise ValueError("pretrained_path is required for cell_dino_cp_vits8")
            self.encoder = torch.hub.load(
                repo, "cell_dino_cp_vits8", source="local",
                pretrained_path=pretrained_path, in_channels=5,
            )
            input_channels = 5
        else:
            _local = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov3_main")
            if os.path.isdir(_local):
                self.encoder = torch.hub.load(_local, model_type, source="local")
            else:
                self.encoder = torch.hub.load("facebookresearch/dinov3", model_type)
            input_channels = 3

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
            if unfreeze_last_n > 0 and hasattr(self.encoder, "blocks"):
                for block in self.encoder.blocks[-unfreeze_last_n:]:
                    for p in block.parameters():
                        p.requires_grad = True
                if hasattr(self.encoder, "norm"):
                    for p in self.encoder.norm.parameters():
                        p.requires_grad = True

        self.input_channels = input_channels
        if input_mode == "learned_adapter":
            self.input_adapter = nn.Conv2d(
                1, input_channels, kernel_size=1, bias=True)
            if model_type == "cell_dino_cp_vits8":
                nn.init.ones_(self.input_adapter.weight)
            else:
                nn.init.kaiming_normal_(self.input_adapter.weight, mode="fan_out")
            nn.init.zeros_(self.input_adapter.bias)
        else:
            self.input_adapter = None

        self.projection = nn.Sequential(
            nn.Linear(self._feat_dim, self._feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self._feat_dim // 2, output_dim),
        )
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def backbone_dim(self) -> int:
        return self._feat_dim

    def _backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                f"ImageEncoder expects BCHW single-channel input, got {tuple(x.shape)}")
        if self.input_mode == "repeat":
            x = x.repeat(1, self.input_channels, 1, 1)
        else:
            x = self.input_adapter(x)
        if not any(p.requires_grad for p in self.encoder.parameters()):
            with torch.no_grad():
                return self.encoder(x)
        return self.encoder(x)

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self._backbone_features(x), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self._backbone_features(x)), dim=-1)
