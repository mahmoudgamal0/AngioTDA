"""Photometric augmentation utilities for fluorescence microscopy images."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class PhotometricAugmentConfig:
    enabled: bool = False
    gaussian_noise_std: float = 0.03
    shot_noise_scale: float = 48.0
    intensity_scale_min: float = 0.8
    intensity_scale_max: float = 1.2
    gamma_min: float = 0.8
    gamma_max: float = 1.2
    gradient_strength_min: float = 0.0
    gradient_strength_max: float = 0.35


def _rand_uniform(low: float, high: float, device: torch.device) -> float:
    if low == high:
        return low
    return float(torch.empty(1, device=device).uniform_(low, high).item())


def _synthetic_gradient(h: int, w: int, device: torch.device) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, h, device=device)
    x = torch.linspace(-1.0, 1.0, w, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    angle = _rand_uniform(0.0, 2.0 * math.pi, device)
    strength = _rand_uniform(0.0, 1.0, device)
    base = math.cos(angle) * xx + math.sin(angle) * yy
    radial = torch.sqrt(xx.square() + yy.square())

    # Mix linear and radial patterns to mimic uneven illumination.
    grad = (1.0 - strength) * base + strength * (radial - radial.mean())
    grad = grad - grad.mean()
    grad = grad / grad.abs().amax().clamp_min(1e-6)
    return grad


def apply_photometric_augment(
    image: torch.Tensor,
    cfg: PhotometricAugmentConfig,
) -> torch.Tensor:
    """Apply a fixed photometric augmentation set to a single-channel image.

    Args:
        image: Tensor of shape (1, H, W) with values in [0, 1].
        cfg: Augmentation configuration.
    """
    if not cfg.enabled:
        return image

    out = image.clone()
    device = out.device

    scale = _rand_uniform(cfg.intensity_scale_min, cfg.intensity_scale_max, device)
    out = out * scale

    gamma = _rand_uniform(cfg.gamma_min, cfg.gamma_max, device)
    out = out.clamp(0.0, 1.0).pow(gamma)

    grad_strength = _rand_uniform(cfg.gradient_strength_min, cfg.gradient_strength_max, device)
    if grad_strength > 0:
        grad = _synthetic_gradient(out.shape[-2], out.shape[-1], device)
        out = out * (1.0 + grad_strength * grad.unsqueeze(0))

    if cfg.shot_noise_scale > 0:
        lam = out.clamp(0.0, 1.0) * cfg.shot_noise_scale
        out = torch.poisson(lam) / cfg.shot_noise_scale

    if cfg.gaussian_noise_std > 0:
        out = out + torch.randn_like(out) * cfg.gaussian_noise_std

    return out.clamp(0.0, 1.0)

