"""Quantization codebooks: E2M1 (NVFP4), Uniform-16, INT4, and utilities."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class Codebook:
    """A quantization codebook with named levels."""
    name: str
    levels: torch.Tensor  # sorted ascending, shape (N,)
    signed: bool = True

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    @property
    def max_abs(self) -> float:
        return float(self.levels[-1].item())

    @property
    def spacing(self) -> torch.Tensor:
        """Spacing between adjacent positive levels."""
        pos = self.levels[self.levels > 0]
        return pos[1:] - pos[:-1]

    def quantize(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Quantize x / scale to nearest codebook level, clip to range.

        Args:
            x: tensor to quantize, any shape
            scale: per-block scale, broadcastable to x

        Returns:
            Quantized tensor, same shape as x: scale * Q(x / scale)
        """
        x_scaled = x / scale
        # Find nearest codebook level
        x_clipped = x_scaled.clamp(-self.max_abs, self.max_abs)
        indices = torch.searchsorted(
            self.levels.to(x.device),
            x_clipped.flatten()
        ).reshape(x_clipped.shape)

        # Handle values between levels
        levels_expanded = self.levels.to(x.device)
        lower = levels_expanded[indices - 1]
        upper = levels_expanded[torch.clamp(indices, max=self.num_levels - 1)]
        mid = (lower + upper) / 2

        xq_scaled = torch.where(x_clipped >= mid, upper, lower)
        return scale * xq_scaled

    def quantize_round(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Simple round-to-nearest quantization (faster, less precise for edge cases)."""
        x_scaled = x / scale
        levels = self.levels.to(x.device)

        # Compute distances to all codebook levels
        x_flat = x_scaled.flatten().unsqueeze(-1)  # (N, 1)
        dists = torch.abs(x_flat - levels.unsqueeze(0))  # (N, num_levels)
        nearest_idx = dists.argmin(dim=-1)  # (N,)
        nearest = levels[nearest_idx].reshape(x_scaled.shape)

        return scale * nearest.clamp(-self.max_abs, self.max_abs)


# ── Standard Codebooks ──────────────────────────────────────────────

def e2m1_codebook() -> Codebook:
    """NVFP4 E2M1 codebook: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}."""
    pos = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
    neg = -pos
    levels = torch.cat([neg.flip(0)[:-1], pos])
    return Codebook(name="E2M1", levels=levels, signed=True)


def uniform_16_codebook(rmax: float = 7.5) -> Codebook:
    """16-level symmetric uniform codebook: {0, ±δ, ±2δ, ..., ±7δ} with δ = rmax/7.

    Uses rmax=7.5 to match E2M1's maximum of 6.0 as closely as possible.
    """
    delta = rmax / 7.0
    pos = torch.arange(0, 8, dtype=torch.float32) * delta  # {0, δ, 2δ, ..., 7δ}
    neg = -pos
    levels = torch.cat([neg.flip(0)[:-1], pos])
    return Codebook(name=f"Uniform-16(r={rmax})", levels=levels, signed=True)


def int4_codebook() -> Codebook:
    """INT4 symmetric uniform codebook: {0, ±1, ±2, ..., ±7}."""
    pos = torch.arange(0, 8, dtype=torch.float32)
    neg = -pos
    levels = torch.cat([neg.flip(0)[:-1], pos])
    return Codebook(name="INT4", levels=levels, signed=True)


def uniform_codebook(num_levels_positive: int = 7, max_val: float = 7.0) -> Codebook:
    """Generic uniform symmetric codebook.

    Args:
        num_levels_positive: number of positive levels (excluding zero)
        max_val: maximum representable absolute value
    """
    pos = torch.linspace(0, max_val, num_levels_positive + 1, dtype=torch.float32)
    neg = -pos
    levels = torch.cat([neg.flip(0)[:-1], pos])
    return Codebook(name=f"Uniform({num_levels_positive})", levels=levels, signed=True)


# ── Codebook Registry ───────────────────────────────────────────────

CODECBOOK_REGISTRY: Dict[str, Codebook] = {
    "E2M1": e2m1_codebook(),
    "Uniform-16": uniform_16_codebook(),
    "INT4": int4_codebook(),
}


def get_codebook(name: str) -> Codebook:
    if name in CODECBOOK_REGISTRY:
        return CODECBOOK_REGISTRY[name]
    raise KeyError(f"Unknown codebook: {name}. Available: {list(CODECBOOK_REGISTRY.keys())}")
