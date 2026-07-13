"""Per-token activation quantization with rotation awareness.

Models the asymmetric treatment of activations vs weights:
- Weights: per-block quantization along output dim (static, known after training)
- Activations: per-token dynamic quantization (input-dependent, computed at runtime)

This asymmetry is why rotation helps activations more than weights:
  - Rotation makes per-channel distributions uniform → outlier-free
  - Per-token quantization then operates on well-behaved distributions
  - The benefit shows up in activation error, not weight error
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .codebooks import Codebook, e2m1_codebook, uniform_16_codebook, int4_codebook
from .scales import quantize_scale


@dataclass
class ActQuantResult:
    """Result of per-token activation quantization."""
    x_quant: torch.Tensor       # quantized activations, shape (seq_len, d_model)
    x_orig: torch.Tensor        # original activations
    codebook_name: str
    scale_format: str

    # Per-token metrics
    scales_optimal: torch.Tensor   # (seq_len, 1) — optimal per-token scales
    scales_quant: torch.Tensor     # (seq_len, 1) — quantized scales
    n_clipped: int

    @property
    def mse(self) -> float:
        return float(((self.x_quant - self.x_orig) ** 2).mean().item())

    @property
    def qsnr_db(self) -> float:
        signal = (self.x_orig ** 2).mean()
        noise = ((self.x_quant - self.x_orig) ** 2).mean()
        if noise == 0:
            return float('inf')
        return float((10 * torch.log10(signal / noise)).item())

    @property
    def per_token_qsnr_db(self) -> torch.Tensor:
        """QSNR per token, shape (seq_len,)."""
        signal = (self.x_orig ** 2).mean(dim=-1)
        noise = ((self.x_quant - self.x_orig) ** 2).mean(dim=-1)
        noise = noise.clamp(min=1e-12)
        return 10 * torch.log10(signal / noise)


def per_token_quantize(
    x: torch.Tensor,
    codebook: Codebook,
    scale_format: str = "FP16",
) -> ActQuantResult:
    """Per-token dynamic quantization of activation matrix.

    Each token (row) gets its own scale factor: scale_t = max(|x_t|) / codebook_max

    This is the standard approach for activation quantization (SmoothQuant, etc.).
    The key weakness: outlier channels dominate the per-token scale, crushing
    precision for normal channels. Rotation fixes this.

    Args:
        x: activation matrix (seq_len, d_model)
        codebook: quantization codebook
        scale_format: how to quantize per-token scales

    Returns:
        ActQuantResult
    """
    seq_len, d_model = x.shape

    # Per-token optimal scales: max_abs along hidden dim
    max_abs = x.abs().max(dim=-1).values  # (seq_len,)
    scales_optimal = (max_abs / codebook.max_abs).clamp(min=1e-12)  # (seq_len,)

    # Quantize scales
    scales_quant = quantize_scale(scales_optimal, scale_format).clamp(min=1e-12)

    # Quantize: round(x_t / s_t) → clip → multiply back
    x_scaled = x / scales_quant.unsqueeze(-1)
    xq = codebook.quantize_round(x, scales_quant.unsqueeze(-1))

    n_clipped = int((x_scaled.abs() > codebook.max_abs).sum().item())

    return ActQuantResult(
        x_quant=xq,
        x_orig=x,
        codebook_name=codebook.name,
        scale_format=scale_format,
        scales_optimal=scales_optimal,
        scales_quant=scales_quant,
        n_clipped=n_clipped,
    )


def per_tensor_quantize(
    x: torch.Tensor,
    codebook: Codebook,
    scale_format: str = "FP16",
) -> ActQuantResult:
    """Per-tensor (static) activation quantization — single scale for all tokens.

    Used as a baseline: static quantization is much harder with outlier channels
    because one global scale must cover both outliers and normal values.
    """
    max_abs = x.abs().max()
    scale = (max_abs / codebook.max_abs).clamp(min=1e-12)
    scale_q = quantize_scale(scale.unsqueeze(0), scale_format).item()
    scale_q = max(scale_q, 1e-12)

    xq = codebook.quantize_round(x, torch.tensor(scale_q, device=x.device))
    n_clipped = int(((x / scale_q).abs() > codebook.max_abs).sum().item())

    return ActQuantResult(
        x_quant=xq,
        x_orig=x,
        codebook_name=codebook.name,
        scale_format=scale_format,
        scales_optimal=torch.full((x.shape[0],), scale),
        scales_quant=torch.full((x.shape[0],), scale_q),
        n_clipped=n_clipped,
    )


def per_group_quantize(
    x: torch.Tensor,
    codebook: Codebook,
    group_size: int,
    scale_format: str = "FP16",
) -> ActQuantResult:
    """Per-group activation quantization (group along hidden dim).

    Intermediate between per-tensor and per-token: groups of `group_size`
    consecutive channels share a scale. This is the activation-side analog
    of block-structured weight quantization.

    Args:
        x: (seq_len, d_model)
        codebook: codebook
        group_size: channels per group
        scale_format: scale quantization format
    """
    seq_len, d_model = x.shape
    assert d_model % group_size == 0, f"d_model={d_model} not divisible by group_size={group_size}"

    n_groups = d_model // group_size
    x_groups = x.view(seq_len, n_groups, group_size)

    # Per-group scales
    max_abs = x_groups.abs().max(dim=-1).values  # (seq_len, n_groups)
    scales_optimal = (max_abs / codebook.max_abs).clamp(min=1e-12)
    scales_quant = quantize_scale(scales_optimal, scale_format).clamp(min=1e-12)

    xq_groups = codebook.quantize_round(x_groups, scales_quant.unsqueeze(-1))
    xq = xq_groups.view(seq_len, d_model)

    n_clipped = int(((x_groups / scales_quant.unsqueeze(-1)).abs() > codebook.max_abs).sum().item())

    return ActQuantResult(
        x_quant=xq,
        x_orig=x,
        codebook_name=codebook.name,
        scale_format=scale_format,
        scales_optimal=scales_optimal.flatten(),
        scales_quant=scales_quant.flatten(),
        n_clipped=n_clipped,
    )
