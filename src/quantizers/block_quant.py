"""Block-structured quantization: INT4, NVFP4, MXFP4, and configurable generic quantizer."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .codebooks import Codebook, e2m1_codebook, int4_codebook, uniform_16_codebook
from .scales import quantize_scale


@dataclass
class QuantizeResult:
    """Result of a block-structured quantization."""
    w_quant: torch.Tensor      # quantized weight matrix
    w_orig: torch.Tensor       # original weight matrix
    block_size: int
    scale_format: str
    codebook_name: str

    # Per-block metrics
    scales_optimal: torch.Tensor   # optimal scales (before format quantization)
    scales_quant: torch.Tensor     # quantized scales (after format quantization)
    error_per_block: torch.Tensor  # per-block MSE, shape (num_blocks,)
    n_clipped: int                 # total elements clipped

    @property
    def mse(self) -> float:
        return float(((self.w_quant - self.w_orig) ** 2).mean().item())

    @property
    def qsnr_db(self) -> float:
        signal = (self.w_orig ** 2).mean()
        noise = ((self.w_quant - self.w_orig) ** 2).mean()
        if noise == 0:
            return float('inf')
        return float((10 * torch.log10(signal / noise)).item())

    @property
    def scale_error(self) -> float:
        """Scale-induced MSE: error from using quantized scale vs optimal scale."""
        return float(self._per_element_scale_error().mean().item())

    @property
    def codebook_error(self) -> float:
        """Codebook MSE: error from mapping to discrete codebook with optimal scale."""
        return float(self._per_element_codebook_error().mean().item())

    def _per_element_scale_error(self) -> torch.Tensor:
        """Per-element component of MSE due to scale quantization."""
        B = self.block_size
        w = self.w_orig.view(-1, B)

        # Reconstruction with quantized scale vs optimal scale
        s_opt = self.scales_optimal  # (N_blocks,)
        s_quant = self.scales_quant  # (N_blocks,)

        x = w / s_opt.unsqueeze(-1)  # normalized values
        xq_opt = x.round().clamp(-6, 6)  # simplified (assumes uniform codebook for this analysis)
        xq_quant = (w / s_quant.unsqueeze(-1)).round().clamp(-6, 6)

        recon_opt = s_opt.unsqueeze(-1) * xq_opt
        recon_quant = s_quant.unsqueeze(-1) * xq_quant

        return (recon_opt - recon_quant) ** 2

    def _per_element_codebook_error(self) -> torch.Tensor:
        """Per-element component of MSE due to codebook discretization."""
        B = self.block_size
        w = self.w_orig.view(-1, B)
        s_opt = self.scales_optimal.unsqueeze(-1)

        x = w / s_opt
        xq = x.round().clamp(-6, 6)

        return (x - xq) ** 2


def block_quantize(
    w: torch.Tensor,
    block_size: int,
    codebook: Codebook,
    scale_format: str = "FP16",
    global_scale_format: Optional[str] = None,
) -> QuantizeResult:
    """Block-structured quantization of a 2D weight matrix.

    Quantizes along dim=1 (output dimension) in blocks of `block_size`.

    Args:
        w: weight matrix, shape (d_in, d_out)
        block_size: number of elements per block (BLK in NVFP4)
        codebook: quantization codebook
        scale_format: how to quantize block scales
        global_scale_format: if set, apply a global FP32 scale first (NVFP4 2-level)

    Returns:
        QuantizeResult with all metrics
    """
    d_in, d_out = w.shape

    # Handle non-divisible d_out
    if d_out % block_size != 0:
        pad = block_size - (d_out % block_size)
        w_padded = torch.nn.functional.pad(w, (0, pad), value=0.0)
    else:
        w_padded = w

    _, d_out_padded = w_padded.shape
    n_blocks = d_out_padded // block_size

    # Reshape to blocks: (d_in, n_blocks, block_size)
    w_blocks = w_padded.view(d_in, n_blocks, block_size)

    # ── Global scaling (NVFP4 2-level) ──
    global_scale = 1.0
    if global_scale_format is not None:
        max_abs_global = w_blocks.abs().max()
        global_scale = (max_abs_global / codebook.max_abs).item()
        if global_scale == 0:
            global_scale = 1.0
        w_blocks = w_blocks / global_scale

    # ── Per-block optimal scales ──
    block_max = w_blocks.abs().max(dim=-1).values  # (d_in, n_blocks)
    scales_optimal = block_max / codebook.max_abs
    scales_optimal = scales_optimal.clamp(min=1e-12)

    # ── Quantize scales ──
    scales_quant = quantize_scale(scales_optimal, scale_format)
    scales_quant = scales_quant.clamp(min=1e-12)

    # ── Quantize values ──
    w_quant_blocks = codebook.quantize_round(w_blocks, scales_quant.unsqueeze(-1))

    # ── Per-block error ──
    error_per_block = ((w_quant_blocks - w_blocks) ** 2).mean(dim=(0, -1))  # (n_blocks,)

    # Clipping count
    x_scaled = w_blocks / scales_quant.unsqueeze(-1)
    n_clipped = int((x_scaled.abs() > codebook.max_abs).sum().item())

    # ── Reshape back ──
    w_quant = w_quant_blocks.view(d_in, d_out_padded) * (global_scale if global_scale_format else 1.0)
    w_orig_scaled = w_padded * (global_scale if global_scale_format else 1.0) if global_scale_format else w_padded

    if d_out % block_size != 0:
        w_quant = w_quant[:, :d_out]
        w_orig_scaled = w_orig_scaled[:, :d_out]

    return QuantizeResult(
        w_quant=w_quant,
        w_orig=w,
        block_size=block_size,
        scale_format=scale_format + (f"+{global_scale_format}" if global_scale_format else ""),
        codebook_name=codebook.name,
        scales_optimal=scales_optimal.flatten(),
        scales_quant=scales_quant.flatten(),
        error_per_block=error_per_block,
        n_clipped=n_clipped,
    )


# ── Convenience quantizers ──────────────────────────────────────────

def int4_quantize(w: torch.Tensor) -> QuantizeResult:
    """INT4 quantization: B=128, FP16 scale, uniform-15 codebook."""
    return block_quantize(w, block_size=128, codebook=int4_codebook(), scale_format="FP16")


def nvfp4_quantize(w: torch.Tensor) -> QuantizeResult:
    """NVFP4 quantization: B=16, E4M3 block scale + FP32 global scale, E2M1 codebook."""
    return block_quantize(
        w, block_size=16, codebook=e2m1_codebook(),
        scale_format="E4M3", global_scale_format="FP32",
    )


def mxfp4_quantize(w: torch.Tensor) -> QuantizeResult:
    """MXFP4 quantization: B=32, E8M0 scale, E2M1 codebook."""
    return block_quantize(w, block_size=32, codebook=e2m1_codebook(), scale_format="E8M0")


QUANTIZER_REGISTRY: Dict[str, callable] = {
    "INT4": int4_quantize,
    "NVFP4": nvfp4_quantize,
    "MXFP4": mxfp4_quantize,
}
