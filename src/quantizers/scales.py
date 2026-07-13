"""Scale format quantization: FP16, E4M3 (NVFP4), E8M0 (MXFP4), FP32 (oracle)."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ScaleFormat:
    """How block scales are quantized."""
    name: str
    mantissa_bits: int
    exponent_bits: int
    bias: int = 0

    @property
    def relative_error_bound(self) -> float:
        """Worst-case relative rounding error."""
        return 2.0 ** -(self.mantissa_bits + 1)


def _fp_quantize(x: torch.Tensor, exp_bits: int, mant_bits: int, bias: int = 0) -> torch.Tensor:
    """Quantize to a minifloat-like format.

    Simplified: quantizes the mantissa to `mant_bits` bits with round-to-nearest.
    Does NOT handle subnormals or overflow — assumed the input range is valid.
    """
    # Find the exponent for each element
    abs_x = x.abs()
    finite_mask = torch.isfinite(abs_x) & (abs_x > 0)

    # For exact zeros, return zeros
    xq = torch.zeros_like(x)

    if finite_mask.any():
        x_finite = x[finite_mask]
        abs_finite = abs_x[finite_mask]

        # log2 gives us the exponent
        log2_val = torch.log2(abs_finite)
        exponent = torch.floor(log2_val)  # integer part
        mantissa = abs_finite / (2.0 ** exponent)  # in [1.0, 2.0)

        # Quantize mantissa
        mantissa_quant = torch.round(mantissa * (2 ** mant_bits)) / (2 ** mant_bits)
        mantissa_quant = mantissa_quant.clamp(1.0, 2.0 - 2.0 ** -mant_bits)

        # Get max exponent
        max_exp = 2 ** (exp_bits - 1) - bias
        min_exp = -bias

        # Simple clamp (no subnormal handling)
        exponent_clamped = exponent.clamp(min_exp, max_exp)

        # Reconstruct
        xq[finite_mask] = (
            torch.sign(x_finite)
            * mantissa_quant
            * (2.0 ** exponent_clamped)
        )

    return xq


def _e4m3_quantize(x: torch.Tensor) -> torch.Tensor:
    """E4M3 format: 4 exponent bits, 3 mantissa bits, bias=7.

    E4M3 range: normalized values approximately 2^{-6} to 2^{7} * (1.875).
    """
    exp_bits, mant_bits, bias = 4, 3, 7
    abs_x = x.abs()
    finite_mask = torch.isfinite(abs_x) & (abs_x > 0)
    xq = torch.zeros_like(x)

    if finite_mask.any():
        x_f = x[finite_mask]
        abs_f = abs_x[finite_mask]

        log2_val = torch.log2(abs_f)
        exponent = torch.floor(log2_val)
        mantissa = abs_f / (2.0 ** exponent)

        # E4M3 mantissa values: {1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875}
        mant_levels = torch.tensor(
            [1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875],
            dtype=torch.float32, device=x.device
        )
        dists = torch.abs(mantissa.unsqueeze(-1) - mant_levels.unsqueeze(0))
        nearest_idx = dists.argmin(dim=-1)
        mant_quant = mant_levels[nearest_idx]

        min_exp = float(-9)  # E4M3 min normal
        max_exp = float(8)
        exponent_clamped = exponent.clamp(min_exp, max_exp)

        xq[finite_mask] = torch.sign(x_f) * mant_quant * (2.0 ** exponent_clamped)

    return xq


def _e8m0_quantize(x: torch.Tensor) -> torch.Tensor:
    """E8M0 format: 8 exponent bits, 0 mantissa bits.

    E8M0 = pure power-of-two scaling: values are 2^n for n in range.
    This is a one-level format where relative error is constant for all inputs.
    """
    abs_x = x.abs()
    finite_mask = torch.isfinite(abs_x) & (abs_x > 0)
    xq = torch.zeros_like(x)

    if finite_mask.any():
        x_f = x[finite_mask]
        abs_f = abs_x[finite_mask]

        # Round to nearest power of 2
        log2_val = torch.log2(abs_f)
        log2_rounded = torch.round(log2_val)

        # E8M0 range approximately -127 to 127
        log2_clamped = log2_rounded.clamp(-127, 127)

        xq[finite_mask] = torch.sign(x_f) * (2.0 ** log2_clamped)

    return xq


def quantize_scale(values: torch.Tensor, format_name: str) -> torch.Tensor:
    """Quantize scale values to the specified format.

    Args:
        values: raw optimal scale values
        format_name: one of {'FP16', 'E4M3', 'E8M0', 'FP32'}

    Returns:
        Quantized scale values, same shape
    """
    if format_name == "FP32":
        return values.clone()
    elif format_name == "FP16":
        return values.to(torch.float16).to(torch.float32)
    elif format_name == "E4M3":
        return _e4m3_quantize(values)
    elif format_name == "E8M0":
        return _e8m0_quantize(values)
    else:
        raise ValueError(f"Unknown scale format: {format_name}")


SCALE_FORMATS = {
    "FP32": ScaleFormat("FP32", 23, 8),
    "FP16": ScaleFormat("FP16", 10, 5),
    "E4M3": ScaleFormat("E4M3", 3, 4),
    "E8M0": ScaleFormat("E8M0", 0, 8),
}
