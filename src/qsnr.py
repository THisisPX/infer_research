"""QSNR computation and error decomposition for block-scaled quantization."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .quantizers.block_quant import QuantizeResult, block_quantize
from .quantizers.codebooks import Codebook, get_codebook, uniform_codebook
from .rotation import apply_hadamard_rotation, pad_to_power_of_2


@dataclass
class QSNRDecomposition:
    """Decomposed quantization error for a single weight matrix."""
    qsnr_raw: float            # QSNR without rotation (dB)
    qsnr_rot: float            # QSNR with rotation (dB)
    rotation_gain: float       # G = qsnr_rot / qsnr_raw (>1 = beneficial)

    # Error components (before rotation)
    mse_raw_total: float
    mse_raw_codebook: float    # codebook discretization error
    mse_raw_scale: float       # scale quantization error

    # Error components (after rotation)
    mse_rot_total: float
    mse_rot_codebook: float
    mse_rot_scale: float

    # Codebook utilization metrics
    codebook_util_raw: Dict[str, float]  # bin_name -> fraction used
    codebook_util_rot: Dict[str, float]

    # Scale distribution metrics
    scale_std_raw: float       # std of optimal block scales
    scale_std_rot: float

    @property
    def gain_linear(self) -> float:
        return self.rotation_gain

    @property
    def scale_contribution(self) -> float:
        """Fraction of total MSE due to scale quantization (post-rotation)."""
        return self.mse_rot_scale / self.mse_rot_total if self.mse_rot_total > 0 else 0.0

    @property
    def codebook_contribution(self) -> float:
        """Fraction of total MSE due to codebook (post-rotation)."""
        return self.mse_rot_codebook / self.mse_rot_total if self.mse_rot_total > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"QSNR raw: {self.qsnr_raw:.2f} dB  |  QSNR rotated: {self.qsnr_rot:.2f} dB",
            f"Gain: {self.rotation_gain:.4f} ({'beneficial' if self.rotation_gain > 1 else 'harmful' if self.rotation_gain < 1 else 'neutral'})",
            f"MSE breakdown (rotated): codebook={self.codebook_contribution:.1%}, scale={self.scale_contribution:.1%}",
            f"Scale std: {self.scale_std_raw:.4f} → {self.scale_std_rot:.4f}",
        ]
        return "\n".join(lines)


def compute_qsnr_decomposition(
    w: torch.Tensor,
    block_size: int,
    codebook: Codebook,
    scale_format: str = "FP16",
    global_scale_format: Optional[str] = None,
) -> QSNRDecomposition:
    """Compute full QSNR decomposition with and without Hadamard rotation.

    Args:
        w: weight matrix (d_in, d_out)
        block_size: quantization block size
        codebook: quantization codebook
        scale_format: block scale quantization format
        global_scale_format: global scale format (NVFP4 2-level)

    Returns:
        QSNRDecomposition with all metrics
    """
    # ── Quantize without rotation ──
    result_raw = block_quantize(
        w, block_size=block_size, codebook=codebook,
        scale_format=scale_format, global_scale_format=global_scale_format,
    )

    # ── Apply rotation ──
    w_padded = pad_to_power_of_2(w, dim=1)
    w_rot = apply_hadamard_rotation(w_padded, dim=1)
    # Trim back to original size
    w_rot = w_rot[:, :w.shape[1]]

    # ── Quantize with rotation ──
    result_rot = block_quantize(
        w_rot, block_size=block_size, codebook=codebook,
        scale_format=scale_format, global_scale_format=global_scale_format,
    )

    # ── Scale distribution metrics ──
    scale_std_raw = float(result_raw.scales_optimal.std().item())
    scale_std_rot = float(result_rot.scales_optimal.std().item())

    # ── QSNR ──
    qsnr_raw = result_raw.qsnr_db
    qsnr_rot = result_rot.qsnr_db
    gain_linear = 10 ** ((qsnr_rot - qsnr_raw) / 10) if qsnr_raw < float('inf') else 1.0

    # ── Codebook utilization ──
    codebook_util_raw = _compute_codebook_utilization(w, result_raw.scales_quant, codebook)
    codebook_util_rot = _compute_codebook_utilization(w_rot, result_rot.scales_quant, codebook)

    return QSNRDecomposition(
        qsnr_raw=qsnr_raw,
        qsnr_rot=qsnr_rot,
        rotation_gain=gain_linear,
        mse_raw_total=result_raw.mse,
        mse_raw_codebook=result_raw.codebook_error,
        mse_raw_scale=result_raw.scale_error,
        mse_rot_total=result_rot.mse,
        mse_rot_codebook=result_rot.codebook_error,
        mse_rot_scale=result_rot.scale_error,
        codebook_util_raw=codebook_util_raw,
        codebook_util_rot=codebook_util_rot,
        scale_std_raw=scale_std_raw,
        scale_std_rot=scale_std_rot,
    )


def _compute_codebook_utilization(
    w: torch.Tensor, scales: torch.Tensor, codebook: Codebook
) -> Dict[str, float]:
    """Compute fraction of values quantized to each codebook bin."""
    B = codebook.num_levels
    # Reshape scales to match blocks
    # This is approximate — we compute per-element utilization
    x = w / (scales.view(-1, 1) + 1e-12)
    levels = codebook.levels

    utilization = {}
    for i, level in enumerate(levels):
        # Find values closest to this level
        if i == 0:
            mask = x <= (levels[0] + levels[1]) / 2
        elif i == len(levels) - 1:
            mask = x > (levels[-2] + levels[-1]) / 2
        else:
            lo = (levels[i - 1] + levels[i]) / 2
            hi = (levels[i] + levels[i + 1]) / 2
            mask = (x > lo) & (x <= hi)

        utilization[f"{float(level):.1f}"] = float(mask.float().mean().item())

    return utilization


def compute_qsnr_matrix(
    w: torch.Tensor,
    block_sizes: List[int],
    scale_formats: List[str],
    codebook_names: List[str],
    with_rotation: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Compute QSNR across a grid of configurations.

    Returns:
        Dict[config_key, {"qsnr": val, "mse": val, ...}]
    """
    results = {}
    for B in block_sizes:
        for sf in scale_formats:
            for cb_name in codebook_names:
                cb = get_codebook(cb_name)

                # Without rotation
                r = block_quantize(w, block_size=B, codebook=cb, scale_format=sf)
                key_raw = f"B={B}/{sf}/{cb_name}/no-rot"
                results[key_raw] = {"qsnr_db": r.qsnr_db, "mse": r.mse}

                if with_rotation:
                    # With rotation
                    w_padded = pad_to_power_of_2(w, dim=1)
                    w_rot = apply_hadamard_rotation(w_padded, dim=1)[:, :w.shape[1]]
                    r_rot = block_quantize(w_rot, block_size=B, codebook=cb, scale_format=sf)
                    key_rot = f"B={B}/{sf}/{cb_name}/hadamard"
                    results[key_rot] = {"qsnr_db": r_rot.qsnr_db, "mse": r_rot.mse}

    return results
