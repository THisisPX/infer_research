"""QSNR computation for joint Weight-Activation (W4A4) quantization.

Key insight: rotation's primary benefit is on ACTIVATION quantization, not weights.
  Y = X · W  =  (X·H) · (H^T·W)
  Rotation X→X·H spreads outlier energy across channels
  → per-token activation quantization of X·H is much more accurate
  → weight quantization of H^T·W is largely unchanged (orthogonal transform)
  → joint output Y_q = (X·H)_q · (H^T·W)_q has lower error
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .quantizers.block_quant import QuantizeResult, block_quantize
from .quantizers.act_quant import (
    ActQuantResult, per_token_quantize,
    per_tensor_quantize, per_group_quantize,
)
from .quantizers.codebooks import Codebook, get_codebook
from .rotation import apply_hadamard_rotation, pad_to_power_of_2


@dataclass
class JointWAQSNR:
    """Full W4A4 quantization analysis for one (X, W) pair."""

    # ── Weight-only metrics ──
    w_qsnr_raw: float = 0.0
    w_qsnr_rot: float = 0.0
    w_gain: float = 1.0

    # ── Activation-only metrics ──
    a_qsnr_raw: float = 0.0      # X_q vs X (no rotation)
    a_qsnr_rot: float = 0.0      # (X·H)_q vs X·H (with rotation)
    a_gain: float = 1.0          # the key metric — rotation gain for activations

    # ── Joint output metrics (the real target) ──
    joint_qsnr_raw: float = 0.0  # Y_q = X_q · W_q vs Y = X · W (no rotation)
    joint_qsnr_rot: float = 0.0  # Y_q = (X·H)_q · (H^T·W)_q vs Y (with rotation)
    joint_gain: float = 1.0      # overall W4A4 rotation gain

    # ── Per-token activation detail ──
    a_qsnr_per_token_raw: List[float] = field(default_factory=list)
    a_qsnr_per_token_rot: List[float] = field(default_factory=list)

    # ── Config ──
    config: Dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"W4A4 Joint QSNR Analysis",
            f"  Weight:   QSNR {self.w_qsnr_raw:.1f} → {self.w_qsnr_rot:.1f} dB  (G={self.w_gain:.4f})",
            f"  Act:      QSNR {self.a_qsnr_raw:.1f} → {self.a_qsnr_rot:.1f} dB  (G={self.a_gain:.4f})",
            f"  Joint:    QSNR {self.joint_qsnr_raw:.1f} → {self.joint_qsnr_rot:.1f} dB  (G={self.joint_gain:.4f})",
            f"  Verdict:  {'BENEFICIAL' if self.joint_gain > 1.01 else 'HARMFUL' if self.joint_gain < 0.99 else 'NEUTRAL'}",
        ]
        return "\n".join(lines)


def compute_wa_qsnr(
    x: torch.Tensor,      # (seq_len, d_model)
    w: torch.Tensor,      # (d_model, d_out)
    w_block_size: int,
    w_codebook: Codebook,
    w_scale_format: str,
    w_global_scale: Optional[str] = None,
    a_codebook: Codebook = None,
    a_scale_format: str = "FP16",
    a_quant_mode: str = "per_token",  # "per_token" | "per_tensor" | "per_group"
    a_group_size: int = 32,
) -> JointWAQSNR:
    """Compute joint W4A4 QSNR with and without Hadamard rotation.

    Rotation: X → X·H, W → H^T·W
    This is the QuaRot/SpinQuant standard: rotate activations and weights
    with the same orthogonal H, preserving Y = X·W exactly in exact arithmetic.

    Args:
        x: activation matrix (seq_len, d_model)
        w: weight matrix (d_model, d_out)
        w_block_size: block size for weight quantization
        w_codebook: weight codebook
        w_scale_format: weight scale quantization format
        w_global_scale: global scale format (NVFP4 2-level)
        a_codebook: activation codebook (defaults to w_codebook)
        a_scale_format: activation scale quantization format
        a_quant_mode: activation quantization mode
        a_group_size: group size for per_group activation quant

    Returns:
        JointWAQSNR with full decomposition
    """
    if a_codebook is None:
        a_codebook = w_codebook

    d_model = x.shape[1]
    d_out = w.shape[1]

    # ── Pad for Hadamard (requires power-of-2) ──
    n_pad = 1
    while n_pad < d_model:
        n_pad <<= 1
    if n_pad > d_model:
        x_pad = torch.nn.functional.pad(x, (0, n_pad - d_model), value=0.0)
        w_pad = torch.nn.functional.pad(w, (0, 0, 0, n_pad - d_model), value=0.0)
    else:
        x_pad, w_pad = x, w

    # ── Quantize activations ──
    aq_fn = {
        "per_token": per_token_quantize,
        "per_tensor": per_tensor_quantize,
        "per_group": lambda x, cb, sf: per_group_quantize(x, cb, a_group_size, sf),
    }[a_quant_mode]

    # Without rotation
    a_raw = aq_fn(x_pad, a_codebook, a_scale_format) if w_block_size > 0 else None

    # With rotation: X·H
    x_rot = apply_hadamard_rotation(x_pad, dim=1)  # rotate last dim (d_model)
    a_rot = aq_fn(x_rot, a_codebook, a_scale_format) if w_block_size > 0 else None

    # ── Quantize weights ──
    # QuaRot identity: Y = X·W = (X·H) (H^T·W)
    # H^T acts on INPUT dimension (dim=0) of W: H^T : d_model → d_model
    w_rot = apply_hadamard_rotation(w_pad, dim=0)  # rotate input dim (rows)

    w_raw_r = block_quantize(
        w_pad, block_size=w_block_size, codebook=w_codebook,
        scale_format=w_scale_format, global_scale_format=w_global_scale,
    )
    w_rot_r = block_quantize(
        w_rot, block_size=w_block_size, codebook=w_codebook,
        scale_format=w_scale_format, global_scale_format=w_global_scale,
    )

    # ── QSNR metrics ──
    result = JointWAQSNR()

    # Weight QSNR
    result.w_qsnr_raw = w_raw_r.qsnr_db
    result.w_qsnr_rot = w_rot_r.qsnr_db
    result.w_gain = _compute_gain(result.w_qsnr_raw, result.w_qsnr_rot)

    # Activation QSNR
    if a_raw is not None:
        result.a_qsnr_raw = a_raw.qsnr_db
        result.a_qsnr_rot = a_rot.qsnr_db
        result.a_gain = _compute_gain(result.a_qsnr_raw, result.a_qsnr_rot)
    else:
        result.a_qsnr_raw = float('inf')
        result.a_qsnr_rot = float('inf')
        result.a_gain = 1.0

    # Per-token detail
    if a_raw is not None:
        pt_raw = a_raw.per_token_qsnr_db
        pt_rot = a_rot.per_token_qsnr_db
        result.a_qsnr_per_token_raw = pt_raw.tolist()
        result.a_qsnr_per_token_rot = pt_rot.tolist()

    # ── Joint output QSNR: ||X·W - X_q·W_q|| / ||X·W|| ──
    # Without rotation
    y_ref = x_pad @ w_pad
    y_raw = (a_raw.x_quant if a_raw else x_pad) @ w_raw_r.w_quant
    result.joint_qsnr_raw = _compute_joint_qsnr_db(y_ref, y_raw)

    # With rotation: (X·H)_q · (H^T·W)_q — but we stored X_q and W_q pre-inverse
    # Actually: X_rot_q = quantize(X·H), W_rot_q = quantize(W) (same as w_raw_r)
    # Then Y_rot_q = X_rot_q · W_rot_q  (no inverse rotation needed at output level)
    y_rot = (a_rot.x_quant if a_rot else x_rot) @ w_rot_r.w_quant
    result.joint_qsnr_rot = _compute_joint_qsnr_db(y_ref, y_rot)

    result.joint_gain = _compute_gain(result.joint_qsnr_raw, result.joint_qsnr_rot)

    if n_pad > d_model:
        # Keep original dimensions for config
        pass

    return result


def _compute_gain(qsnr_raw: float, qsnr_rot: float) -> float:
    """Linear gain: >1 means rotation beneficial."""
    if qsnr_raw >= float('inf') or qsnr_raw <= -float('inf'):
        return 1.0
    return float(10 ** ((qsnr_rot - qsnr_raw) / 10))


def _compute_joint_qsnr_db(y_ref: torch.Tensor, y_quant: torch.Tensor) -> float:
    """QSNR of quantized matrix product vs reference."""
    signal = (y_ref ** 2).mean()
    noise = ((y_quant - y_ref) ** 2).mean()
    if noise == 0:
        return float('inf')
    return float((10 * torch.log10(signal / noise)).item())
