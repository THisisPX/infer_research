"""Hadamard and Cayley rotation transforms for quantization.

Implements Fast Walsh-Hadamard Transform (FWHT) in O(n log n).
"""

import math
from typing import Optional

import torch
import torch.nn as nn


# ── Fast Walsh-Hadamard Transform ───────────────────────────────────

def _hadamard_matrix(n: int) -> torch.Tensor:
    """Construct an n×n Hadamard matrix (n must be power of 2)."""
    if n & (n - 1) != 0:
        raise ValueError(f"n={n} must be a power of 2 for Hadamard matrix")
    if n == 1:
        return torch.ones(1, 1)
    H = _hadamard_matrix(n // 2)
    return torch.cat(
        [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)],
        dim=0,
    )


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform along the last dimension.

    Normalized so that H * H^T = I (orthogonal).

    Args:
        x: tensor of shape (..., n) where n is a power of 2

    Returns:
        H @ x, same shape
    """
    n = x.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError(f"Last dim {n} must be power of 2. Got shape {x.shape}")

    h = 1
    result = x.clone()
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(h):
                u = result[..., i + j].clone()
                v = result[..., i + j + h].clone()
                result[..., i + j] = u + v
                result[..., i + j + h] = u - v
        h *= 2
    return result / math.sqrt(n)


def apply_hadamard_rotation(w: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Apply Hadamard rotation to weight matrix for quantization.

    W' = H · W  (rotate along output dimension).

    This is the standard QuaRot-style rotation. The inverse rotation
    H^T · W' = H^T · H · W = W is applied during inference to recover
    original outputs.

    Args:
        w: weight matrix (d_in, d_out)
        dim: dimension to rotate along (0=input, 1=output)

    Returns:
        Rotated weight matrix, same shape
    """
    d = w.shape[dim]
    n_pad = _next_power_of_2(d)

    if n_pad != d:
        if dim == 1:
            w_padded = torch.nn.functional.pad(w, (0, n_pad - d), value=0.0)
        else:
            w_padded = torch.nn.functional.pad(w, (0, 0, 0, n_pad - d), value=0.0)
    else:
        w_padded = w

    if dim == 1:
        # Rotate output dimension: H @ W^T → transpose back
        rotated = fwht(w_padded.T).T
    else:
        # Rotate input dimension
        rotated = fwht(w_padded.T).T  # same operation

    if n_pad != d:
        if dim == 1:
            rotated = rotated[:, :d]
        else:
            rotated = rotated[:d, :]

    return rotated


def _next_power_of_2(n: int) -> int:
    """Smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


# ── Block-wise Hadamard (MR-GPTQ style) ─────────────────────────────

def apply_block_hadamard(w: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Apply Hadamard rotation within blocks (MR-GPTQ style).

    Rotates within each 1×block_size block independently.

    Args:
        w: weight matrix (d_in, d_out), d_out must be divisible by block_size
        block_size: block size (must be power of 2)

    Returns:
        Block-rotated weight matrix
    """
    d_in, d_out = w.shape
    assert d_out % block_size == 0, f"d_out={d_out} not divisible by block_size={block_size}"

    n_blocks = d_out // block_size
    w_blocks = w.view(d_in, n_blocks, block_size)
    rotated = fwht(w_blocks)  # rotates last dim (block_size)
    return rotated.view(d_in, d_out)


# ── Random Orthogonal Rotation ──────────────────────────────────────

def random_orthogonal_rotation(w: torch.Tensor, dim: int = 1, seed: int = 42) -> torch.Tensor:
    """Apply a random orthogonal matrix (QR-based).

    For comparison with Hadamard — tests whether Hadamard's specific
    structure matters vs any orthogonal transform.

    Args:
        w: weight matrix (d_in, d_out)
        dim: dimension to rotate along
        seed: random seed for reproducibility

    Returns:
        Rotated weight matrix
    """
    d = w.shape[dim]
    generator = torch.Generator(device=w.device).manual_seed(seed)

    rand_mat = torch.randn(d, d, generator=generator, device=w.device, dtype=w.dtype)
    Q, _ = torch.linalg.qr(rand_mat)

    if dim == 1:
        return w @ Q.T
    else:
        return Q @ w


# ── Utilities ───────────────────────────────────────────────────────

def pad_to_block_multiple(w: torch.Tensor, block_size: int) -> torch.Tensor:
    """Pad output dimension to be divisible by block_size."""
    d_out = w.shape[1]
    if d_out % block_size == 0:
        return w
    pad = block_size - (d_out % block_size)
    return torch.nn.functional.pad(w, (0, pad), value=0.0)


def pad_to_power_of_2(w: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Pad dimension to next power of 2 for Hadamard transform."""
    d = w.shape[dim]
    n = _next_power_of_2(d)
    if n == d:
        return w
    pad = n - d
    if dim == 1:
        return torch.nn.functional.pad(w, (0, pad), value=0.0)
    else:
        return torch.nn.functional.pad(w, (0, 0, 0, pad), value=0.0)
