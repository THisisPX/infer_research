"""Synthetic activation generators that model LLM outlier channel phenomena.

Real LLM activations have persistent outlier channels: specific feature dimensions
where values are 10-100x larger than normal, across ALL tokens. This is the
phenomenon that rotation-based quantization targets.

Rotation effect on activations:
  X  →  X·H  (rotate input)
  Before: outlier channel k makes that column huge → per-token quantization scale
          is dominated by the outlier → normal channels get crushed
  After:  each element of X·H is a linear combination of all channels → outlier
          energy is spread → per-token distributions are more uniform
"""

from typing import Optional

import torch


def activation_with_outliers(
    seq_len: int,
    d_model: int,
    outlier_fraction: float = 0.01,
    outlier_scale: float = 50.0,
    normal_scale: float = 1.0,
    seed: int = 42,
) -> torch.Tensor:
    """Generate activations with persistent outlier channels.

    Models the LLM activation pattern: ~1% of channels are systematic outliers
    with 50-100x normal magnitude, and these channels are the same for ALL tokens.

    Args:
        seq_len: number of tokens
        d_model: hidden dimension
        outlier_fraction: fraction of channels that are outliers
        outlier_scale: RMS of outlier channels (relative to normal=1.0)
        normal_scale: RMS of normal channels

    Returns:
        X: (seq_len, d_model) — activation matrix
    """
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(seq_len, d_model, generator=gen) * normal_scale

    n_outlier = max(1, int(d_model * outlier_fraction))
    outlier_channels = torch.randperm(d_model, generator=gen)[:n_outlier]
    X[:, outlier_channels] *= outlier_scale

    return X


def activation_with_mixture_outliers(
    seq_len: int,
    d_model: int,
    outlier_fraction: float = 0.005,
    massive_scale: float = 200.0,     # massive outliers (attention "no-op" sink)
    normal_outlier_scale: float = 10.0,  # regular outliers
    seed: int = 42,
) -> torch.Tensor:
    """Two-class outlier model: massive spike outliers + regular channel outliers.

    Real LLM activations have TWO types of outliers (from DuQuant, DuQuant++):
    1. Channel-wise (Normal): large magnitude in specific channels, all tokens
    2. Spike (Massive): ~1400x median, in few tokens, FFN down_proj layers

    Args:
        seq_len: number of tokens
        d_model: hidden dimension
        outlier_fraction: fraction of normal outlier channels
        massive_scale: RMS of massive outliers
        normal_outlier_scale: RMS of normal outliers

    Returns:
        X: (seq_len, d_model)
    """
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(seq_len, d_model, generator=gen)

    # Channel-wise outliers (persistent across all tokens)
    n_outlier = max(1, int(d_model * outlier_fraction))
    outlier_channels = torch.randperm(d_model, generator=gen)[:n_outlier]
    X[:, outlier_channels] *= normal_outlier_scale

    # Massive spike outliers (few tokens, down_proj specific)
    # These appear in a small fraction of tokens, in specific channels
    n_massive_tokens = max(1, seq_len // 200)
    massive_tokens = torch.randperm(seq_len, generator=gen)[:n_massive_tokens]
    X[massive_tokens] *= massive_scale

    return X


def activation_llm_like(
    seq_len: int,
    d_model: int,
    layer_type: str = "ffn_down",
    seed: int = 42,
) -> torch.Tensor:
    """Generate activations matching typical LLM layer statistics.

    Based on observations from FlatQuant, DuQuant, and OSP:
    - Attention output: moderate outliers (kurtosis ~10-100)
    - FFN up/gate input: moderate outliers
    - FFN down input: extreme outliers (kurtosis ~1921 for SwiGLU w2 input)

    Args:
        seq_len: number of tokens
        d_model: hidden dimension
        layer_type: one of {'attention_out', 'ffn_up', 'ffn_down', 'ffn_gate'}
    """
    if layer_type == "ffn_down":
        # SwiGLU bilinear product: (silu(x·W1) * x·W3) → extreme skew
        return activation_with_mixture_outliers(
            seq_len, d_model,
            outlier_fraction=0.02, massive_scale=500.0, normal_outlier_scale=20.0,
            seed=seed,
        )
    elif layer_type == "ffn_up":
        return activation_with_outliers(
            seq_len, d_model, outlier_fraction=0.01, outlier_scale=15.0, seed=seed,
        )
    elif layer_type == "ffn_gate":
        return activation_with_outliers(
            seq_len, d_model, outlier_fraction=0.01, outlier_scale=12.0, seed=seed,
        )
    else:  # attention output
        return activation_with_outliers(
            seq_len, d_model, outlier_fraction=0.008, outlier_scale=8.0, seed=seed,
        )


ACTIVATION_GENERATORS = {
    "outlier": activation_with_outliers,
    "mixture_outlier": activation_with_mixture_outliers,
    "llm_like": activation_llm_like,
}
