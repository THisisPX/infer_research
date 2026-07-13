"""Synthetic weight matrix generators for controlled experiments."""

from typing import Optional

import torch


def gaussian_weights(d_in: int, d_out: int, std: float = 1.0,
                     seed: int = 42) -> torch.Tensor:
    """IID Gaussian weights W_{i,j} ~ N(0, std²)."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(d_in, d_out, generator=gen) * std


def laplacian_weights(d_in: int, d_out: int, scale: float = 1.0,
                      seed: int = 42) -> torch.Tensor:
    """IID Laplacian weights W_{i,j} ~ Laplace(0, scale)."""
    gen = torch.Generator().manual_seed(seed)
    u = torch.rand(d_in, d_out, generator=gen) - 0.5
    return -scale * torch.sign(u) * torch.log(1 - 2 * u.abs())


def channel_outlier_weights(
    d_in: int, d_out: int,
    outlier_fraction: float = 0.05,
    outlier_scale: float = 10.0,
    base_std: float = 1.0,
    seed: int = 42,
) -> torch.Tensor:
    """Weights with per-channel variance: 95% channels σ≈1, 5% channels σ≈10.

    This models the outlier channel phenomenon in LLM activations/weights.
    """
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(d_in, d_out, generator=gen) * base_std

    n_outlier = max(1, int(d_in * outlier_fraction))
    outlier_channels = torch.randperm(d_in, generator=gen)[:n_outlier]
    w[outlier_channels] *= outlier_scale

    return w


def gamma_mixture_weights(
    d_in: int, d_out: int,
    alphas: list = [0.5, 2.0, 5.0],
    weights: list = [0.7, 0.2, 0.1],
    seed: int = 42,
) -> torch.Tensor:
    """Per-channel gamma-distributed variances (more realistic than outlier model).

    Each channel i has σ_i ~ Gamma(α_k, 1/α_k) with mixture probability w_k.
    """
    gen = torch.Generator().manual_seed(seed)
    stds = torch.zeros(d_in)
    for alpha, weight in zip(alphas, weights):
        n = max(1, int(d_in * weight))
        idx = torch.randperm(d_in, generator=gen)[:n]
        stds[idx] = torch.distributions.Gamma(alpha, 1.0 / alpha).sample((n,))

    stds = stds.unsqueeze(1)  # (d_in, 1)
    return stds * torch.randn(d_in, d_out, generator=gen)


def llm_like_weights(
    d_in: int, d_out: int,
    layer_type: str = "ffn_down",
    seed: int = 42,
) -> torch.Tensor:
    """Weights with statistics matching typical LLM layer types.

    Args:
        layer_type: one of {'attention', 'ffn_up', 'ffn_down', 'ffn_gate'}
    """
    gen = torch.Generator().manual_seed(seed)

    # Different layer types have different outlier severities
    if layer_type == "ffn_down":
        # FC2/down_proj: most outlier-heavy (kurtosis ~1921)
        outlier_scale = 15.0
        outlier_fraction = 0.08
    elif layer_type == "ffn_up":
        outlier_scale = 5.0
        outlier_fraction = 0.03
    elif layer_type == "ffn_gate":
        outlier_scale = 8.0
        outlier_fraction = 0.05
    else:  # attention (Q, K, V, O)
        outlier_scale = 3.0
        outlier_fraction = 0.02

    return channel_outlier_weights(
        d_in, d_out,
        outlier_fraction=outlier_fraction,
        outlier_scale=outlier_scale,
        seed=seed,
    )


SYNTHETIC_GENERATORS = {
    "gaussian": gaussian_weights,
    "laplacian": laplacian_weights,
    "channel_outlier": channel_outlier_weights,
    "gamma_mixture": gamma_mixture_weights,
    "llm_like": llm_like_weights,
}
