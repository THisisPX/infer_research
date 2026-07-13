"""Weight generators: synthetic and real LLM weight loaders."""

from .synthetic import (
    gaussian_weights, laplacian_weights, channel_outlier_weights,
    gamma_mixture_weights, llm_like_weights, SYNTHETIC_GENERATORS,
)

__all__ = [
    "gaussian_weights", "laplacian_weights", "channel_outlier_weights",
    "gamma_mixture_weights", "llm_like_weights", "SYNTHETIC_GENERATORS",
]
