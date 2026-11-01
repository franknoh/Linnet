"""Linnet models on Apple's MLX: every entry of a block as generated
`mlx.core` source over one copy of the weights (`load_model`)."""

from .model import MlxModel, load_model

__all__ = ["MlxModel", "load_model"]
