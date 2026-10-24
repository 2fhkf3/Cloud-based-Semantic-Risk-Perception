"""Reject invalid model parameters before inference."""
import torch


def require_finite(tensor, name):
    if not isinstance(tensor, torch.Tensor) or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f'{name} contains nonfinite or invalid tensor values')
