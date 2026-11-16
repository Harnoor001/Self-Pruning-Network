"""Gated layers and gate regularisers for the research framework.

The existing method (``model.PrunableLinear``) multiplies every connection by a
learnable gate ``sigmoid(S)`` with ``S`` initialised to 2.0. ``PrunableConv2d``
applies exactly the same parameterisation to convolution kernels so that the
existing method can be evaluated on CNNs without changing its formulation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from self_pruning_network.model import PrunableLinear

GATE_INIT = 2.0

__all__ = [
    "GATE_INIT", "PrunableLinear", "PrunableConv2d", "is_gated", "gated_modules",
    "gate_penalty", "effective_weight",
]


class PrunableConv2d(nn.Module):
    """Conv2d with one sigmoid gate per kernel weight: ``W_eff = W * sigmoid(S)``."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, bias: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.groups = 1
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.gate_scores = nn.Parameter(torch.full_like(self.weight, GATE_INIT))
        self._dense_mode = False
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores)

    def set_mode(self, mode: str) -> None:
        if mode not in {"soft", "dense"}:
            raise ValueError("PrunableConv2d supports 'soft' and 'dense' modes")
        self._dense_mode = mode == "dense"

    def pruned_weight(self) -> torch.Tensor:
        return self.weight if self._dense_mode else self.weight * self.gates()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.conv2d(inputs, self.pruned_weight(), self.bias, self.stride, self.padding)


def is_gated(module: nn.Module) -> bool:
    return isinstance(module, (PrunableLinear, PrunableConv2d))


def gated_modules(model: nn.Module) -> list[nn.Module]:
    return [m for m in model.modules() if is_gated(m)]


def effective_weight(module: nn.Module) -> torch.Tensor:
    """Weight actually used in the forward pass (gates/masks folded in)."""
    if is_gated(module):
        return module.pruned_weight()
    return module.weight


def gate_penalty(model: nn.Module, normalization: str = "sum") -> torch.Tensor:
    """L1 penalty on gate values.

    ``sum`` reproduces the legacy ``SelfPruningMLP.sparsity_loss`` (sum over all
    gates, unnormalised). ``mean`` divides by the number of gates and is an
    experimental variant (ablation A), not the existing method.
    """
    modules = gated_modules(model)
    if not modules:
        return next(model.parameters()).new_zeros(())
    total = torch.stack([m.gates().sum() for m in modules]).sum()
    if normalization == "sum":
        return total
    if normalization == "mean":
        return total / sum(m.gate_scores.numel() for m in modules)
    raise ValueError(f"unknown gate penalty normalization: {normalization}")
