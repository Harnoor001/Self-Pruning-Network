"""Generic structured (neuron / channel) pruning on flat ``network`` Sequentials.

A *unit group* is one prunable hidden layer::

    producer (Linear/Conv2d, rows = units) -> [BatchNorm] -> activation -> ... -> consumer

Removing unit ``j`` deletes producer row ``j`` (+ bias), BN entry ``j`` and the
consumer input column(s) of ``j``. In ``eval`` mode this is exactly equivalent to
forcing the activation output of unit ``j`` to zero (tested), which is how the
removal-damage experiment measures true importance without rebuilding the model.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from self_pruning_network.architectures import build_model, model_config, to_plain, with_widths
from self_pruning_network.gates import PrunableConv2d, PrunableLinear

PRODUCERS = (nn.Linear, nn.Conv2d, PrunableLinear, PrunableConv2d)
ACTIVATIONS = (nn.ReLU, nn.GELU)
NORMS = (nn.BatchNorm1d, nn.BatchNorm2d)


@dataclass(frozen=True)
class UnitGroup:
    index: int
    producer: int          # index in model.network
    bn: int | None
    activation: int
    consumer: int
    n_units: int
    spatial_factor: int    # consumer input features per unit (>1 for Linear after Flatten)

    @property
    def name(self) -> str:
        return f"group{self.index}"


def _out_units(module: nn.Module) -> int:
    return int(module.out_features if isinstance(module, (nn.Linear, PrunableLinear)) else module.out_channels)


def _in_units(module: nn.Module) -> int:
    return int(module.in_features if isinstance(module, (nn.Linear, PrunableLinear)) else module.in_channels)


def find_unit_groups(model: nn.Module) -> list[UnitGroup]:
    modules = list(model.network)
    producers = [i for i, m in enumerate(modules) if isinstance(m, PRODUCERS)]
    groups: list[UnitGroup] = []
    for g, (start, end) in enumerate(zip(producers[:-1], producers[1:])):
        between = modules[start + 1:end]
        bn = next((start + 1 + k for k, m in enumerate(between) if isinstance(m, NORMS)), None)
        act = next((start + 1 + k for k, m in enumerate(between) if isinstance(m, ACTIVATIONS)), None)
        if act is None:
            raise ValueError(f"no activation between producers {start} and {end}")
        n = _out_units(modules[start])
        consumer_in = _in_units(modules[end])
        if consumer_in % n:
            raise ValueError(f"consumer input {consumer_in} is not a multiple of {n} units")
        groups.append(UnitGroup(g, start, bn, act, end, n, consumer_in // n))
    return groups


# --------------------------------------------------------------------------- selection
def keep_count(n_units: int, ratio: float, min_keep: int = 1) -> int:
    """Units kept for a removal ratio: ``max(min_keep, n - floor(n * ratio))``.

    Matches the legacy rule (floor of removed units, never fewer than one kept).
    """
    if not 0.0 <= ratio < 1.0:
        raise ValueError("ratio must be in [0, 1)")
    return min(n_units, max(int(min_keep), n_units - int(math.floor(n_units * ratio + 1e-9))))


def select_top(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the ``k`` highest scores (stable: ties keep the lower index), ascending."""
    order = torch.argsort(scores.detach().double().cpu(), descending=True, stable=True)
    return order[:k].sort().values


def uniform_keep(scores: list[torch.Tensor], ratio: float, min_keep: int = 1) -> list[torch.Tensor]:
    return [select_top(s, keep_count(s.numel(), ratio, min_keep)) for s in scores]


def global_keep(scores: list[torch.Tensor], ratio: float, min_keep: int = 1) -> list[torch.Tensor]:
    """Single cross-layer ranking (Network Slimming style) with the uniform total budget.

    The number of removed units equals the total removed by ``uniform_keep`` so that
    both allocations have the same unit budget; ``min_keep`` is enforced per layer.
    """
    budget = sum(s.numel() - keep_count(s.numel(), ratio, min_keep) for s in scores)
    entries = sorted(((float(v), g, j) for g, s in enumerate(scores) for j, v in enumerate(s.double().cpu().tolist())),
                     key=lambda e: (e[0], -e[1], -e[2]))  # ascending score; deterministic ties
    remaining = [s.numel() for s in scores]
    removed: list[set[int]] = [set() for _ in scores]
    for value, g, j in entries:
        if budget == 0:
            break
        if remaining[g] <= min_keep:
            continue
        removed[g].add(j)
        remaining[g] -= 1
        budget -= 1
    return [torch.tensor(sorted(set(range(s.numel())) - removed[g]), dtype=torch.long) for g, s in enumerate(scores)]


# --------------------------------------------------------------------------- reconstruction
def _expand(keep: torch.Tensor, factor: int) -> torch.Tensor:
    if factor == 1:
        return keep
    return (keep[:, None] * factor + torch.arange(factor)[None, :]).reshape(-1)


@torch.no_grad()
def compact_model(model: nn.Module, keep: list[torch.Tensor]) -> nn.Module:
    """Physically remove units. Gates are folded first; the result is gate-free."""
    plain = to_plain(model)
    groups = find_unit_groups(plain)
    if len(keep) != len(groups):
        raise ValueError(f"expected {len(groups)} keep lists, got {len(keep)}")
    keep = [k.to(torch.long).cpu() for k in keep]
    for g, k in zip(groups, keep):
        if k.numel() == 0:
            raise ValueError(f"{g.name}: at least one unit must be kept")
        if k.unique().numel() != k.numel() or int(k.max()) >= g.n_units or int(k.min()) < 0:
            raise ValueError(f"{g.name}: invalid keep indices")
    compact = build_model(with_widths(model_config(plain), [int(k.numel()) for k in keep]))
    device = next(plain.parameters()).device
    compact.to(device)
    out_keep = {g.producer: k for g, k in zip(groups, keep)}
    in_keep = {g.consumer: _expand(k, g.spatial_factor) for g, k in zip(groups, keep)}
    bn_keep = {g.bn: k for g, k in zip(groups, keep) if g.bn is not None}
    for index, (src, dst) in enumerate(zip(plain.network, compact.network)):
        if isinstance(src, (nn.Linear, nn.Conv2d)):
            weight = src.weight
            if index in out_keep:
                weight = weight[out_keep[index].to(device)]
            if index in in_keep:
                weight = weight[:, in_keep[index].to(device)]
            dst.weight.copy_(weight)
            if src.bias is not None:
                bias = src.bias[out_keep[index].to(device)] if index in out_keep else src.bias
                dst.bias.copy_(bias)
        elif isinstance(src, NORMS):
            idx = bn_keep[index].to(device)
            dst.weight.copy_(src.weight[idx])
            dst.bias.copy_(src.bias[idx])
            dst.running_mean.copy_(src.running_mean[idx])
            dst.running_var.copy_(src.running_var[idx])
            dst.num_batches_tracked.copy_(src.num_batches_tracked)
    compact.train(model.training)
    return compact


@contextlib.contextmanager
def zeroed_units(model: nn.Module, group: UnitGroup, units: torch.Tensor | list[int]):
    """Temporarily force the activation output of ``units`` in ``group`` to zero."""
    module = model.network[group.activation]
    units = torch.as_tensor(units, dtype=torch.long)

    def hook(_module, _inputs, output):
        mask = torch.ones(output.shape[1], device=output.device, dtype=output.dtype)
        mask[units.to(output.device)] = 0
        shape = [1, -1] + [1] * (output.dim() - 2)
        return output * mask.view(*shape)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def pruning_summary(source: nn.Module, keep: list[torch.Tensor]) -> dict[str, object]:
    groups = find_unit_groups(source)
    total = sum(g.n_units for g in groups)
    kept = sum(int(k.numel()) for k in keep)
    return {
        "source_widths": [g.n_units for g in groups],
        "compact_widths": [int(k.numel()) for k in keep],
        "units_total": total,
        "units_kept": kept,
        "units_removed_percent": (total - kept) / total * 100.0 if total else 0.0,
        "layer_removed_percent": [(g.n_units - int(k.numel())) / g.n_units * 100.0 for g, k in zip(groups, keep)],
    }
