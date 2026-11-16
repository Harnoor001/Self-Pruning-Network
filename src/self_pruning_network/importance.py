"""Common importance-scoring interface.

Every criterion maps a source model to one score vector per unit group (higher =
more important). Criteria are registered by name; see ``SCORERS`` and the
mathematical definitions in ``research/experimental_protocol.md`` §5.

All weight-based criteria use *effective* weights (gates folded), so they score
the function the source network actually computes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

from self_pruning_network.gates import effective_weight, is_gated
from self_pruning_network.structured_pruning import UnitGroup, find_unit_groups


@dataclass
class ScoringContext:
    model: nn.Module
    device: torch.device
    seed: int = 0
    calibration_loader: object | None = None
    groups: list[UnitGroup] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.groups:
            self.groups = find_unit_groups(self.model)

    def module(self, index: int) -> nn.Module:
        return self.model.network[index]


@dataclass(frozen=True)
class Scorer:
    name: str
    fn: Callable[[ScoringContext], list[torch.Tensor]]
    needs_data: bool = False
    needs_gates: bool = False
    description: str = ""


SCORERS: dict[str, Scorer] = {}


def register(name: str, *, needs_data: bool = False, needs_gates: bool = False, description: str = ""):
    def decorator(fn: Callable[[ScoringContext], list[torch.Tensor]]):
        if name in SCORERS:
            raise ValueError(f"scorer {name!r} registered twice")
        SCORERS[name] = Scorer(name, fn, needs_data, needs_gates, description)
        return fn
    return decorator


def compute_scores(name: str, context: ScoringContext) -> list[torch.Tensor]:
    if name not in SCORERS:
        raise KeyError(f"unknown importance criterion {name!r}; available: {sorted(SCORERS)}")
    scorer = SCORERS[name]
    if scorer.needs_data and context.calibration_loader is None:
        raise ValueError(f"criterion {name!r} requires a calibration loader")
    if scorer.needs_gates and not any(is_gated(m) for m in context.model.modules()):
        raise ValueError(f"criterion {name!r} requires a gated source model")
    was_training = context.model.training
    context.model.eval()
    try:
        scores = scorer.fn(context)
    finally:
        context.model.train(was_training)
    for group, values in zip(context.groups, scores):
        if values.shape != (group.n_units,):
            raise RuntimeError(f"{name}: score shape {tuple(values.shape)} != ({group.n_units},)")
        if not torch.isfinite(values).all():
            raise RuntimeError(f"{name}: non-finite scores in {group.name}")
    return [s.detach().double().cpu() for s in scores]


# --------------------------------------------------------------------------- tensor helpers
def incoming_rows(tensor: torch.Tensor) -> torch.Tensor:
    """(n_units, fan_in) view of a producer tensor (Linear [out,in] or Conv [out,in,k,k])."""
    return tensor.reshape(tensor.shape[0], -1)


def outgoing_columns(tensor: torch.Tensor, group: UnitGroup) -> torch.Tensor:
    """(n_units, fan_out) view of a consumer tensor restricted to each unit's inputs."""
    if tensor.dim() == 2:  # Linear [out, n*f]
        out = tensor.shape[0]
        return tensor.reshape(out, group.n_units, group.spatial_factor).permute(1, 0, 2).reshape(group.n_units, -1)
    return tensor.transpose(0, 1).reshape(group.n_units, -1)  # Conv [out, n, k, k]


def producer_weight(ctx: ScoringContext, group: UnitGroup) -> torch.Tensor:
    return incoming_rows(effective_weight(ctx.module(group.producer)).detach())


def consumer_weight(ctx: ScoringContext, group: UnitGroup) -> torch.Tensor:
    return outgoing_columns(effective_weight(ctx.module(group.consumer)).detach(), group)


def mean_normalized_sum(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """``a/mean(a) + b/mean(b)``: equal-weight, scale-free combination within a layer."""
    eps = torch.finfo(torch.float64).tiny
    first, second = first.double(), second.double()
    return first / first.mean().clamp_min(eps) + second / second.mean().clamp_min(eps)


@torch.no_grad()
def collect_activation_statistic(ctx: ScoringContext) -> list[torch.Tensor]:
    totals = [torch.zeros(g.n_units, dtype=torch.float64, device=ctx.device) for g in ctx.groups]
    count = 0
    handles = []
    for g, total in zip(ctx.groups, totals):
        def hook(_m, _i, output, total=total):
            per_unit = output.abs().mean(dim=list(range(2, output.dim()))) if output.dim() > 2 else output.abs()
            total.add_(per_unit.double().sum(dim=0))
        handles.append(ctx.module(g.activation).register_forward_hook(hook))
    try:
        for inputs, _targets in ctx.calibration_loader:
            ctx.model(inputs.to(ctx.device))
            count += inputs.shape[0]
    finally:
        for h in handles:
            h.remove()
    return [t / max(1, count) for t in totals]


def collect_taylor_statistic(ctx: ScoringContext) -> list[torch.Tensor]:
    """``E_x |sum_spatial a_j(x) * dL(x)/da_j(x)|`` with per-sample CE (sum reduction)."""
    totals = [torch.zeros(g.n_units, dtype=torch.float64, device=ctx.device) for g in ctx.groups]
    stored: dict[int, torch.Tensor] = {}
    handles = []
    for g in ctx.groups:
        def hook(_m, _i, output, key=g.index):
            output.retain_grad()
            stored[key] = output
        handles.append(ctx.module(g.activation).register_forward_hook(hook))
    count = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    try:
        for inputs, targets in ctx.calibration_loader:
            ctx.model.zero_grad(set_to_none=True)
            stored.clear()
            loss = loss_fn(ctx.model(inputs.to(ctx.device)), targets.to(ctx.device))
            loss.backward()
            for g, total in zip(ctx.groups, totals):
                a = stored[g.index]
                product = (a * a.grad).detach()
                if product.dim() > 2:
                    product = product.sum(dim=list(range(2, product.dim())))
                total.add_(product.abs().double().sum(dim=0))
            count += inputs.shape[0]
    finally:
        for h in handles:
            h.remove()
        ctx.model.zero_grad(set_to_none=True)
    return [t / max(1, count) for t in totals]


# Importing the baseline package registers every criterion.
from self_pruning_network import baselines  # noqa: E402,F401
