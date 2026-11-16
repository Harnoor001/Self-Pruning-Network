"""Method 6 (existing) and gate-aggregation variants.

* ``gate_in``  — the existing method: ``mean_i sigmoid(S_l[j, i])`` over the incoming
  row of unit j (for convolutions: over ``in_channels x k x k``). Identical to
  ``StructuredPrunedMLP.from_self_pruning``'s ``layer.gates().mean(dim=1)``.
* ``gate_out`` — variant V1: mean outgoing gate ``mean_k sigmoid(S_{l+1}[k, j])``.
* ``gate_combined`` — variant V2: mean-normalised sum of the two.
"""
import torch

from self_pruning_network.importance import (
    ScoringContext, incoming_rows, mean_normalized_sum, outgoing_columns, register,
)


def _gate_in(ctx: ScoringContext) -> list[torch.Tensor]:
    return [incoming_rows(ctx.module(g.producer).gates().detach()).mean(1).double() for g in ctx.groups]


def _gate_out(ctx: ScoringContext) -> list[torch.Tensor]:
    return [outgoing_columns(ctx.module(g.consumer).gates().detach(), g).mean(1).double() for g in ctx.groups]


@register("gate_in", needs_gates=True, description="existing method: mean incoming gate")
def gate_in_scores(ctx: ScoringContext) -> list[torch.Tensor]:
    return _gate_in(ctx)


@register("gate_out", needs_gates=True, description="variant: mean outgoing gate")
def gate_out_scores(ctx: ScoringContext) -> list[torch.Tensor]:
    return _gate_out(ctx)


@register("gate_combined", needs_gates=True, description="variant: normalised incoming + outgoing gate")
def gate_combined_scores(ctx: ScoringContext) -> list[torch.Tensor]:
    return [mean_normalized_sum(a, b) for a, b in zip(_gate_in(ctx), _gate_out(ctx))]
