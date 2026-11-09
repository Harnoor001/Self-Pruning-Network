"""Method 2 — L1 magnitude of incoming and outgoing weights.

``I_j = ||W_l[j]||_1 / mean_k ||W_l[k]||_1 + ||W_{l+1}[:, j]||_1 / mean_k ||W_{l+1}[:, k]||_1``
"""
import torch

from self_pruning_network.importance import (
    ScoringContext, consumer_weight, mean_normalized_sum, producer_weight, register,
)


@register("l1", description="mean-normalised L1(incoming) + L1(outgoing) of effective weights")
def l1_scores(ctx: ScoringContext) -> list[torch.Tensor]:
    return [mean_normalized_sum(producer_weight(ctx, g).abs().sum(1), consumer_weight(ctx, g).abs().sum(1))
            for g in ctx.groups]


@register("l1_in", description="L1 of incoming effective weights only (diagnostic)")
def l1_in_scores(ctx: ScoringContext) -> list[torch.Tensor]:
    return [producer_weight(ctx, g).abs().sum(1).double() for g in ctx.groups]
