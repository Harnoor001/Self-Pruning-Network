"""Method 3 — L2 magnitude of incoming and outgoing weights (same aggregation as L1)."""
import torch

from self_pruning_network.importance import (
    ScoringContext, consumer_weight, mean_normalized_sum, producer_weight, register,
)


@register("l2", description="mean-normalised L2(incoming) + L2(outgoing) of effective weights")
def l2_scores(ctx: ScoringContext) -> list[torch.Tensor]:
    return [mean_normalized_sum(producer_weight(ctx, g).norm(dim=1), consumer_weight(ctx, g).norm(dim=1))
            for g in ctx.groups]
