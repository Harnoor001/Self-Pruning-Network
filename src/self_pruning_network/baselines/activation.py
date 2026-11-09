"""Method 4 — mean absolute post-activation output on a fixed training calibration set."""
import torch

from self_pruning_network.importance import ScoringContext, collect_activation_statistic, register


@register("activation", needs_data=True, description="E_x mean_spatial |a_j(x)| (eval mode)")
def activation_scores(ctx: ScoringContext) -> list[torch.Tensor]:
    return collect_activation_statistic(ctx)
