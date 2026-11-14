"""Method 5 — first-order Taylor criterion (Molchanov et al., 2017).

``I_j = E_x | sum_spatial a_j(x) * dL(x)/da_j(x) |`` — the magnitude of the
first-order estimate of the loss change when unit ``j``'s output is zeroed for
input ``x``. Computed in eval mode on the calibration set with per-sample loss.
"""
import torch

from self_pruning_network.importance import ScoringContext, collect_taylor_statistic, register


@register("taylor", needs_data=True, description="E_x |sum_spatial a * dL/da| (eval mode)")
def taylor_scores(ctx: ScoringContext) -> list[torch.Tensor]:
    return collect_taylor_statistic(ctx)
