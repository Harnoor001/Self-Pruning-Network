"""Method 7 — Network Slimming (Liu et al., 2017).

Scoring is ``|gamma_j|`` of the BatchNorm following each producer. The method's
distinguishing part is the *training objective* ``CE + lambda_s * sum |gamma|``
(``training.TrainConfig.slimming_lambda``), so in experiments the ``slimming``
criterion is applied to a separately trained slimming source. ``bn_gamma`` is the
same score applied to any other source (diagnostic).
"""
import torch
import torch.nn as nn

from self_pruning_network.importance import ScoringContext, register


def bn_gamma(ctx: ScoringContext) -> list[torch.Tensor]:
    scores = []
    for g in ctx.groups:
        if g.bn is None:
            raise ValueError(f"{g.name} has no BatchNorm; BN-gamma criteria are undefined")
        scores.append(ctx.module(g.bn).weight.detach().abs().double())
    return scores


@register("slimming", description="|gamma| of BN (apply to a slimming-trained source)")
def slimming_scores(ctx: ScoringContext) -> list[torch.Tensor]:
    return bn_gamma(ctx)


@register("bn_gamma", description="|gamma| of BN on any source (diagnostic)")
def bn_gamma_scores(ctx: ScoringContext) -> list[torch.Tensor]:
    return bn_gamma(ctx)


def slimming_penalty(model: nn.Module) -> torch.Tensor:
    terms = [m.weight.abs().sum() for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    return torch.stack(terms).sum() if terms else next(model.parameters()).new_zeros(())
