"""Method 1 — random structured pruning."""
import torch

from self_pruning_network.importance import ScoringContext, register


@register("random", description="U(0,1) scores from a CPU generator seeded with the run seed")
def random_scores(ctx: ScoringContext) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(ctx.seed))
    return [torch.rand(g.n_units, generator=generator, dtype=torch.float64) for g in ctx.groups]
