"""Tests for the comparative-pruning research framework."""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from self_pruning_network.architectures import MLP, VGG, build_model, model_config, to_plain, unit_widths
from self_pruning_network.evaluation import (
    count_macs, deployable_parameter_count, deployable_size_bytes, evaluate,
)
from self_pruning_network.gates import PrunableConv2d, gate_penalty
from self_pruning_network.importance import SCORERS, ScoringContext, compute_scores
from self_pruning_network.model import SelfPruningMLP, StructuredPrunedMLP
from self_pruning_network.structured_pruning import (
    compact_model, find_unit_groups, global_keep, keep_count, select_top, uniform_keep, zeroed_units,
)
from self_pruning_network.training import TrainConfig, fine_tune, set_determinism, train_model

TOL = dict(atol=1e-5, rtol=1e-4)


def randomize(model: nn.Module, seed: int = 0) -> nn.Module:
    """Give gates, BN stats and BN affine non-trivial values."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "gate_scores" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 1.5 + 1.0)
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                n = m.num_features
                m.weight.copy_(torch.rand(n, generator=g) + 0.5)
                m.bias.copy_(torch.randn(n, generator=g) * 0.1)
                m.running_mean.copy_(torch.randn(n, generator=g) * 0.1)
                m.running_var.copy_(torch.rand(n, generator=g) + 0.5)
    return model


def tiny_mlp(gated=True):
    torch.manual_seed(0)
    model = SelfPruningMLP(input_dim=3 * 32 * 32, hidden_dims=[12, 8], num_classes=4, dropout=0.3) if gated \
        else MLP(input_dim=3 * 32 * 32, hidden_dims=[12, 8], num_classes=4)
    return randomize(model).eval()


def tiny_vgg(gated=True, cfg="vgg_tiny"):
    torch.manual_seed(0)
    return randomize(VGG(cfg, num_classes=4, gated=gated)).eval()


def inputs(n=6, seed=1):
    return torch.randn(n, 3, 32, 32, generator=torch.Generator().manual_seed(seed))


def loader(n=32, classes=4, seed=2, batch=8):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.randn(n, 3, 32, 32, generator=g), torch.randint(0, classes, (n,), generator=g)),
                      batch_size=batch, shuffle=False)


MODELS = [lambda: tiny_mlp(True), lambda: tiny_mlp(False), lambda: tiny_vgg(True), lambda: tiny_vgg(False),
          lambda: tiny_vgg(True, cfg=[8, "M", 16, "M", 16, "M", 8, "M"])]  # last: 2x2 spatial before classifier


# --------------------------------------------------------------------------- reconstruction
@pytest.mark.parametrize("factory", MODELS)
def test_to_plain_is_function_preserving(factory):
    model = factory()
    x = inputs()
    assert torch.allclose(to_plain(model).eval()(x), model(x), **TOL)


@pytest.mark.parametrize("factory", MODELS)
def test_compact_at_zero_ratio_is_identity(factory):
    model = factory()
    keep = [torch.arange(g.n_units) for g in find_unit_groups(model)]
    compact = compact_model(model, keep).eval()
    x = inputs()
    assert torch.allclose(compact(x), model(x), **TOL)


@pytest.mark.parametrize("factory", MODELS)
def test_removal_equals_zeroing_activation(factory):
    model = factory()
    groups = find_unit_groups(model)
    keep = [select_top(torch.rand(g.n_units, generator=torch.Generator().manual_seed(g.index)), max(1, g.n_units // 2))
            for g in groups]
    compact = compact_model(model, keep).eval()
    x = inputs()
    with torch.no_grad(), contextlib_exit_stack() as stack:
        for g, k in zip(groups, keep):
            removed = sorted(set(range(g.n_units)) - set(k.tolist()))
            stack.enter_context(zeroed_units(model, g, removed))
        expected = model(x)
    assert torch.allclose(compact(x), expected, **TOL)
    assert [int(k.numel()) for k in keep] == unit_widths(compact)


def contextlib_exit_stack():
    import contextlib
    return contextlib.ExitStack()


def test_bn_parameters_transferred_exactly():
    model = tiny_vgg(gated=False)
    groups = find_unit_groups(model)
    keep = [torch.tensor([0, 2, 5]) for _ in groups]
    compact = compact_model(model, keep)
    for g in groups:
        src = model.network[g.bn]
        dst = compact.network[g.bn]
        for attr in ("weight", "bias", "running_mean", "running_var"):
            assert torch.equal(getattr(dst, attr), getattr(src, attr)[keep[g.index]])


def test_consumer_columns_sliced_with_spatial_factor():
    model = tiny_vgg(gated=False, cfg=[8, "M", 16, "M", 16, "M", 8, "M"])
    last = find_unit_groups(model)[-1]
    assert last.spatial_factor == 4  # 8 channels x 2 x 2 flattened
    keep = [torch.arange(g.n_units) for g in find_unit_groups(model)]
    keep[-1] = torch.tensor([1, 6])
    compact = compact_model(model, keep)
    expected_cols = torch.tensor([4, 5, 6, 7, 24, 25, 26, 27])
    assert torch.equal(compact.network[last.consumer].weight, model.network[last.consumer].weight[:, expected_cols])


def test_compact_rejects_invalid_keep_sets():
    model = tiny_mlp()
    groups = find_unit_groups(model)
    with pytest.raises(ValueError):
        compact_model(model, [torch.tensor([], dtype=torch.long), torch.arange(groups[1].n_units)])
    with pytest.raises(ValueError):
        compact_model(model, [torch.tensor([0, 0]), torch.arange(groups[1].n_units)])


def test_compact_checkpoint_roundtrip(tmp_path):
    model = tiny_vgg()
    compact = compact_model(model, uniform_keep([torch.rand(g.n_units) for g in find_unit_groups(model)], 0.4)).eval()
    path = tmp_path / "c.pt"
    torch.save({"model_config": model_config(compact), "state_dict": compact.state_dict()}, path)
    payload = torch.load(path, weights_only=False)
    rebuilt = build_model(payload["model_config"])
    rebuilt.load_state_dict(payload["state_dict"])
    rebuilt.eval()
    x = inputs()
    assert torch.equal(rebuilt(x), compact(x))


# --------------------------------------------------------------------------- selection
def test_keep_count_matches_legacy_rule_and_min_keep():
    assert keep_count(2048, 0.2) == 1639 and keep_count(1024, 0.4) == 615 and keep_count(512, 0.6) == 205
    assert keep_count(10, 0.6, min_keep=8) == 8
    assert keep_count(3, 0.9) == 1
    with pytest.raises(ValueError):
        keep_count(10, 1.0)


def test_select_top_is_deterministic_with_ties():
    scores = torch.tensor([1.0, 3.0, 3.0, 0.5, 3.0])
    assert select_top(scores, 2).tolist() == [1, 2]
    assert select_top(scores, 2).tolist() == select_top(scores.clone(), 2).tolist()


def test_uniform_ratio_gives_identical_architectures_for_all_criteria():
    model = tiny_mlp()
    ctx = ScoringContext(model, torch.device("cpu"), seed=3, calibration_loader=loader())
    widths = set()
    for name in ["random", "l1", "l2", "activation", "taylor", "gate_in", "gate_out", "gate_combined", "bn_gamma"]:
        widths.add(tuple(int(k.numel()) for k in uniform_keep(compute_scores(name, ctx), 0.4)))
    assert widths == {(8, 5)}


def test_global_allocation_matches_uniform_budget_and_min_keep():
    scores = [torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), torch.tensor([5.0, 6.0, 7.0, 8.0])]
    keep = global_keep(scores, 0.5, min_keep=2)
    uniform = uniform_keep(scores, 0.5, min_keep=2)
    assert sum(k.numel() for k in keep) == sum(k.numel() for k in uniform)
    assert keep[0].numel() == 2 and keep[0].tolist() == [4, 5]


# --------------------------------------------------------------------------- criteria
def test_gate_in_matches_legacy_structured_pruning():
    source = tiny_mlp()
    ctx = ScoringContext(source, torch.device("cpu"))
    scores = compute_scores("gate_in", ctx)
    for layer, s in zip(source.prunable_layers[:-1], scores):
        assert torch.allclose(s, layer.gates().mean(dim=1).double())
    legacy = StructuredPrunedMLP.from_self_pruning(source, 0.4, fold_gates=True).eval()
    keep = uniform_keep(scores, 0.4)
    assert [k.tolist() for k in keep] == legacy.keep_indices
    ours = compact_model(source, keep).eval()
    x = inputs()
    assert torch.allclose(ours(x), legacy(x), **TOL)


def test_legacy_reconstruction_without_folding_is_not_function_preserving():
    """Documents audit finding 1 (legacy default copies raw W)."""
    source = tiny_mlp()
    x = inputs()
    legacy_raw = StructuredPrunedMLP.from_self_pruning(source, 0.0).eval()
    legacy_fold = StructuredPrunedMLP.from_self_pruning(source, 0.0, fold_gates=True).eval()
    assert not torch.allclose(legacy_raw(x), source(x), **TOL)
    assert torch.allclose(legacy_fold(x), source(x), **TOL)


def test_gate_out_uses_consumer_columns():
    source = tiny_vgg(cfg=[8, "M", 16, "M", 16, "M", 8, "M"])
    ctx = ScoringContext(source, torch.device("cpu"))
    scores = compute_scores("gate_out", ctx)
    g0 = ctx.groups[0]
    expected = source.network[g0.consumer].gates().mean(dim=(0, 2, 3)).double()
    assert torch.allclose(scores[0], expected)
    last = ctx.groups[-1]
    lin = source.network[last.consumer].gates()  # [classes, 8*4]
    assert torch.allclose(scores[-1], lin.reshape(lin.shape[0], 8, 4).mean(dim=(0, 2)).double())


def test_l1_l2_scores_match_manual_formula():
    model = tiny_mlp(gated=False)
    ctx = ScoringContext(model, torch.device("cpu"))
    w_in, w_out = model.network[0].weight, model.network[4].weight
    a, b = w_in.abs().sum(1).double(), w_out.abs().sum(0).double()
    assert torch.allclose(compute_scores("l1", ctx)[0], a / a.mean() + b / b.mean())
    a, b = w_in.norm(dim=1).double(), w_out.norm(dim=0).double()
    assert torch.allclose(compute_scores("l2", ctx)[0], a / a.mean() + b / b.mean())


def test_weight_scores_use_effective_gated_weights():
    gated = tiny_mlp(gated=True)
    plain = to_plain(gated)
    for name in ["l1", "l2"]:
        a = compute_scores(name, ScoringContext(gated, torch.device("cpu")))
        b = compute_scores(name, ScoringContext(plain, torch.device("cpu")))
        assert all(torch.allclose(x, y) for x, y in zip(a, b))


def test_activation_and_taylor_match_manual_computation():
    model = tiny_mlp(gated=False)
    data = loader(n=16, batch=5)
    ctx = ScoringContext(model, torch.device("cpu"), calibration_loader=data)
    act, tay = compute_scores("activation", ctx), compute_scores("taylor", ctx)
    x = torch.cat([b[0] for b in data])
    y = torch.cat([b[1] for b in data])
    model.eval()
    h = model.network[:3](x.view(x.shape[0], -1)).detach().requires_grad_(True)
    loss = nn.functional.cross_entropy(model.network[3:](h), y, reduction="sum")
    (grad,) = torch.autograd.grad(loss, h)
    assert torch.allclose(act[0], h.abs().mean(0).double(), atol=1e-6)
    assert torch.allclose(tay[0], (h * grad).abs().mean(0).double(), atol=1e-6)


def test_random_scores_reproducible_and_seed_dependent():
    model = tiny_mlp()
    a = compute_scores("random", ScoringContext(model, torch.device("cpu"), seed=5))
    b = compute_scores("random", ScoringContext(model, torch.device("cpu"), seed=5))
    c = compute_scores("random", ScoringContext(model, torch.device("cpu"), seed=6))
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not torch.equal(a[0], c[0])


def test_criteria_requirements_enforced():
    with pytest.raises(ValueError):
        compute_scores("gate_in", ScoringContext(tiny_mlp(gated=False), torch.device("cpu")))
    with pytest.raises(ValueError):
        compute_scores("taylor", ScoringContext(tiny_mlp(), torch.device("cpu")))
    with pytest.raises(KeyError):
        compute_scores("nope", ScoringContext(tiny_mlp(), torch.device("cpu")))
    assert {"random", "l1", "l2", "activation", "taylor", "gate_in", "slimming"} <= set(SCORERS)


def test_incoming_gate_level_is_batchnorm_invariant_in_training_mode():
    """Novelty audit O2: scaling one neuron's incoming gates does not change BN output."""
    model = tiny_mlp().train()
    model.dropout = 0.0
    for m in model.network:
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    x = inputs(16)
    before = model(x)
    with torch.no_grad():
        layer = model.prunable_layers[0]
        target = torch.sigmoid(layer.gate_scores[3]) * 0.3  # shrink neuron 3's incoming gates by 70 %
        layer.gate_scores[3] = torch.logit(target)
    after = model(x)
    assert torch.allclose(before, after, atol=1e-3)


# --------------------------------------------------------------------------- metrics
def test_macs_and_parameters_for_known_mlp():
    model = MLP(input_dim=3072, hidden_dims=[20, 10], num_classes=4)
    assert count_macs(model) == 3072 * 20 + 20 * 10 + 10 * 4
    expected = (3072 * 20 + 20) + (20 * 10 + 10) + (10 * 4 + 4) + 2 * 20 + 2 * 10
    assert deployable_parameter_count(model) == expected
    gated = SelfPruningMLP(input_dim=3072, hidden_dims=[20, 10], num_classes=4)
    assert deployable_parameter_count(gated) == expected
    assert deployable_size_bytes(to_plain(gated)) <= deployable_size_bytes(gated) + 4096


def test_conv_macs():
    model = VGG([8, "M"], num_classes=2)
    assert count_macs(model) == 32 * 32 * 8 * 3 * 9 + 8 * 16 * 16 * 2


def test_gate_penalty_sum_matches_legacy():
    model = tiny_mlp()
    assert torch.allclose(gate_penalty(model, "sum"), model.sparsity_loss())
    n = sum(l.gate_scores.numel() for l in model.prunable_layers)
    assert torch.allclose(gate_penalty(model, "mean"), model.sparsity_loss() / n)
    conv = PrunableConv2d(2, 3, 3)
    assert torch.allclose(conv.gates(), torch.full_like(conv.weight, torch.sigmoid(torch.tensor(2.0)).item()))


# --------------------------------------------------------------------------- training
def _train_once(seed):
    set_determinism(seed)
    model = VGG("vgg_tiny", num_classes=4, gated=True)
    cfg = TrainConfig(epochs=2, optimizer="sgd", lr=0.05, weight_decay=5e-4, gate_weight_decay=0.0,
                      gate_lambda=1e-4, loader_seed=seed, amp=False)
    data = DataLoader(loader().dataset, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(0))
    model, history, info = train_model(model, data, loader(), cfg, torch.device("cpu"), log=lambda *_: None)
    return model, history


def test_training_is_reproducible():
    a, ha = _train_once(7)
    b, hb = _train_once(7)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert [h["validation_loss"] for h in ha] == [h["validation_loss"] for h in hb]


def test_training_resume_matches_uninterrupted(tmp_path):
    def run(epochs, ckpt):
        set_determinism(3)
        model = MLP(3072, [8], 4, dropout=0.0)
        data = DataLoader(loader().dataset, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(0))
        cfg = TrainConfig(epochs=epochs, lr=1e-3, loader_seed=3, scheduler="constant")
        return train_model(model, data, loader(), cfg, torch.device("cpu"), checkpoint_dir=ckpt, log=lambda *_: None)
    full, _, _ = run(3, None)
    run(2, tmp_path)
    resumed, history, _ = run(3, tmp_path)
    assert [h["epoch"] for h in history] == [1, 2, 3]
    full_last = full  # best-state models: compare the final validation metric path
    assert evaluate(resumed, loader(), torch.device("cpu"))["loss"] == pytest.approx(
        evaluate(full_last, loader(), torch.device("cpu"))["loss"], rel=1e-5)


def test_fine_tune_keeps_architecture_and_zero_epochs_returns_initial():
    model = tiny_vgg()
    compact = compact_model(model, uniform_keep([torch.rand(g.n_units) for g in find_unit_groups(model)], 0.6))
    widths = unit_widths(compact)
    before = copy.deepcopy(compact.state_dict())
    cfg = TrainConfig(epochs=0, optimizer="sgd", lr=0.01)
    tuned, history, info = fine_tune(compact, loader(), loader(), cfg, torch.device("cpu"), log=lambda *_: None)
    assert info["best_epoch"] == 0 and len(history) == 1
    assert all(torch.equal(before[k], v) for k, v in tuned.state_dict().items())
    tuned, history, _ = fine_tune(compact, loader(), loader(), TrainConfig(epochs=1, optimizer="sgd", lr=0.01),
                                  torch.device("cpu"), log=lambda *_: None)
    assert unit_widths(tuned) == widths and [h["epoch"] for h in history] == [0, 1]


def test_dense_mode_freezes_gates():
    model = SelfPruningMLP(input_dim=3072, hidden_dims=[8], num_classes=4)
    model.set_mode("dense")
    before = model.prunable_layers[0].gate_scores.detach().clone()
    train_model(model, loader(), loader(), TrainConfig(epochs=1, train_gates=False), torch.device("cpu"),
                log=lambda *_: None)
    assert torch.equal(before, model.prunable_layers[0].gate_scores)
