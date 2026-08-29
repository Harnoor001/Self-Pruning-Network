import torch
import pytest

from self_pruning_network.model import PrunableLinear, SelfPruningMLP
from self_pruning_network.train import accuracy_drop_percent_points


def test_prunable_linear_returns_expected_shape() -> None:
    layer = PrunableLinear(4, 3)
    inputs = torch.randn(2, 4)
    outputs = layer(inputs)
    assert outputs.shape == (2, 3)


def test_gradients_flow_through_weight_and_gate_scores() -> None:
    layer = PrunableLinear(4, 3)
    inputs = torch.randn(5, 4)
    targets = torch.randn(5, 3)

    predictions = layer(inputs)
    loss = torch.nn.functional.mse_loss(predictions, targets)
    loss.backward()

    assert layer.weight.grad is not None
    assert layer.gate_scores.grad is not None
    assert torch.count_nonzero(layer.gate_scores.grad).item() > 0


def test_sparsity_loss_is_positive() -> None:
    model = SelfPruningMLP(input_dim=8, hidden_dims=[6], num_classes=2)
    penalty = model.sparsity_loss()
    assert penalty.item() > 0


def test_gate_summary_counts_weights() -> None:
    model = SelfPruningMLP(input_dim=8, hidden_dims=[6], num_classes=2)
    summary = model.gate_summary()
    assert summary.total_weights == (8 * 6) + (6 * 2)
    assert summary.pruned_weights >= 0
    assert 0.0 <= summary.sparsity_percent <= 100.0


def test_gate_conversion_and_threshold_mask() -> None:
    layer = PrunableLinear(5, 1)
    with torch.no_grad():
        layer.gate_scores.copy_(torch.logit(torch.tensor([[0.001, 0.01, 0.1, 0.5, 0.9]])))
    assert torch.allclose(layer.gates(), torch.tensor([[0.001, 0.01, 0.1, 0.5, 0.9]]), atol=1e-5)
    assert torch.equal(layer.pruning_mask(0.1), torch.tensor([[0., 0., 1., 1., 1.]]))


def test_hard_pruning_zeroes_weights_and_survives_optimizer_step() -> None:
    layer = PrunableLinear(2, 2)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        layer.gate_scores.copy_(torch.logit(torch.tensor([[0.01, 0.9], [0.9, 0.01]])))
    layer.apply_mask(layer.pruning_mask(0.1))
    assert torch.equal(layer.weight, torch.tensor([[0.0, 2.0], [3.0, 0.0]]))
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.1, weight_decay=0.1)
    layer(torch.ones(3, 2)).sum().backward()
    optimizer.step()
    layer.enforce_mask()
    assert layer.weight[0, 0].item() == 0.0
    assert layer.weight[1, 1].item() == 0.0


def test_target_sparsity_accounting_and_persistence() -> None:
    model = SelfPruningMLP(input_dim=4, hidden_dims=[3], num_classes=2)
    with torch.no_grad():
        model.prunable_layers[0].gate_scores.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4))
        model.prunable_layers[1].gate_scores.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    model.hard_prune_target_sparsity(0.5)
    summary = model.gate_summary()
    assert summary.pruned_weights == 9
    assert summary.active_weights + summary.pruned_weights == summary.total_weights
    clone = SelfPruningMLP(input_dim=4, hidden_dims=[3], num_classes=2)
    clone.load_state_dict(model.state_dict())
    assert torch.equal(clone.prunable_layers[0].mask, model.prunable_layers[0].mask)
    assert torch.equal(clone.prunable_layers[1].weight, model.prunable_layers[1].weight)


def test_dense_mode_and_accuracy_drop_reference() -> None:
    model = SelfPruningMLP(input_dim=4, hidden_dims=[3], num_classes=2)
    inputs = torch.randn(2, 4)
    model.set_mode("dense")
    assert model(inputs).shape == (2, 2)
    assert accuracy_drop_percent_points(0.80, 0.75) == pytest.approx(5.0)


def test_gate_conversion_and_threshold_mask() -> None:
    layer = PrunableLinear(5, 1)
    with torch.no_grad():
        layer.gate_scores.copy_(torch.logit(torch.tensor([[0.001, 0.01, 0.1, 0.5, 0.9]])))
    assert torch.allclose(layer.gates(), torch.tensor([[0.001, 0.01, 0.1, 0.5, 0.9]]), atol=1e-5)
    assert torch.equal(layer.pruning_mask(0.1), torch.tensor([[0., 0., 1., 1., 1.]]))


def test_hard_pruning_zeroes_weights_and_survives_optimizer_step() -> None:
    layer = PrunableLinear(2, 2)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        layer.gate_scores.copy_(torch.logit(torch.tensor([[0.01, 0.9], [0.9, 0.01]])))
    layer.apply_mask(layer.pruning_mask(0.1))
    assert torch.equal(layer.weight, torch.tensor([[0.0, 2.0], [3.0, 0.0]]))
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.1, weight_decay=0.1)
    layer(torch.ones(3, 2)).sum().backward()
    optimizer.step()
    layer.enforce_mask()
    assert layer.weight[0, 0].item() == 0.0
    assert layer.weight[1, 1].item() == 0.0


def test_target_sparsity_and_mask_persistence() -> None:
    model = SelfPruningMLP(input_dim=4, hidden_dims=[3], num_classes=2)
    with torch.no_grad():
        model.prunable_layers[0].gate_scores.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4))
        model.prunable_layers[1].gate_scores.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    model.hard_prune_target_sparsity(0.5)
    summary = model.gate_summary()
    assert summary.pruned_weights == 9
    assert summary.active_weights == 9
    clone = SelfPruningMLP(input_dim=4, hidden_dims=[3], num_classes=2)
    clone.load_state_dict(model.state_dict())
    assert torch.equal(clone.prunable_layers[0].mask, model.prunable_layers[0].mask)
    assert torch.equal(clone.prunable_layers[1].mask, model.prunable_layers[1].mask)
    assert torch.equal(clone.prunable_layers[0].weight, model.prunable_layers[0].weight)


def test_dense_mode_remains_available() -> None:
    model = SelfPruningMLP(input_dim=4, hidden_dims=[3], num_classes=2)
    inputs = torch.randn(2, 4)
    model.set_mode("dense")
    dense = model(inputs)
    model.set_mode("soft")
    soft = model(inputs)
    assert dense.shape == soft.shape == (2, 2)


def test_pruned_checkpoint_reload_preserves_prediction(tmp_path) -> None:
    model = SelfPruningMLP(input_dim=4, hidden_dims=[3], num_classes=2)
    model.hard_prune_target_sparsity(0.5)
    inputs = torch.randn(2, 4)
    expected = model(inputs)
    path = tmp_path / "pruned.pt"
    torch.save({"model_config": {"input_dim": 4, "hidden_dims": [3], "num_classes": 2,
                                  "dropout": 0.3, "use_batchnorm": True},
                "model_state_dict": model.state_dict()}, path)
    payload = torch.load(path, weights_only=False)
    restored = SelfPruningMLP(**{key: payload["model_config"][key] for key in
                                 ("input_dim", "hidden_dims", "num_classes", "dropout", "use_batchnorm")})
    restored.load_state_dict(payload["model_state_dict"])
    assert torch.equal(restored.prunable_layers[0].mask, model.prunable_layers[0].mask)
    assert torch.equal(restored(inputs), expected)


def test_efficiency_accounting_and_mac_estimation() -> None:
    model = SelfPruningMLP(input_dim=4, hidden_dims=[3], num_classes=2, dropout=0.0, use_batchnorm=False)
    with torch.no_grad():
        model.prunable_layers[0].mask.copy_(torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]))
        model.prunable_layers[1].mask.copy_(torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))
        model.prunable_layers[0].hard_pruned.fill_(True)
        model.prunable_layers[1].hard_pruned.fill_(True)

    summary = model.efficiency_summary()

    assert summary["total_weights"] == 18
    assert summary["active_connections"] == 10
    assert summary["pruned_connections"] == 8
    assert summary["sparsity_percent"] == pytest.approx(44.444444)
    assert summary["density_percent"] == pytest.approx(55.555556)
    assert summary["logical_connectivity_reduction_percent"] == pytest.approx(44.444444)
    assert summary["estimated_dense_macs"] == 18
    assert summary["estimated_effective_macs"] == 10
    assert summary["theoretical_mac_reduction_percent"] == pytest.approx(44.444444)
    assert summary["layers"][0]["dense_macs"] == 12
    assert summary["layers"][0]["estimated_effective_macs"] == 7
    assert summary["layers"][1]["dense_macs"] == 6
    assert summary["layers"][1]["estimated_effective_macs"] == 3


def test_random_pruning_is_reproducible_and_exact() -> None:
    first = SelfPruningMLP(input_dim=4, hidden_dims=[3], num_classes=2)
    second = SelfPruningMLP(input_dim=4, hidden_dims=[3], num_classes=2)
    third = SelfPruningMLP(input_dim=4, hidden_dims=[3], num_classes=2)
    first.random_prune_target_sparsity(0.5, seed=42)
    second.random_prune_target_sparsity(0.5, seed=42)
    third.random_prune_target_sparsity(0.5, seed=123)

    first_mask = torch.cat([layer.mask.reshape(-1) for layer in first.prunable_layers])
    second_mask = torch.cat([layer.mask.reshape(-1) for layer in second.prunable_layers])
    third_mask = torch.cat([layer.mask.reshape(-1) for layer in third.prunable_layers])

    assert torch.equal(first_mask, second_mask)
    assert not torch.equal(first_mask, third_mask)
    assert int((first_mask == 0).sum().item()) == 9
    assert int((first_mask == 1).sum().item()) == 9


def test_learned_and_random_pruning_have_same_target_count() -> None:
    learned = SelfPruningMLP(input_dim=4, hidden_dims=[3], num_classes=2)
    random_model = SelfPruningMLP(input_dim=4, hidden_dims=[3], num_classes=2)
    learned.hard_prune_target_sparsity(0.5)
    random_model.random_prune_target_sparsity(0.5, seed=42)
    learned_summary = learned.efficiency_summary()
    random_summary = random_model.efficiency_summary()
    assert learned_summary["active_connections"] == random_summary["active_connections"] == 9
    assert learned_summary["pruned_connections"] == random_summary["pruned_connections"] == 9


def test_learned_mask_is_deterministic_for_same_gate_scores() -> None:
    first = SelfPruningMLP(input_dim=4, hidden_dims=[3], num_classes=2)
    second = SelfPruningMLP(input_dim=4, hidden_dims=[3], num_classes=2)
    scores = [torch.arange(12, dtype=torch.float32).reshape(3, 4),
              torch.arange(6, dtype=torch.float32).reshape(2, 3)]
    with torch.no_grad():
        for first_layer, second_layer, score in zip(first.prunable_layers, second.prunable_layers, scores):
            first_layer.gate_scores.copy_(score)
            second_layer.gate_scores.copy_(score)

    first.hard_prune_target_sparsity(0.5)
    second.hard_prune_target_sparsity(0.5)
    for first_layer, second_layer in zip(first.prunable_layers, second.prunable_layers):
        assert torch.equal(first_layer.mask, second_layer.mask)
