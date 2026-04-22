import torch

from self_pruning_network.model import PrunableLinear, SelfPruningMLP


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

