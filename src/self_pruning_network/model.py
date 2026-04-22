from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GateSummary:
    layers: list[dict[str, float | int]]
    mean_gate_value: float
    sparsity_percent: float
    total_weights: int
    pruned_weights: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.bias)
        nn.init.constant_(self.gate_scores, 2.0)

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores)

    def sparsity_penalty(self) -> torch.Tensor:
        return self.gates().sum()

    def pruned_weight(self) -> torch.Tensor:
        return self.weight * self.gates()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.pruned_weight(), self.bias)


class SelfPruningMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 3072,
        hidden_dims: list[int] | None = None,
        num_classes: int = 10,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [2048, 1024, 512]
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm

        layers: list[nn.Module] = []
        layer_dims = [input_dim, *hidden_dims, num_classes]
        self.prunable_layers = nn.ModuleList()
        for index in range(len(layer_dims) - 1):
            layer = PrunableLinear(layer_dims[index], layer_dims[index + 1])
            self.prunable_layers.append(layer)
            layers.append(layer)
            if index < len(layer_dims) - 2:
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(layer_dims[index + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(p=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.view(inputs.size(0), -1)
        return self.network(batch)

    def sparsity_loss(self) -> torch.Tensor:
        penalties = [layer.sparsity_penalty() for layer in self.prunable_layers]
        return torch.stack(penalties).sum()

    def all_gate_values(self) -> torch.Tensor:
        return torch.cat([layer.gates().reshape(-1) for layer in self.prunable_layers], dim=0)

    def gate_summary(self, threshold: float = 1e-2) -> GateSummary:
        layers: list[dict[str, float | int]] = []
        total_weights = 0
        pruned_weights = 0
        weighted_gate_sum = 0.0

        for index, layer in enumerate(self.prunable_layers):
            gates = layer.gates().detach()
            count = gates.numel()
            pruned = int((gates < threshold).sum().item())
            mean_gate = float(gates.mean().item())
            total_weights += count
            pruned_weights += pruned
            weighted_gate_sum += mean_gate * count
            layers.append(
                {
                    "layer_index": index,
                    "weight_count": count,
                    "pruned_count": pruned,
                    "mean_gate_value": mean_gate,
                    "sparsity_percent": (pruned / count) * 100.0,
                }
            )

        mean_gate_value = weighted_gate_sum / total_weights if total_weights else 0.0
        sparsity_percent = (pruned_weights / total_weights) * 100.0 if total_weights else 0.0
        return GateSummary(
            layers=layers,
            mean_gate_value=mean_gate_value,
            sparsity_percent=sparsity_percent,
            total_weights=total_weights,
            pruned_weights=pruned_weights,
        )
