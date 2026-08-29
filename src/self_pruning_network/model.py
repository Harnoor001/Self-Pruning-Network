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
    active_weights: int
    density_percent: float
    threshold: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _percentage(numerator: int, denominator: int) -> float:
    return (numerator / denominator * 100.0) if denominator else 0.0


class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))
        self.register_buffer("mask", torch.ones(out_features, in_features))
        self.register_buffer("hard_pruned", torch.tensor(False, dtype=torch.bool))
        self._dense_mode = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.bias)
        nn.init.constant_(self.gate_scores, 2.0)

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores)

    def pruning_mask(self, threshold: float) -> torch.Tensor:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        return (self.gates() >= threshold).to(self.weight.dtype)

    @torch.no_grad()
    def apply_mask(self, mask: torch.Tensor, hard: bool = True) -> None:
        if mask.shape != self.weight.shape:
            raise ValueError("mask shape must match weight shape")
        self.mask.copy_(mask.to(device=self.weight.device, dtype=self.weight.dtype))
        self.weight.mul_(self.mask)
        self.hard_pruned.fill_(hard)

    @torch.no_grad()
    def enforce_mask(self) -> None:
        if bool(self.hard_pruned.item()):
            self.weight.mul_(self.mask)

    def set_mode(self, mode: str) -> None:
        if mode not in {"soft", "dense", "hard"}:
            raise ValueError("mode must be 'soft', 'dense', or 'hard'")
        self.hard_pruned.fill_(mode == "hard")
        self._dense_mode = mode == "dense"

    def sparsity_penalty(self) -> torch.Tensor:
        return self.gates().sum()

    def pruned_weight(self) -> torch.Tensor:
        if bool(self.hard_pruned.item()):
            return self.weight * self.mask
        if self._dense_mode:
            return self.weight
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
        self._dense_mode = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.view(inputs.size(0), -1)
        return self.network(batch)

    def sparsity_loss(self) -> torch.Tensor:
        penalties = [layer.sparsity_penalty() for layer in self.prunable_layers]
        return torch.stack(penalties).sum()

    def all_gate_values(self) -> torch.Tensor:
        return torch.cat([layer.gates().reshape(-1) for layer in self.prunable_layers], dim=0)

    def set_mode(self, mode: str) -> None:
        if mode not in {"soft", "dense", "hard"}:
            raise ValueError("mode must be 'soft', 'dense', or 'hard'")
        self._dense_mode = mode == "dense"
        for layer in self.prunable_layers:
            layer.set_mode(mode)

    @torch.no_grad()
    def hard_prune(self, threshold: float) -> list[torch.Tensor]:
        masks = [layer.pruning_mask(threshold) for layer in self.prunable_layers]
        for layer, mask in zip(self.prunable_layers, masks):
            layer.apply_mask(mask, hard=True)
        self._dense_mode = False
        return masks

    @torch.no_grad()
    def hard_prune_target_sparsity(self, target_sparsity: float) -> list[torch.Tensor]:
        if not 0.0 <= target_sparsity <= 1.0:
            raise ValueError("target_sparsity must be between 0 and 1")
        gates = self.all_gate_values()
        prune_count = int(round(gates.numel() * target_sparsity))
        keep_count = gates.numel() - prune_count
        if keep_count == 0:
            flat_mask = torch.zeros_like(gates)
        elif prune_count == 0:
            flat_mask = torch.ones_like(gates)
        else:
            keep_indices = torch.topk(gates, k=keep_count, largest=True, sorted=False).indices
            flat_mask = torch.zeros_like(gates)
            flat_mask[keep_indices] = 1.0
        masks: list[torch.Tensor] = []
        offset = 0
        for layer in self.prunable_layers:
            size = layer.weight.numel()
            mask = flat_mask[offset : offset + size].reshape_as(layer.weight)
            layer.apply_mask(mask, hard=True)
            masks.append(mask)
            offset += size
        self._dense_mode = False
        return masks

    @torch.no_grad()
    def random_prune_target_sparsity(self, target_sparsity: float, seed: int) -> list[torch.Tensor]:
        """Apply a reproducible global random mask at the requested sparsity.

        The random selection is made over one population containing every
        prunable connection, making it directly comparable with global gate
        ranking. A local CPU generator keeps the result independent of global
        RNG state and reproducible for the same model shape, target, and seed.
        """
        if not 0.0 <= target_sparsity <= 1.0:
            raise ValueError("target_sparsity must be between 0 and 1")
        total = sum(layer.weight.numel() for layer in self.prunable_layers)
        prune_count = int(round(total * target_sparsity))
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        permutation = torch.randperm(total, generator=generator, device="cpu")
        flat_mask = torch.ones(total, dtype=torch.float32)
        if prune_count:
            flat_mask[permutation[:prune_count]] = 0.0

        masks: list[torch.Tensor] = []
        offset = 0
        for layer in self.prunable_layers:
            size = layer.weight.numel()
            mask = flat_mask[offset : offset + size].reshape_as(layer.weight).to(layer.weight.device)
            layer.apply_mask(mask, hard=True)
            masks.append(mask)
            offset += size
        self._dense_mode = False
        return masks

    @torch.no_grad()
    def enforce_masks(self) -> None:
        for layer in self.prunable_layers:
            layer.enforce_mask()

    def mask_summary(self, threshold: float | None = None) -> dict[str, float | int]:
        total = sum(layer.weight.numel() for layer in self.prunable_layers)
        active = sum(int(layer.mask.count_nonzero().item()) for layer in self.prunable_layers)
        return {
            "total_weights": total,
            "active_weights": active,
            "pruned_weights": total - active,
            "sparsity_percent": ((total - active) / total * 100.0) if total else 0.0,
            "density_percent": (active / total * 100.0) if total else 0.0,
            "threshold": threshold,
        }

    def parameter_summary(self) -> dict[str, float | int]:
        """Report logical connection counts; gate parameters are training-only."""
        summary = self.mask_summary()
        summary["parameter_reduction_percent"] = summary["sparsity_percent"]
        summary["total_dense_parameter_slots"] = summary["total_weights"]
        summary["active_connections"] = summary["active_weights"]
        summary["pruned_connections"] = summary["pruned_weights"]
        summary["logical_connectivity_reduction_percent"] = summary["sparsity_percent"]
        return summary

    def efficiency_summary(self) -> dict[str, object]:
        """Return logical connectivity and ideal sparse-compute estimates.

        MAC counts intentionally describe an ideal sparse execution path. The
        model still owns dense tensors, so this method does not claim physical
        parameter removal or measured runtime acceleration.
        """
        layers: list[dict[str, int | float | str]] = []
        total_weights = 0
        active_weights = 0
        dense_macs = 0
        effective_macs = 0

        for index, layer in enumerate(self.prunable_layers):
            total = int(layer.weight.numel())
            active = int(layer.mask.count_nonzero().item())
            pruned = total - active
            layer_dense_macs = int(layer.in_features * layer.out_features)
            layer_effective_macs = active
            total_weights += total
            active_weights += active
            dense_macs += layer_dense_macs
            effective_macs += layer_effective_macs
            layers.append(
                {
                    "layer_name": f"layer{index + 1}",
                    "layer_index": index,
                    "input_features": int(layer.in_features),
                    "output_features": int(layer.out_features),
                    "total_weights": total,
                    "active_weights": active,
                    "pruned_weights": pruned,
                    "sparsity_percent": _percentage(pruned, total),
                    "density_percent": _percentage(active, total),
                    "dense_macs": layer_dense_macs,
                    "estimated_effective_macs": layer_effective_macs,
                }
            )

        pruned_weights = total_weights - active_weights
        return {
            "total_dense_parameter_slots": total_weights,
            "total_weights": total_weights,
            "active_connections": active_weights,
            "active_weights": active_weights,
            "pruned_connections": pruned_weights,
            "pruned_weights": pruned_weights,
            "density_percent": _percentage(active_weights, total_weights),
            "sparsity_percent": _percentage(pruned_weights, total_weights),
            "logical_connectivity_reduction_percent": _percentage(pruned_weights, total_weights),
            "estimated_dense_macs": dense_macs,
            "estimated_effective_macs": effective_macs,
            "theoretical_mac_reduction_percent": _percentage(dense_macs - effective_macs, dense_macs),
            "layers": layers,
        }

    def gate_summary(self, threshold: float | None = None) -> GateSummary:
        layers: list[dict[str, float | int]] = []
        total_weights = 0
        pruned_weights = 0
        weighted_gate_sum = 0.0

        for index, layer in enumerate(self.prunable_layers):
            gates = layer.gates().detach()
            count = gates.numel()
            mask = layer.mask.detach()
            pruned = int((mask == 0).sum().item())
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
                    "active_count": count - pruned,
                    "density_percent": ((count - pruned) / count) * 100.0,
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
            active_weights=total_weights - pruned_weights,
            density_percent=((total_weights - pruned_weights) / total_weights) * 100.0 if total_weights else 0.0,
            threshold=threshold,
        )
