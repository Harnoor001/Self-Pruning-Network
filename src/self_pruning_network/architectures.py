"""Model builders for the comparative pruning study.

Every research model exposes a flat ``network`` ``nn.Sequential`` so that unit
groups (producer layer -> BatchNorm -> activation -> consumer layer) can be
discovered generically by ``structured_pruning.find_unit_groups``.

* MLP sources reuse the legacy ``SelfPruningMLP`` unchanged (gated or dense mode);
  ``MLP`` is its plain (gate-free) twin with an identical module layout, used for
  compact deployable models.
* ``VGG`` is a CIFAR-10 VGG with BatchNorm; ``gated=True`` swaps every conv/linear
  for its gated counterpart (the existing method applied to a CNN).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from self_pruning_network.gates import PrunableConv2d, PrunableLinear, effective_weight, is_gated
from self_pruning_network.model import SelfPruningMLP

VGG_CFGS: dict[str, list[int | str]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    # Tiny configuration for fast unit tests and CPU smoke runs only.
    "vgg_tiny": [8, "M", 16, "M", 16, "M", 32, "M", 32, "M"],
}


class MLP(nn.Module):
    """Gate-free MLP with exactly the same ``network`` layout as ``SelfPruningMLP``."""

    family = "mlp"

    def __init__(self, input_dim: int = 3072, hidden_dims: list[int] | None = None, num_classes: int = 10,
                 dropout: float = 0.3, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = [int(v) for v in (hidden_dims or [2048, 1024, 512])]
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        dims = [self.input_dim, *self.hidden_dims, self.num_classes]
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                if self.use_batchnorm:
                    layers.append(nn.BatchNorm1d(dims[index + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(p=self.dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs.view(inputs.size(0), -1))


class VGG(nn.Module):
    family = "vgg"

    def __init__(self, cfg: list[int | str] | str = "vgg16", num_classes: int = 10, gated: bool = False,
                 in_channels: int = 3, input_size: int = 32) -> None:
        super().__init__()
        self.cfg = list(VGG_CFGS[cfg]) if isinstance(cfg, str) else list(cfg)
        self.num_classes = int(num_classes)
        self.gated = bool(gated)
        self.in_channels = int(in_channels)
        self.input_size = int(input_size)
        layers: list[nn.Module] = []
        channels, size = self.in_channels, self.input_size
        for value in self.cfg:
            if value == "M":
                layers.append(nn.MaxPool2d(2, 2))
                size //= 2
                continue
            width = int(value)
            conv = (PrunableConv2d(channels, width, 3, padding=1, bias=False) if self.gated
                    else nn.Conv2d(channels, width, 3, padding=1, bias=False))
            if not self.gated:
                nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            layers += [conv, nn.BatchNorm2d(width), nn.ReLU(inplace=False)]
            channels = width
        layers.append(nn.Flatten())
        features = channels * size * size
        layers.append(PrunableLinear(features, self.num_classes) if self.gated else nn.Linear(features, self.num_classes))
        self.network = nn.Sequential(*layers)

    @property
    def widths(self) -> list[int]:
        return [int(v) for v in self.cfg if v != "M"]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def model_config(model: nn.Module) -> dict[str, object]:
    if isinstance(model, SelfPruningMLP):
        return {"family": "mlp", "gated": True, "input_dim": model.input_dim, "hidden_dims": list(model.hidden_dims),
                "num_classes": model.num_classes, "dropout": model.dropout, "use_batchnorm": model.use_batchnorm}
    if isinstance(model, MLP):
        return {"family": "mlp", "gated": False, "input_dim": model.input_dim, "hidden_dims": list(model.hidden_dims),
                "num_classes": model.num_classes, "dropout": model.dropout, "use_batchnorm": model.use_batchnorm}
    if isinstance(model, VGG):
        return {"family": "vgg", "gated": model.gated, "cfg": list(model.cfg), "num_classes": model.num_classes,
                "in_channels": model.in_channels, "input_size": model.input_size}
    raise TypeError(f"unsupported model type: {type(model).__name__}")


def build_model(config: dict[str, object]) -> nn.Module:
    config = dict(config)
    family = config.pop("family")
    gated = bool(config.pop("gated", False))
    if family == "mlp":
        cls = SelfPruningMLP if gated else MLP
        return cls(input_dim=int(config.get("input_dim", 3072)), hidden_dims=list(config["hidden_dims"]),
                   num_classes=int(config.get("num_classes", 10)), dropout=float(config.get("dropout", 0.3)),
                   use_batchnorm=bool(config.get("use_batchnorm", True)))
    if family == "vgg":
        return VGG(cfg=config.get("cfg", "vgg16"), num_classes=int(config.get("num_classes", 10)), gated=gated,
                   in_channels=int(config.get("in_channels", 3)), input_size=int(config.get("input_size", 32)))
    raise ValueError(f"unknown model family: {family}")


def with_widths(config: dict[str, object], widths: list[int]) -> dict[str, object]:
    """Return a gate-free config with the hidden widths replaced."""
    config = dict(config)
    config["gated"] = False
    if config["family"] == "mlp":
        config["hidden_dims"] = [int(w) for w in widths]
        return config
    cfg = VGG_CFGS[config["cfg"]] if isinstance(config["cfg"], str) else config["cfg"]
    if sum(v != "M" for v in cfg) != len(widths):
        raise ValueError(f"expected {sum(v != 'M' for v in cfg)} widths, got {len(widths)}")
    iterator = iter(widths)
    config["cfg"] = [v if v == "M" else int(next(iterator)) for v in cfg]
    return config


def unit_widths(model: nn.Module) -> list[int]:
    config = model_config(model)
    if config["family"] == "mlp":
        return list(config["hidden_dims"])
    return [int(v) for v in config["cfg"] if v != "M"]


def set_source_mode(model: nn.Module, mode: str) -> None:
    """Set dense/soft mode for every gated module (no-op for gate-free models)."""
    if isinstance(model, SelfPruningMLP):
        model.set_mode(mode)
        return
    for module in model.modules():
        if is_gated(module):
            module.set_mode(mode)


@torch.no_grad()
def to_plain(model: nn.Module) -> nn.Module:
    """Fold gates into weights and return a gate-free model computing the same function.

    For a gated source in soft mode the plain model uses ``W * sigmoid(S)``; in
    dense mode it uses ``W``; in hard mode ``W * M``. BatchNorm parameters and
    running statistics are copied unchanged.
    """
    config = model_config(model)
    if not config["gated"]:
        return copy.deepcopy(model)
    plain = build_model({**config, "gated": False})
    device = next(model.parameters()).device
    plain.to(device)
    for source, target in zip(model.network, plain.network):
        if is_gated(source):
            target.weight.copy_(effective_weight(source).detach())
            if source.bias is not None:
                target.bias.copy_(source.bias.detach())
        elif isinstance(source, (nn.BatchNorm1d, nn.BatchNorm2d)):
            target.load_state_dict(source.state_dict())
        elif isinstance(source, (nn.Linear, nn.Conv2d)):
            target.load_state_dict(source.state_dict())
    plain.train(model.training)
    return plain
