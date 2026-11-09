"""Generic training / fine-tuning loop for the research framework.

Supports the three source objectives of the study:

* dense:     ``CE``
* gated:     ``CE + gate_lambda * P(sigmoid(S))``      (existing method, P = sum by default)
* slimming:  ``CE + slimming_lambda * sum |gamma|``    (Network Slimming)

Checkpoint selection is by validation accuracy. Optional resumable state is
written to ``<checkpoint_dir>/last.pt`` after every epoch.
"""
from __future__ import annotations

import math
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from self_pruning_network.baselines.network_slimming import slimming_penalty
from self_pruning_network.evaluation import evaluate
from self_pruning_network.gates import gate_penalty, gated_modules
from self_pruning_network.research_data import reseed_loader


@dataclass
class TrainConfig:
    epochs: int = 1
    optimizer: str = "adamw"            # adamw | sgd
    lr: float = 8e-4
    weight_decay: float = 1e-4
    gate_weight_decay: float | None = None   # None -> same as weight_decay (legacy AdamW behaviour)
    momentum: float = 0.9
    nesterov: bool = True
    scheduler: str = "cosine"           # cosine | constant
    label_smoothing: float = 0.05
    gate_lambda: float = 0.0
    gate_penalty_norm: str = "sum"
    slimming_lambda: float = 0.0
    train_gates: bool = True            # False -> gates frozen (dense mode)
    amp: bool = True                    # used only on CUDA
    include_initial: bool = False       # evaluate epoch 0 as a selection candidate (fine-tuning)
    loader_seed: int = 0
    max_steps_per_epoch: int | None = None  # smoke tests only

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def set_determinism(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    gate_params, other = [], []
    for name, parameter in model.named_parameters():
        if "gate_scores" in name:
            if cfg.train_gates:
                gate_params.append(parameter)
            else:
                parameter.requires_grad_(False)
        else:
            other.append(parameter)
    gate_wd = cfg.weight_decay if cfg.gate_weight_decay is None else cfg.gate_weight_decay
    groups = [{"params": other, "weight_decay": cfg.weight_decay}]
    if gate_params:
        groups.append({"params": gate_params, "weight_decay": gate_wd})
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum, nesterov=cfg.nesterov)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_model(model: nn.Module, train_loader, validation_loader, cfg: TrainConfig, device: torch.device,
                checkpoint_dir: str | Path | None = None, resume: bool = True,
                log: Callable[[str], None] = print, tag: str = "") -> tuple[nn.Module, list[dict], dict]:
    model.to(device)
    optimizer = build_optimizer(model, cfg)
    total_epochs = max(0, int(cfg.epochs))
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs))
                 if cfg.scheduler == "cosine" else torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0))
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    has_gates = bool(gated_modules(model)) and cfg.train_gates
    history: list[dict] = []
    best_acc, best_epoch, best_state = -math.inf, -1, None
    start_epoch, elapsed = 1, 0.0
    last_path = Path(checkpoint_dir) / "last.pt" if checkpoint_dir else None

    if last_path is not None and resume and last_path.exists():
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        history, best_acc, best_epoch = payload["history"], payload["best_acc"], payload["best_epoch"]
        best_state, start_epoch, elapsed = payload["best_state"], payload["epoch"] + 1, payload["elapsed"]
        log(f"[{tag}] resumed from epoch {payload['epoch']}")
    elif cfg.include_initial or total_epochs == 0:
        metrics = evaluate(model, validation_loader, device)
        history.append({"epoch": 0, "train_loss": None, "train_accuracy": None, "penalty": None,
                        "validation_loss": metrics["loss"], "validation_accuracy": metrics["accuracy"],
                        "lr": None, "epoch_time_s": 0.0})
        best_acc, best_epoch, best_state = metrics["accuracy"], 0, _state(model)

    for epoch in range(start_epoch, total_epochs + 1):
        reseed_loader(train_loader, cfg.loader_seed * 100_003 + epoch)
        model.train()
        start = time.perf_counter()
        loss_sum = correct = seen = 0.0
        penalty_value = 0.0
        for step, (inputs, targets) in enumerate(train_loader):
            if cfg.max_steps_per_epoch is not None and step >= cfg.max_steps_per_epoch:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits, targets)
            penalty = torch.zeros((), device=device)
            if has_gates and cfg.gate_lambda:
                penalty = penalty + cfg.gate_lambda * gate_penalty(model, cfg.gate_penalty_norm)
            if cfg.slimming_lambda:
                penalty = penalty + cfg.slimming_lambda * slimming_penalty(model)
            total = loss.float() + penalty
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
            n = targets.shape[0]
            loss_sum += float(loss.detach()) * n
            correct += float((logits.detach().argmax(1) == targets).sum())
            seen += n
            penalty_value = float(penalty.detach())
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - start
        elapsed += epoch_time
        metrics = evaluate(model, validation_loader, device)
        history.append({"epoch": epoch, "train_loss": loss_sum / max(1, seen), "train_accuracy": correct / max(1, seen),
                        "penalty": penalty_value, "validation_loss": metrics["loss"],
                        "validation_accuracy": metrics["accuracy"], "lr": lr, "epoch_time_s": epoch_time})
        if metrics["accuracy"] > best_acc:
            best_acc, best_epoch, best_state = metrics["accuracy"], epoch, _state(model)
        log(f"[{tag}] epoch {epoch}/{total_epochs} train_loss={loss_sum / max(1, seen):.4f} "
            f"val_acc={metrics['accuracy']:.4f} ({epoch_time:.1f}s)")
        if last_path is not None:
            last_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = last_path.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "history": history,
                        "best_acc": best_acc, "best_epoch": best_epoch, "best_state": best_state,
                        "epoch": epoch, "elapsed": elapsed, "config": cfg.to_dict()}, tmp)
            os.replace(tmp, last_path)

    if best_state is None:  # epochs == 0 and no initial evaluation requested
        best_state, best_epoch = _state(model), 0
    model.load_state_dict(best_state)
    info = {"best_epoch": best_epoch, "best_validation_accuracy": best_acc, "train_time_s": elapsed,
            "epochs": total_epochs, "amp": use_amp}
    return model, history, info


def fine_tune(model: nn.Module, train_loader, validation_loader, cfg: TrainConfig, device: torch.device,
              **kwargs) -> tuple[nn.Module, list[dict], dict]:
    """Fine-tune a compact model; epoch 0 (no fine-tuning) is a selection candidate."""
    return train_model(model, train_loader, validation_loader, replace(cfg, include_initial=True), device, **kwargs)


def train_config_from_dict(values: dict[str, object]) -> TrainConfig:
    known = set(TrainConfig.__dataclass_fields__)
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown training options: {sorted(unknown)}")
    return TrainConfig(**values)

