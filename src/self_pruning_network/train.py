from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from self_pruning_network.data import build_cifar10_loaders
from self_pruning_network.model import SelfPruningMLP
from self_pruning_network.reporting import (
    plot_accuracy_vs_sparsity, plot_gate_distribution, plot_layer_metrics,
    plot_lambda_metric, plot_training_history, write_markdown_report,
    write_results_table,
)


@dataclass
class RunResult:
    model_type: str
    lambda_value: float | None
    target_sparsity_percent: float | None
    validation_accuracy: float
    test_accuracy: float
    sparsity_percent: float
    density_percent: float
    total_weights: int
    active_weights: int
    pruned_weights: int
    reference_label: str | None
    reference_test_accuracy: float | None
    accuracy_drop_percent_points: float | None
    pre_finetune_test_accuracy: float | None
    checkpoint_path: str
    gate_summary: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "lambda": self.lambda_value,
            "target_sparsity_percent": self.target_sparsity_percent,
            "validation_accuracy": round(self.validation_accuracy, 6),
            "test_accuracy": round(self.test_accuracy, 6),
            "sparsity_percent": round(self.sparsity_percent, 6),
            "density_percent": round(self.density_percent, 6),
            "total_weights": self.total_weights,
            "active_weights": self.active_weights,
            "pruned_weights": self.pruned_weights,
            "reference_label": self.reference_label,
            "reference_test_accuracy": (
                round(self.reference_test_accuracy, 6)
                if self.reference_test_accuracy is not None else None
            ),
            "accuracy_drop_percent_points": (
                round(self.accuracy_drop_percent_points, 6)
                if self.accuracy_drop_percent_points is not None else None
            ),
            "pre_finetune_test_accuracy": (
                round(self.pre_finetune_test_accuracy, 6)
                if self.pre_finetune_test_accuracy is not None else None
            ),
            "checkpoint_path": self.checkpoint_path,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_loader_seed(train_loader, seed: int) -> None:
    generator = getattr(getattr(train_loader, "sampler", None), "generator", None)
    if generator is not None:
        generator.manual_seed(seed)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == targets).float().mean().item())


def accuracy_drop_percent_points(reference_accuracy: float, evaluated_accuracy: float) -> float:
    return (reference_accuracy - evaluated_accuracy) * 100.0


def evaluate(model: SelfPruningMLP, loader, device: torch.device, criterion: nn.Module, lambda_value: float = 0.0) -> dict[str, float | int]:
    model.eval()
    total_loss = total_accuracy = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets) + lambda_value * model.sparsity_loss()
            count = inputs.size(0)
            total_loss += float(loss.item()) * count
            total_accuracy += accuracy_from_logits(logits, targets) * count
            total_samples += count
    summary = model.gate_summary()
    return {
        "loss": total_loss / total_samples, "accuracy": total_accuracy / total_samples,
        "sparsity_percent": summary.sparsity_percent, "density_percent": summary.density_percent,
        "total_weights": summary.total_weights, "active_weights": summary.active_weights,
        "pruned_weights": summary.pruned_weights,
    }


def _model_config(model: SelfPruningMLP) -> dict[str, object]:
    return {"input_dim": model.input_dim, "hidden_dims": model.hidden_dims,
            "num_classes": model.num_classes, "dropout": model.dropout,
            "use_batchnorm": model.use_batchnorm}


def _save_checkpoint(path: Path, model: SelfPruningMLP, *, lambda_value: float | None,
                     metrics: dict[str, object], history: list[dict[str, float]],
                     seed: int, experiment_config: dict[str, object],
                     pruning: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 3, "lambda_value": lambda_value, "seed": seed,
        "model_config": _model_config(model), "experiment_config": experiment_config,
        "metrics": metrics, "gate_summary": model.gate_summary(pruning.get("threshold")).to_dict(),
        "pruning": pruning, "training_history": history,
        "model_state_dict": model.state_dict(),
    }, path)


def _trainable_parameters(model: SelfPruningMLP, mode: str) -> list[nn.Parameter]:
    if mode == "dense":
        return [parameter for name, parameter in model.named_parameters()
                if "gate_scores" not in name]
    return list(model.parameters())


def train_model(model_type: str, lambda_value: float, train_loader, validation_loader,
                test_loader, device: torch.device, epochs: int, learning_rate: float,
                weight_decay: float, label_smoothing: float, hidden_dims: list[int],
                dropout: float, use_batchnorm: bool, output_dir: Path, seed: int,
                experiment_config: dict[str, object]) -> tuple[RunResult, SelfPruningMLP, list[dict[str, float]]]:
    set_seed(seed)
    reset_loader_seed(train_loader, seed)
    model = SelfPruningMLP(hidden_dims=hidden_dims, dropout=dropout,
                           use_batchnorm=use_batchnorm).to(device)
    mode = "dense" if model_type == "dense" else "soft"
    model.set_mode(mode)
    optimizer = AdamW(_trainable_parameters(model, mode), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    best_validation_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    path = output_dir / "checkpoints" / f"{model_type}_lambda_{lambda_value:.4f}.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = running_accuracy = 0.0
        sample_count = 0
        for inputs, targets in tqdm(train_loader, desc=f"{model_type} lambda={lambda_value} epoch={epoch}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            penalty = lambda_value * model.sparsity_loss() if model_type == "soft" else 0.0
            loss = criterion(logits, targets) + penalty
            loss.backward()
            optimizer.step()
            count = inputs.size(0)
            running_loss += float(loss.item()) * count
            running_accuracy += accuracy_from_logits(logits, targets) * count
            sample_count += count
        validation_metrics = evaluate(model, validation_loader, device, criterion,
                                      lambda_value if model_type == "soft" else 0.0)
        history.append({"model_type": model_type, "lambda": lambda_value, "epoch": epoch,
                        "train_loss": running_loss / sample_count,
                        "train_accuracy": running_accuracy / sample_count,
                        "validation_loss": float(validation_metrics["loss"]),
                        "validation_accuracy": float(validation_metrics["accuracy"]),
                        "validation_sparsity_percent": float(validation_metrics["sparsity_percent"])})
        if float(validation_metrics["accuracy"]) > best_validation_accuracy:
            best_validation_accuracy = float(validation_metrics["accuracy"])
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        scheduler.step()

    assert best_state is not None
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, criterion,
                            lambda_value if model_type == "soft" else 0.0)
    summary = model.gate_summary()
    _save_checkpoint(path, model, lambda_value=lambda_value, metrics=test_metrics,
                     history=history, seed=seed, experiment_config=experiment_config,
                     pruning={"mode": mode, "strategy": None, "threshold": None,
                              "target_sparsity": None})
    result = RunResult(model_type, lambda_value, None, best_validation_accuracy,
                       float(test_metrics["accuracy"]), summary.sparsity_percent,
                       summary.density_percent, summary.total_weights,
                       summary.active_weights, summary.pruned_weights, None, None, None,
                       None, str(path), summary.to_dict())
    return result, model, history


def fine_tune(model: SelfPruningMLP, train_loader, validation_loader, test_loader,
              device: torch.device, epochs: int, learning_rate: float,
              weight_decay: float, label_smoothing: float,
              lambda_value: float) -> tuple[float, float, float, list[dict[str, float]]]:
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = AdamW(_trainable_parameters(model, "hard"), lr=learning_rate, weight_decay=weight_decay)
    before_metrics = evaluate(model, test_loader, device, criterion)
    before_accuracy = float(before_metrics["accuracy"])
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_accuracy = -1.0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            criterion(model(inputs), targets).backward()
            optimizer.step()
            model.enforce_masks()
        metrics = evaluate(model, validation_loader, device, criterion)
        history.append({"epoch": epoch, "validation_accuracy": float(metrics["accuracy"]),
                        "validation_loss": float(metrics["loss"])})
        if float(metrics["accuracy"]) > best_accuracy:
            best_accuracy = float(metrics["accuracy"])
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.enforce_masks()
    after_metrics = evaluate(model, test_loader, device, criterion)
    return best_accuracy, float(after_metrics["accuracy"]), before_accuracy, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible dense/soft/hard CIFAR-10 benchmark.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[2048, 1024, 512])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--disable-batchnorm", action="store_true")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.001])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--pruning-strategy", choices=["threshold", "target"], default="target")
    parser.add_argument("--prune-threshold", type=float, default=0.1)
    parser.add_argument("--target-sparsities", type=float, nargs="+", default=[20, 40, 60, 80])
    parser.add_argument("--fine-tune-epochs", type=int, default=1)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=2e-4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    output_dir, reports_dir = Path(args.output_dir), Path(args.output_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train_loader, validation_loader, test_loader = build_cifar10_loaders(
        args.data_dir, args.batch_size, args.num_workers, train_subset=args.train_subset,
        test_subset=args.test_subset, seed=args.seed)

    base_config = {
        "dataset": "CIFAR-10", "data_dir": str(args.data_dir), "train_subset": args.train_subset,
        "test_subset": args.test_subset, "validation_ratio": 0.1, "seed": args.seed,
        "train_samples": len(train_loader.dataset), "validation_samples": len(validation_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "architecture": _model_config(SelfPruningMLP(hidden_dims=args.hidden_dims,
                                                      dropout=args.dropout,
                                                      use_batchnorm=not args.disable_batchnorm)),
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "optimizer": "AdamW", "weight_decay": args.weight_decay, "scheduler": "CosineAnnealingLR",
        "label_smoothing": args.label_smoothing, "fine_tune_epochs": args.fine_tune_epochs,
        "fine_tune_learning_rate": args.fine_tune_learning_rate,
    }

    dense_result, dense_model, dense_history = train_model(
        "dense", 0.0, train_loader, validation_loader, test_loader, device, args.epochs,
        args.learning_rate, args.weight_decay, args.label_smoothing, args.hidden_dims,
        args.dropout, not args.disable_batchnorm, output_dir, args.seed, base_config)
    dense_accuracy = dense_result.test_accuracy
    results = [RunResult(
        "dense", None, 0.0, dense_result.validation_accuracy, dense_result.test_accuracy,
        0.0, 100.0, dense_result.total_weights, dense_result.total_weights, 0,
        "dense", dense_accuracy, 0.0, None, dense_result.checkpoint_path,
        dense_result.gate_summary)]
    histories = dense_history

    soft_results: list[RunResult] = []
    best_soft: RunResult | None = None
    best_soft_model: SelfPruningMLP | None = None
    for lambda_value in args.lambdas:
        result, model, history = train_model(
            "soft", lambda_value, train_loader, validation_loader, test_loader, device,
            args.epochs, args.learning_rate, args.weight_decay, args.label_smoothing,
            args.hidden_dims, args.dropout, not args.disable_batchnorm, output_dir, args.seed,
            {**base_config, "lambda": lambda_value})
        result.reference_label = "dense"
        result.reference_test_accuracy = dense_accuracy
        result.accuracy_drop_percent_points = accuracy_drop_percent_points(dense_accuracy, result.test_accuracy)
        soft_results.append(result)
        histories.extend(history)
        if best_soft is None or result.validation_accuracy > best_soft.validation_accuracy:
            best_soft, best_soft_model = result, model
    assert best_soft is not None and best_soft_model is not None
    results.extend(soft_results)

    pruning_results: list[RunResult] = []
    if args.prune:
        if args.pruning_strategy == "threshold":
            variants = [("threshold", args.prune_threshold, None)]
        else:
            variants = [("target", None, value / 100.0)
                        for value in [0.0, *args.target_sparsities]]
        for strategy, threshold, target in variants:
            variant = copy.deepcopy(best_soft_model).to(device)
            if strategy == "threshold":
                variant.hard_prune(float(threshold))
            else:
                variant.hard_prune_target_sparsity(float(target))
            val_acc, test_acc, before_acc, ft_history = fine_tune(
                variant, train_loader, validation_loader, test_loader, device,
                args.fine_tune_epochs, args.fine_tune_learning_rate, args.weight_decay,
                args.label_smoothing, best_soft.lambda_value or 0.0)
            summary = variant.gate_summary(threshold)
            selector = threshold if threshold is not None else target
            path = output_dir / "checkpoints" / f"hard_{strategy}_{selector:.4f}.pt"
            metrics = evaluate(variant, test_loader, device, nn.CrossEntropyLoss())
            pruning = {"mode": "hard", "strategy": strategy, "threshold": threshold,
                        "target_sparsity": target}
            _save_checkpoint(path, variant, lambda_value=best_soft.lambda_value,
                             metrics=metrics, history=ft_history, seed=args.seed,
                             experiment_config={**base_config, "lambda": best_soft.lambda_value,
                                                "pruning_strategy": strategy,
                                                "prune_threshold": threshold,
                                                "target_sparsity": target},
                             pruning=pruning)
            result = RunResult(
                "hard", best_soft.lambda_value, None if target is None else target * 100.0,
                val_acc, test_acc, summary.sparsity_percent, summary.density_percent,
                summary.total_weights, summary.active_weights, summary.pruned_weights,
                f"soft_lambda_{best_soft.lambda_value}", best_soft.test_accuracy,
                accuracy_drop_percent_points(best_soft.test_accuracy, test_acc),
                before_acc, str(path), summary.to_dict())
            pruning_results.append(result)
        results.extend(pruning_results)

    results_frame = write_results_table([item.to_dict() for item in results], reports_dir / "results.csv")
    history_frame = pd.DataFrame(histories)
    history_frame.to_csv(reports_dir / "training_history.csv", index=False)
    write_markdown_report(
        results_frame, best_soft.lambda_value, best_soft.test_accuracy, best_soft.sparsity_percent,
        reports_dir / "results.md", dense_accuracy=dense_accuracy, pruning_enabled=args.prune,
        experiment_config={**base_config, "selected_lambda": best_soft.lambda_value,
                           "pruning_strategy": args.pruning_strategy,
                           "prune_threshold": args.prune_threshold,
                           "target_sparsities_percent": args.target_sparsities})
    plot_gate_distribution(best_soft_model.all_gate_values().detach().cpu().numpy(),
                            reports_dir / "gate_distribution.png",
                            f"Gate Distribution for lambda={best_soft.lambda_value}")
    soft_frame = results_frame[results_frame["model_type"] == "soft"]
    plot_lambda_metric(soft_frame, "test_accuracy", reports_dir / "lambda_vs_test_accuracy.png",
                       "Lambda vs Test Accuracy", "Test Accuracy")
    plot_lambda_metric(soft_frame, "sparsity_percent", reports_dir / "lambda_vs_sparsity.png",
                       "Lambda vs Soft-Model Mask Sparsity", "Logical Sparsity (%)")
    plot_accuracy_vs_sparsity(results_frame, reports_dir / "accuracy_vs_sparsity.png")
    if not history_frame.empty:
        plot_training_history(history_frame, reports_dir / "training_history.png")
    plot_layer_metrics(best_soft.gate_summary["layers"], reports_dir / "layer_metrics.png")

    hard_results = [item.to_dict() for item in pruning_results]
    summary_payload = {
        "format_version": 3, "experiment_config": base_config,
        "dense_baseline": dense_result.to_dict(),
        "soft_results": [item.to_dict() for item in soft_results],
        "hard_pruned_results": hard_results,
        "selected_lambda": best_soft.lambda_value,
        "selected_soft_result": best_soft.to_dict(),
        "pruning_config": {"enabled": args.prune, "strategy": args.pruning_strategy,
                           "threshold": args.prune_threshold,
                           "target_sparsities_percent": args.target_sparsities},
        "report_files": sorted(path.name for path in reports_dir.iterdir() if path.is_file()),
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2),
                                               encoding="utf-8")
    return 0
