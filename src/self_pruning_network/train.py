from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from self_pruning_network.data import build_cifar10_loaders
from self_pruning_network.model import SelfPruningMLP
from self_pruning_network.reporting import plot_gate_distribution, write_markdown_report, write_results_table


@dataclass
class RunResult:
    lambda_value: float
    validation_accuracy: float
    test_accuracy: float
    sparsity_percent: float
    checkpoint_path: str

    def to_dict(self) -> dict[str, float | str]:
        return {
            "lambda": self.lambda_value,
            "validation_accuracy": round(self.validation_accuracy, 4),
            "test_accuracy": round(self.test_accuracy, 4),
            "sparsity_percent": round(self.sparsity_percent, 2),
            "checkpoint_path": self.checkpoint_path,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == targets).float().mean().item())


def evaluate(model: SelfPruningMLP, loader, device: torch.device, criterion: nn.Module, lambda_value: float) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            classification_loss = criterion(logits, targets)
            sparsity_loss = model.sparsity_loss()
            loss = classification_loss + lambda_value * sparsity_loss

            batch_size = inputs.size(0)
            total_loss += float(loss.item()) * batch_size
            total_accuracy += accuracy_from_logits(logits, targets) * batch_size
            total_samples += batch_size

    gate_summary = model.gate_summary()
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_accuracy / total_samples,
        "sparsity_percent": gate_summary.sparsity_percent,
    }


def train_single_lambda(
    lambda_value: float,
    train_loader,
    validation_loader,
    test_loader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    label_smoothing: float,
    hidden_dims: list[int],
    dropout: float,
    use_batchnorm: bool,
    output_dir: Path,
) -> tuple[RunResult, SelfPruningMLP]:
    model = SelfPruningMLP(hidden_dims=hidden_dims, dropout=dropout, use_batchnorm=use_batchnorm).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_validation_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    checkpoint_path = output_dir / "checkpoints" / f"best_lambda_{lambda_value:.4f}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"lambda={lambda_value} epoch={epoch}", leave=False)
        running_loss = 0.0
        running_accuracy = 0.0
        sample_count = 0

        for inputs, targets in progress:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            classification_loss = criterion(logits, targets)
            sparsity_loss = model.sparsity_loss()
            loss = classification_loss + lambda_value * sparsity_loss
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += float(loss.item()) * batch_size
            running_accuracy += accuracy_from_logits(logits, targets) * batch_size
            sample_count += batch_size
            progress.set_postfix(
                train_loss=f"{running_loss / sample_count:.4f}",
                train_acc=f"{running_accuracy / sample_count:.4f}",
            )

        validation_metrics = evaluate(model, validation_loader, device, criterion, lambda_value)
        if validation_metrics["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_metrics["accuracy"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        scheduler.step()

    assert best_state is not None, "Best model state was not captured."
    model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device, criterion, lambda_value)
    gate_summary = model.gate_summary()

    torch.save(
        {
            "lambda_value": lambda_value,
            "model_config": {
                "input_dim": model.input_dim,
                "hidden_dims": model.hidden_dims,
                "num_classes": model.num_classes,
                "dropout": model.dropout,
                "use_batchnorm": model.use_batchnorm,
            },
            "metrics": test_metrics,
            "gate_summary": gate_summary.to_dict(),
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    return (
        RunResult(
            lambda_value=lambda_value,
            validation_accuracy=best_validation_accuracy,
            test_accuracy=test_metrics["accuracy"],
            sparsity_percent=test_metrics["sparsity_percent"],
            checkpoint_path=str(checkpoint_path),
        ),
        model,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a self-pruning neural network on CIFAR-10.")
    parser.add_argument("--data-dir", default="data", help="Dataset root directory.")
    parser.add_argument("--output-dir", default="artifacts", help="Artifact output directory.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per lambda.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=8e-4, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay.")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Cross entropy label smoothing.")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[2048, 1024, 512], help="Hidden layer sizes.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability for hidden layers.")
    parser.add_argument("--disable-batchnorm", action="store_true", help="Disable batch normalization in hidden layers.")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2], help="Lambda values to compare.")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loader worker count.")
    parser.add_argument("--train-subset", type=int, default=None, help="Optional train subset size for quick experiments.")
    parser.add_argument("--test-subset", type=int, default=None, help="Optional test subset size for quick experiments.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    train_loader, validation_loader, test_loader = build_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        seed=args.seed,
    )

    results: list[RunResult] = []
    best_model: SelfPruningMLP | None = None
    best_result: RunResult | None = None

    for lambda_value in args.lambdas:
        result, model = train_single_lambda(
            lambda_value=lambda_value,
            train_loader=train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            use_batchnorm=not args.disable_batchnorm,
            output_dir=output_dir,
        )
        results.append(result)

        if best_result is None or result.validation_accuracy > best_result.validation_accuracy:
            best_result = result
            best_model = model

    assert best_result is not None and best_model is not None

    results_frame = write_results_table(
        [result.to_dict() for result in results],
        reports_dir / "results.csv",
    )
    write_markdown_report(
        results_frame=results_frame,
        best_lambda=best_result.lambda_value,
        best_accuracy=best_result.test_accuracy,
        best_sparsity=best_result.sparsity_percent,
        destination=reports_dir / "results.md",
    )
    plot_gate_distribution(
        gate_values=best_model.all_gate_values().detach().cpu().numpy(),
        destination=reports_dir / "gate_distribution.png",
        title=f"Gate Distribution for lambda={best_result.lambda_value}",
    )
    (reports_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_lambda": best_result.lambda_value,
                "best_validation_accuracy": best_result.validation_accuracy,
                "best_test_accuracy": best_result.test_accuracy,
                "best_sparsity_percent": best_result.sparsity_percent,
                "best_checkpoint": best_result.checkpoint_path,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0
