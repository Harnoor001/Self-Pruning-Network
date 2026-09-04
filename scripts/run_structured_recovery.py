from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for directory in (SRC_DIR, SCRIPTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from benchmark import choose_device, load_model
from run_structured_pruning import save_structured_checkpoint
from self_pruning_network.data import build_cifar10_loaders
from self_pruning_network.model import StructuredPrunedMLP, trainable_parameter_count
from self_pruning_network.train import accuracy_from_logits, evaluate, reset_loader_seed, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure structured-pruning accuracy recovery versus fine-tuning budget.")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/final_benchmark/checkpoints/soft_lambda_0.0000.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/structured_recovery")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--targets", type=float, nargs="+", default=[20.0, 40.0, 60.0])
    parser.add_argument("--fine-tune-budgets", type=int, nargs="+", default=[0, 1, 3, 5])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def evaluate_row(model: StructuredPrunedMLP, loaders, device: torch.device, criterion: nn.Module,
                 target: float, budget: int, before_accuracy: float, source_accuracy: float,
                 dense_accuracy: float, architecture: list[int]) -> dict[str, object]:
    validation = float(evaluate(model, loaders[1], device, criterion)["accuracy"])
    test = float(evaluate(model, loaders[2], device, criterion)["accuracy"])
    summary = model.efficiency_summary()
    current_architecture = [model.input_dim, *model.hidden_dims, model.num_classes]
    if current_architecture != architecture:
        raise AssertionError("fine-tuning changed the compact architecture")
    parameter_count = trainable_parameter_count(model)
    if parameter_count != sum(parameter.numel() for parameter in model.parameters()):
        raise AssertionError("reported parameter count does not match model tensors")
    return {
        "seed": 42,
        "target_neuron_sparsity_percent": target,
        "fine_tune_epochs": budget,
        "architecture": " -> ".join(str(value) for value in architecture),
        "validation_accuracy_percent": validation * 100.0,
        "test_accuracy_percent": test * 100.0,
        "before_finetune_test_accuracy_percent": before_accuracy * 100.0,
        "fine_tune_recovery_percent_points": (test - before_accuracy) * 100.0,
        "accuracy_drop_vs_soft_percent_points": source_accuracy - test * 100.0,
        "accuracy_drop_vs_dense_percent_points": dense_accuracy - test * 100.0,
        "trainable_parameters": parameter_count,
        "deployable_parameters": parameter_count,
        "effective_macs": int(summary["estimated_effective_macs"]),
        "parameter_reduction_percent": float(summary["parameter_reduction_percent"]),
        "mac_reduction_percent": float(summary["mac_reduction_percent"]),
        "checkpoint": None,
    }


def write_reports(frame: pd.DataFrame, metadata: dict[str, object], output_dir: Path) -> None:
    reports = output_dir / "reports"
    plots = output_dir / "plots"
    reports.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    frame.to_csv(reports / "recovery_results.csv", index=False)
    frame.to_csv(reports / "fine_tuning_recovery.csv", index=False)
    view = frame.copy()
    for column in ["test_accuracy_percent", "fine_tune_recovery_percent_points", "accuracy_drop_vs_soft_percent_points",
                   "accuracy_drop_vs_dense_percent_points", "parameter_reduction_percent", "mac_reduction_percent"]:
        view[column] = view[column].map(lambda value: f"{value:.6f}")
    columns = ["target_neuron_sparsity_percent", "fine_tune_epochs", "architecture", "validation_accuracy_percent",
               "test_accuracy_percent", "before_finetune_test_accuracy_percent", "fine_tune_recovery_percent_points",
               "accuracy_drop_vs_soft_percent_points", "accuracy_drop_vs_dense_percent_points",
               "trainable_parameters", "parameter_reduction_percent", "mac_reduction_percent"]
    text = (
        "# Structured Pruning Fine-Tuning Recovery\n\n"
        "Every budget starts from the same transferred compact model for its target. The optimizer, data, seed, augmentation, and evaluation protocol are unchanged; only the number of fixed-architecture fine-tuning epochs varies.\n\n"
        f"```json\n{json.dumps(metadata, indent=2)}\n```\n\n"
        "## Results\n\n" + view[columns].to_markdown(index=False) + "\n"
    )
    (reports / "recovery_results.md").write_text(text, encoding="utf-8")
    summary = {"experiment": "structured_finetuning_recovery", "metadata": metadata,
               "results": json.loads(frame.to_json(orient="records"))}
    (reports / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(8, 5))
    for target, group in frame.groupby("target_neuron_sparsity_percent"):
        group = group.sort_values("fine_tune_epochs")
        plt.plot(group["fine_tune_epochs"], group["test_accuracy_percent"], marker="o", label=f"Structured {target:.0f}%")
    plt.xlabel("Fine-tuning epochs")
    plt.ylabel("Test accuracy (%)")
    plt.title("Fine-Tuning Epochs vs Structured-Model Accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "fine_tuning_epochs_vs_accuracy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for target, group in frame.groupby("target_neuron_sparsity_percent"):
        group = group.sort_values("fine_tune_epochs")
        plt.plot(group["fine_tune_epochs"], group["fine_tune_recovery_percent_points"], marker="o", label=f"Structured {target:.0f}%")
    plt.xlabel("Fine-tuning epochs")
    plt.ylabel("Accuracy recovery (percentage points)")
    plt.title("Fine-Tuning Epochs vs Accuracy Recovery")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "fine_tuning_epochs_vs_recovery.png", dpi=160)
    plt.close()


def main() -> int:
    args = parse_args()
    if any(target < 0.0 or target >= 100.0 for target in args.targets):
        raise ValueError("targets must be in [0, 100)")
    if any(budget < 0 for budget in args.fine_tune_budgets):
        raise ValueError("fine-tuning budgets must be non-negative")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    set_seed(args.seed)
    device = choose_device(args.device)
    source, _ = load_model(args.checkpoint.resolve(), device)
    loaders = build_cifar10_loaders(args.data_dir, args.batch_size, args.num_workers, seed=args.seed)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    output_dir = args.output_dir.resolve()
    max_budget = max(args.fine_tune_budgets)
    accuracy_report = args.checkpoint.parent.parent / "reports" / "results.csv"
    accuracy_frame = pd.read_csv(accuracy_report)
    source_accuracy = float(accuracy_frame[
        (accuracy_frame["model_type"] == "soft") & (accuracy_frame["lambda"] == 0.0)
    ].iloc[0]["test_accuracy"]) * 100.0
    dense_accuracy = float(accuracy_frame[accuracy_frame["model_type"] == "dense"].iloc[0]["test_accuracy"]) * 100.0
    rows: list[dict[str, object]] = []
    for target_percent in args.targets:
        set_seed(args.seed)
        reset_loader_seed(loaders[0], args.seed)
        model = StructuredPrunedMLP.from_self_pruning(source, target_percent / 100.0).to(device)
        model.eval()
        architecture = [model.input_dim, *model.hidden_dims, model.num_classes]
        before_accuracy = float(evaluate(model, loaders[2], device, criterion)["accuracy"])
        optimizer = AdamW(model.parameters(), lr=args.fine_tune_learning_rate, weight_decay=args.weight_decay)
        for epoch in range(max_budget + 1):
            if epoch > 0:
                model.train()
                for inputs, targets in loaders[0]:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), targets)
                    loss.backward()
                    optimizer.step()
            if epoch in args.fine_tune_budgets:
                model.eval()
                row = evaluate_row(model, loaders, device, criterion, target_percent, epoch,
                                   before_accuracy, source_accuracy, dense_accuracy, architecture)
                checkpoint = output_dir / "checkpoints" / f"structured_target_{int(target_percent):02d}_ft{epoch}.pt"
                save_structured_checkpoint(checkpoint, model,
                                           {"test_accuracy": row["test_accuracy_percent"] / 100.0, **row},
                                           {"dataset": "CIFAR-10", "seed": args.seed,
                                            "source_checkpoint": str(args.checkpoint),
                                            "target_neuron_sparsity_percent": target_percent,
                                            "fine_tune_epochs": epoch,
                                            "fine_tune_learning_rate": args.fine_tune_learning_rate,
                                            "batch_size": args.batch_size})
                row["checkpoint"] = str(checkpoint.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
                rows.append(row)
    metadata = {
        "dataset": "CIFAR-10", "seed": args.seed, "train_samples": len(loaders[0].dataset),
        "validation_samples": len(loaders[1].dataset), "test_samples": len(loaders[2].dataset),
        "source_checkpoint": str(args.checkpoint.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_architecture": [source.input_dim, *source.hidden_dims, source.num_classes],
        "targets_percent": args.targets, "fine_tune_budgets": args.fine_tune_budgets,
        "optimizer": "AdamW", "fine_tune_learning_rate": args.fine_tune_learning_rate,
        "weight_decay": args.weight_decay, "label_smoothing": args.label_smoothing,
        "importance": "mean sigmoid gate value across outgoing rows",
    }
    frame = pd.DataFrame(rows).sort_values(["target_neuron_sparsity_percent", "fine_tune_epochs"])
    write_reports(frame, metadata, output_dir)
    print(frame.to_string(index=False))
    print(f"\nWrote recovery reports to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
