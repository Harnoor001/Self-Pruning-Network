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
for directory in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from benchmark import choose_device, load_model
from run_structured_pruning import save_structured_checkpoint
from self_pruning_network.data import build_cifar10_loaders
from self_pruning_network.model import StructuredPrunedMLP, trainable_parameter_count
from self_pruning_network.train import evaluate, reset_loader_seed, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate compact structured pruning across source-checkpoint seeds.")
    parser.add_argument(
        "--source-checkpoints", nargs="+", default=[
            "42=artifacts/final_benchmark/checkpoints/soft_lambda_0.0000.pt",
            "123=artifacts/ablation_key_benchmark/seed123/checkpoints/soft_lambda_0.0000.pt",
            "2024=artifacts/ablation_key_benchmark/seed2024/checkpoints/soft_lambda_0.0000.pt",
        ], help="Seed=soft-checkpoint path entries.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/structured_multiseed")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--targets", type=float, nargs="+", default=[40.0, 60.0])
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def parse_sources(entries: list[str]) -> list[tuple[int, Path]]:
    sources = []
    for entry in entries:
        seed_text, path_text = entry.split("=", 1)
        path = Path(path_text)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"source checkpoint not found: {path}")
        sources.append((int(seed_text), path.resolve()))
    return sources


def metric_from_checkpoint(path: Path, key: str) -> float | None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    value = payload.get("metrics", {}).get(key)
    return None if value is None else float(value) * 100.0


def run_one(seed: int, source_path: Path, args: argparse.Namespace, device: torch.device) -> list[dict[str, object]]:
    set_seed(seed)
    source, _ = load_model(source_path, device)
    loaders = build_cifar10_loaders(args.data_dir, args.batch_size, args.num_workers, seed=seed)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    source_accuracy = metric_from_checkpoint(source_path, "accuracy")
    dense_path = source_path.parent / "dense_lambda_0.0000.pt"
    dense_accuracy = metric_from_checkpoint(dense_path, "accuracy") if dense_path.exists() else None
    rows: list[dict[str, object]] = []
    for target in args.targets:
        set_seed(seed)
        reset_loader_seed(loaders[0], seed)
        model = StructuredPrunedMLP.from_self_pruning(source, target / 100.0).to(device)
        model.eval()
        before = float(evaluate(model, loaders[2], device, criterion)["accuracy"])
        optimizer = AdamW(model.parameters(), lr=args.fine_tune_learning_rate, weight_decay=args.weight_decay)
        architecture = [model.input_dim, *model.hidden_dims, model.num_classes]
        initial_parameter_count = trainable_parameter_count(model)
        for _ in range(args.fine_tune_epochs):
            model.train()
            for inputs, labels in loaders[0]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                criterion(model(inputs), labels).backward()
                optimizer.step()
            model.eval()
        result = evaluate(model, loaders[2], device, criterion)
        validation = evaluate(model, loaders[1], device, criterion)
        summary = model.efficiency_summary()
        if [model.input_dim, *model.hidden_dims, model.num_classes] != architecture:
            raise AssertionError("fine-tuning changed compact architecture")
        if trainable_parameter_count(model) != initial_parameter_count:
            raise AssertionError("fine-tuning changed compact parameter count")
        checkpoint = args.output_dir / "checkpoints" / f"structured_seed{seed}_target{int(target):02d}_ft{args.fine_tune_epochs}.pt"
        row = {
            "seed": seed, "method": "structured_learned", "target_sparsity_percent": target,
            "actual_structural_pruning_percent": float(target), "fine_tune_epochs": args.fine_tune_epochs,
            "architecture": " -> ".join(map(str, architecture)),
            "validation_accuracy_percent": float(validation["accuracy"]) * 100.0,
            "test_accuracy_percent": float(result["accuracy"]) * 100.0,
            "before_finetune_test_accuracy_percent": before * 100.0,
            "fine_tune_recovery_percent_points": (float(result["accuracy"]) - before) * 100.0,
            "soft_source_accuracy_percent": source_accuracy,
            "dense_reference_accuracy_percent": dense_accuracy,
            "accuracy_drop_vs_soft_percent_points": None if source_accuracy is None else source_accuracy - float(result["accuracy"]) * 100.0,
            "accuracy_drop_vs_dense_percent_points": None if dense_accuracy is None else dense_accuracy - float(result["accuracy"]) * 100.0,
            "parameters": trainable_parameter_count(model), "parameter_reduction_percent": float(summary["parameter_reduction_percent"]),
            "effective_macs": int(summary["estimated_effective_macs"]), "mac_reduction_percent": float(summary["mac_reduction_percent"]),
            "checkpoint": str(checkpoint.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"),
        }
        save_structured_checkpoint(checkpoint, model, {"test_accuracy": float(result["accuracy"]), **row},
                                   {"dataset": "CIFAR-10", "seed": seed, "source_checkpoint": str(source_path),
                                    "target_neuron_sparsity_percent": target, "fine_tune_epochs": args.fine_tune_epochs,
                                    "fine_tune_learning_rate": args.fine_tune_learning_rate, "batch_size": args.batch_size})
        rows.append(row)
    return rows


def write_reports(frame: pd.DataFrame, metadata: dict[str, object], output_dir: Path) -> None:
    reports, plots = output_dir / "reports", output_dir / "plots"
    reports.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    frame.to_csv(reports / "multi_seed_results.csv", index=False)
    grouped = frame.groupby(["method", "target_sparsity_percent"], as_index=False).agg(
        mean_accuracy_percent=("test_accuracy_percent", "mean"), std_accuracy_percent=("test_accuracy_percent", "std"),
        mean_accuracy_drop_vs_dense_percent_points=("accuracy_drop_vs_dense_percent_points", "mean"),
        std_accuracy_drop_vs_dense_percent_points=("accuracy_drop_vs_dense_percent_points", "std"),
    )
    grouped.to_csv(reports / "multi_seed_summary.csv", index=False)
    text = "# Structured Pruning Multi-Seed Validation\n\n" + f"```json\n{json.dumps(metadata, indent=2)}\n```\n\n"
    text += "## Per-seed results\n\n" + frame.to_markdown(index=False) + "\n\n## Mean +/- sample standard deviation\n\n" + grouped.to_markdown(index=False) + "\n"
    (reports / "multi_seed_summary.md").write_text(text, encoding="utf-8")
    summary = {"experiment": "structured_pruning_multiseed", "metadata": metadata,
               "results": json.loads(frame.to_json(orient="records")), "grouped": json.loads(grouped.to_json(orient="records"))}
    (reports / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plt.figure(figsize=(8, 5))
    for target, group in grouped.groupby("target_sparsity_percent"):
        plt.errorbar([target], group["mean_accuracy_percent"], yerr=group["std_accuracy_percent"], fmt="o", capsize=4, label=f"Structured {target:.0f}%")
    plt.xlabel("Target structural pruning (%)"); plt.ylabel("Test accuracy (%)"); plt.title("Multi-Seed Structured-Pruning Accuracy")
    plt.grid(alpha=0.25); plt.legend(); plt.tight_layout(); plt.savefig(plots / "multi_seed_accuracy.png", dpi=160); plt.close()


def main() -> int:
    args = parse_args()
    if args.fine_tune_epochs < 0 or any(target < 0 or target >= 100 for target in args.targets):
        raise ValueError("fine-tune epochs must be non-negative and targets must be in [0, 100)")
    sources = parse_sources(args.source_checkpoints)
    device = choose_device(args.device)
    rows = []
    for seed, path in sources:
        rows.extend(run_one(seed, path, args, device))
    metadata = {"dataset": "CIFAR-10", "seeds": [seed for seed, _ in sources], "targets_percent": args.targets,
                "fine_tune_epochs": args.fine_tune_epochs, "batch_size": args.batch_size,
                "fine_tune_learning_rate": args.fine_tune_learning_rate, "weight_decay": args.weight_decay,
                "label_smoothing": args.label_smoothing, "source_checkpoints": {str(seed): str(path) for seed, path in sources}}
    frame = pd.DataFrame(rows).sort_values(["target_sparsity_percent", "seed"])
    write_reports(frame, metadata, args.output_dir.resolve())
    print(frame.to_string(index=False))
    print(f"\nWrote multi-seed reports to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
