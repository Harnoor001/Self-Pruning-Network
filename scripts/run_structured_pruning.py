from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for directory in (SRC_DIR, SCRIPTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from benchmark import choose_device, load_model, measure_latency
from self_pruning_network.data import build_cifar10_loaders
from self_pruning_network.model import (
    SelfPruningMLP,
    StructuredPrunedMLP,
    deployable_parameter_count,
    trainable_parameter_count,
)
from self_pruning_network.train import evaluate, fine_tune, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and benchmark physically compact neuron-pruned MLPs.")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/final_benchmark/checkpoints/soft_lambda_0.0000.pt")
    parser.add_argument("--dense-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/final_benchmark/checkpoints/dense_lambda_0.0000.pt")
    parser.add_argument("--unstructured-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/final_benchmark/checkpoints/hard_target_0.6000.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/structured_pruning")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--targets", type=float, nargs="+", default=[20.0, 40.0, 60.0])
    parser.add_argument("--fine-tune-epochs", type=int, default=1)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def architecture(model: torch.nn.Module) -> list[int]:
    if isinstance(model, StructuredPrunedMLP):
        return [model.input_dim, *model.hidden_dims, model.num_classes]
    return [model.input_dim, *model.hidden_dims, model.num_classes]


def save_structured_checkpoint(path: Path, model: StructuredPrunedMLP, metrics: dict[str, object], config: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 4,
            "model_type": "structured",
            "model_config": {
                "model_type": "structured",
                "input_dim": model.input_dim,
                "hidden_dims": model.hidden_dims,
                "num_classes": model.num_classes,
                "dropout": model.dropout,
                "use_batchnorm": model.use_batchnorm,
            },
            "structured_pruning": model.structured_summary(),
            "experiment_config": config,
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def input_tensors(batch_sizes: list[int], seed: int, device: torch.device) -> dict[int, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return {
        size: torch.randn(size, 3, 32, 32, generator=generator, dtype=torch.float32).to(device)
        for size in batch_sizes
    }


def accuracy_source(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    dense = frame[frame["model_type"] == "dense"].iloc[0]
    hard = frame[(frame["model_type"] == "hard") & np.isclose(frame["target_sparsity_percent"], 60.0)].iloc[0]
    soft = frame[(frame["model_type"] == "soft") & (frame["lambda"] == 0.0)].iloc[0]
    return {
        "dense": float(dense["test_accuracy"]) * 100.0,
        "unstructured": float(hard["test_accuracy"]) * 100.0,
        "soft": float(soft["test_accuracy"]) * 100.0,
        "unstructured_drop": float(hard["accuracy_drop_percent_points"]),
    }


def make_row(
    label: str,
    model: torch.nn.Module,
    checkpoint: Path,
    accuracy: float,
    accuracy_drop: float,
    before_accuracy: float | None,
    validation_accuracy: float | None,
    source_accuracy: float,
    latency: dict[int, dict[str, float]],
    target: float,
    source_architecture: list[int],
) -> dict[str, object]:
    if isinstance(model, StructuredPrunedMLP):
        efficiency = model.efficiency_summary()
        parameter_count = trainable_parameter_count(model)
        deployment_count = deployable_parameter_count(model)
        model_type = "structured"
        logical_sparsity = 0.0
    else:
        efficiency = model.efficiency_summary()
        parameter_count = trainable_parameter_count(model)
        deployment_count = deployable_parameter_count(model)
        model_type = "unstructured" if label.startswith("Unstructured") else "dense"
        logical_sparsity = float(efficiency["sparsity_percent"])
    dense_deployment_count = int(deployable_parameter_count(model)) if model_type == "dense" else None
    row: dict[str, object] = {
        "model": label,
        "model_type": model_type,
        "target_neuron_sparsity_percent": target if model_type == "structured" else None,
        "architecture": " -> ".join(str(value) for value in architecture(model)),
        "source_architecture": " -> ".join(str(value) for value in source_architecture),
        "test_accuracy_percent": accuracy,
        "validation_accuracy_percent": validation_accuracy,
        "accuracy_drop_percent_points": accuracy_drop,
        "pre_finetune_test_accuracy_percent": before_accuracy,
        "fine_tune_recovery_percent_points": (accuracy - before_accuracy) if before_accuracy is not None else None,
        "logical_sparsity_percent": logical_sparsity,
        "total_connections": int(efficiency["total_weights"]),
        "active_connections": int(efficiency["active_connections"]),
        "pruned_connections": int(efficiency["pruned_connections"]),
        "trainable_parameters": parameter_count,
        "deployable_parameters": deployment_count,
        "parameter_reduction_percent": None,
        "estimated_dense_macs": int(efficiency["estimated_dense_macs"]),
        "effective_macs": int(efficiency["estimated_effective_macs"]),
        "mac_reduction_percent": None,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_size_mb": checkpoint.stat().st_size / 1_000_000,
        "checkpoint_size_change_vs_dense_percent": None,
        "checkpoint": relative(checkpoint),
    }
    if model_type == "structured":
        row["parameter_reduction_percent"] = None
        row["mac_reduction_percent"] = None
    for batch_size, values in latency.items():
        for metric, value in values.items():
            row[f"batch_{batch_size}_{metric}"] = value
    row["source_test_accuracy_percent"] = source_accuracy
    return row


def write_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    structured = frame[frame["model_type"] == "structured"].sort_values("target_neuron_sparsity_percent")
    comparisons = frame[frame["model"].isin(["Dense", "Unstructured 60%"])]

    def plot(x: str, y: str, filename: str, title: str, xlabel: str, ylabel: str, include_comparison: bool = True) -> None:
        plt.figure(figsize=(8, 5))
        if include_comparison:
            plt.scatter(comparisons[x], comparisons[y], color="#555555", label="Existing baselines", zorder=3)
        if not structured.empty:
            plt.plot(structured[x], structured[y], marker="o", linewidth=2, color="#1768ac", label="Structured")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots / filename, dpi=160)
        plt.close()

    plot("parameter_reduction_percent", "test_accuracy_percent", "accuracy_vs_parameter_reduction.png",
         "Accuracy vs Physical Parameter Reduction", "Parameter reduction (%)", "Test accuracy (%)")
    plot("mac_reduction_percent", "test_accuracy_percent", "accuracy_vs_mac_reduction.png",
         "Accuracy vs Structured MAC Reduction", "MAC reduction (%)", "Test accuracy (%)")
    plot("parameter_reduction_percent", "batch_1_mean_ms", "accuracy_vs_latency.png",
         "Batch-1 Latency vs Physical Parameter Reduction", "Parameter reduction (%)", "Mean latency (ms)")
    plot("parameter_reduction_percent", "checkpoint_size_mb", "checkpoint_size_vs_parameter_reduction.png",
         "Checkpoint Size vs Physical Parameter Reduction", "Parameter reduction (%)", "Checkpoint size (MB)")


def write_reports(frame: pd.DataFrame, layers: pd.DataFrame, metadata: dict[str, object], output_dir: Path) -> None:
    reports = output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    frame.to_csv(reports / "structured_results.csv", index=False)
    layers.to_csv(reports / "layerwise_neuron_pruning.csv", index=False)
    recovery_columns = [
        "model", "target_neuron_sparsity_percent", "pre_finetune_test_accuracy_percent",
        "test_accuracy_percent", "fine_tune_recovery_percent_points",
    ]
    frame[recovery_columns].to_csv(reports / "fine_tuning_recovery.csv", index=False)
    view = frame.copy()
    for column in ["test_accuracy_percent", "validation_accuracy_percent", "accuracy_drop_percent_points",
                   "parameter_reduction_percent", "mac_reduction_percent", "checkpoint_size_mb",
                   "batch_1_mean_ms", "batch_1_p50_ms", "batch_1_p95_ms", "batch_32_mean_ms",
                   "batch_32_p50_ms", "batch_32_p95_ms"]:
        if column in view:
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    columns = ["model", "architecture", "target_neuron_sparsity_percent", "test_accuracy_percent",
               "accuracy_drop_percent_points", "trainable_parameters", "deployable_parameters",
               "parameter_reduction_percent", "effective_macs", "mac_reduction_percent",
               "checkpoint_size_mb", "batch_1_mean_ms", "batch_32_mean_ms"]
    text = (
        "# Structured Pruning Benchmark\n\n"
        "Structured models physically remove hidden neurons and transfer surviving rows, columns, biases, and BatchNorm state. Parameter and MAC reductions below are computed from the instantiated compact models.\n\n"
        "## Environment and protocol\n\n"
        f"```json\n{json.dumps(metadata, indent=2)}\n```\n\n"
        "## Results\n\n" + view[columns].to_markdown(index=False) + "\n\n"
        "## Layer-wise neuron pruning\n\n" + layers.to_markdown(index=False) + "\n\n"
        "## Interpretation\n\n"
        "The Dense and Unstructured 60% rows are existing validated checkpoints. Unstructured MAC reduction is an ideal masked estimate, while Structured MAC reduction is calculated from physically smaller dense matrices. Latency and checkpoint sizes are measured in this run.\n"
    )
    (reports / "structured_results.md").write_text(text, encoding="utf-8")
    summary = {"experiment": "structured_neuron_pruning", "metadata": metadata,
               "results": json.loads(frame.to_json(orient="records")),
               "layerwise": json.loads(layers.to_json(orient="records")),
               "report_files": sorted(path.name for path in reports.iterdir() if path.is_file())}
    (reports / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_plots(frame, output_dir)


def main() -> int:
    args = parse_args()
    if any(target < 0.0 or target >= 100.0 for target in args.targets):
        raise ValueError("structured targets must be in [0, 100), leaving at least one hidden neuron")
    if args.fine_tune_epochs < 0 or args.iterations <= 0 or args.warmup < 0:
        raise ValueError("fine-tune epochs and warmup must be non-negative; iterations must be positive")
    set_seed(args.seed)
    device = choose_device(args.device)
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    for path in (args.checkpoint, args.dense_checkpoint, args.unstructured_checkpoint):
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")

    source, source_payload = load_model(args.checkpoint.resolve(), device)
    dense, _ = load_model(args.dense_checkpoint.resolve(), device)
    unstructured, _ = load_model(args.unstructured_checkpoint.resolve(), device)
    source_architecture = architecture(source)
    accuracy_report = args.checkpoint.parent.parent / "reports" / "results.csv"
    accuracies = accuracy_source(accuracy_report)
    loaders = build_cifar10_loaders(args.data_dir, args.batch_size, args.num_workers, seed=args.seed)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    inputs = input_tensors(args.batch_sizes, args.seed, device)
    output_dir = args.output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"
    config = {
        "dataset": "CIFAR-10", "seed": args.seed, "source_checkpoint": relative(args.checkpoint),
        "architecture": source_architecture, "targets_percent": args.targets,
        "fine_tune_epochs": args.fine_tune_epochs, "fine_tune_learning_rate": args.fine_tune_learning_rate,
        "batch_size": args.batch_size, "batch_sizes": args.batch_sizes,
        "warmup_iterations": args.warmup, "measurement_iterations": args.iterations,
        "device": str(device), "torch_version": torch.__version__, "python_version": platform.python_version(),
        "neuron_importance": "mean sigmoid gate value across each output row",
        "parameter_definition": "all trainable compact weights, biases, and BatchNorm parameters; gate scores excluded from deployment",
    }

    models: list[tuple[str, torch.nn.Module, Path, float, float, float | None, float | None, float]] = []
    dense_latency = {size: measure_latency(dense, inputs[size], args.warmup, args.iterations, device) for size in args.batch_sizes}
    unstructured_latency = {size: measure_latency(unstructured, inputs[size], args.warmup, args.iterations, device) for size in args.batch_sizes}
    models.append(("Dense", dense, args.dense_checkpoint.resolve(), accuracies["dense"], 0.0, None, None, accuracies["dense"]))
    models.append(("Unstructured 60%", unstructured, args.unstructured_checkpoint.resolve(), accuracies["unstructured"], accuracies["unstructured_drop"], None, None, accuracies["soft"]))
    rows: list[dict[str, object]] = []
    for label, model, checkpoint, accuracy, drop, before, validation, reference in models:
        latency = dense_latency if label == "Dense" else unstructured_latency
        rows.append(make_row(label, model, checkpoint, accuracy, drop, before, validation, reference, latency, 0.0, source_architecture))

    layer_rows: list[dict[str, object]] = []
    for target_percent in args.targets:
        target = target_percent / 100.0
        compact = StructuredPrunedMLP.from_self_pruning(source, target).to(device)
        compact.eval()
        before_metrics = evaluate(compact, loaders[2], device, criterion)
        before_accuracy = float(before_metrics["accuracy"])
        before_validation = float(evaluate(compact, loaders[1], device, criterion)["accuracy"])
        _, after_accuracy, _, _ = fine_tune(
            compact, loaders[0], loaders[1], loaders[2], device, args.fine_tune_epochs,
            args.fine_tune_learning_rate, args.weight_decay, args.label_smoothing, 0.0,
        )
        after_validation = float(evaluate(compact, loaders[1], device, criterion)["accuracy"])
        checkpoint = checkpoint_dir / f"structured_target_{int(target_percent):02d}.pt"
        efficiency = compact.efficiency_summary()
        metrics = {"before_finetune_test_accuracy": before_accuracy, "test_accuracy": after_accuracy,
                   "validation_accuracy": after_validation, **efficiency}
        save_structured_checkpoint(checkpoint, compact, metrics, {**config, "target_neuron_sparsity_percent": target_percent})
        latency = {size: measure_latency(compact, inputs[size], args.warmup, args.iterations, device) for size in args.batch_sizes}
        row = make_row(f"Structured {int(target_percent)}%", compact, checkpoint, after_accuracy * 100.0,
                       (accuracies["soft"] - after_accuracy * 100.0), before_accuracy * 100.0,
                       after_validation * 100.0, accuracies["soft"], latency, target_percent, source_architecture)
        rows.append(row)
        for layer_index, layer in enumerate(compact.linear_layers):
            original = source.hidden_dims[layer_index] if layer_index < len(source.hidden_dims) else source.num_classes
            remaining = layer.out_features
            if layer_index == len(compact.linear_layers) - 1:
                original = source.num_classes
            layer_rows.append({"model": f"Structured {int(target_percent)}%", "layer": layer_index + 1,
                               "layer_type": "hidden" if layer_index < len(source.hidden_dims) else "output",
                               "original_neurons": original, "remaining_neurons": remaining,
                               "pruned_neurons": original - remaining,
                               "pruning_percent": (original - remaining) / original * 100.0})

    frame = pd.DataFrame(rows)
    dense_parameters = int(frame.loc[frame["model"] == "Dense", "deployable_parameters"].iloc[0])
    dense_macs = int(frame.loc[frame["model"] == "Dense", "effective_macs"].iloc[0])
    dense_size = int(frame.loc[frame["model"] == "Dense", "checkpoint_size_bytes"].iloc[0])
    frame["parameter_reduction_percent"] = (dense_parameters - frame["deployable_parameters"]) / dense_parameters * 100.0
    frame["mac_reduction_percent"] = (dense_macs - frame["effective_macs"]) / dense_macs * 100.0
    frame["checkpoint_size_change_vs_dense_percent"] = (dense_size - frame["checkpoint_size_bytes"]) / dense_size * 100.0
    structured_layers = pd.DataFrame(layer_rows)
    metadata = {**config, "benchmark_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dense_checkpoint": relative(args.dense_checkpoint),
                "unstructured_checkpoint": relative(args.unstructured_checkpoint),
                "source_soft_test_accuracy_percent": accuracies["soft"],
                "source_payload_format_version": source_payload.get("format_version")}
    write_reports(frame, structured_layers, metadata, output_dir)
    print(frame.to_string(index=False))
    print(f"\nWrote structured-pruning reports to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
