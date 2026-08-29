from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from self_pruning_network.model import SelfPruningMLP


TARGETS = [
    ("Dense", "dense_lambda_0.0000.pt", "dense", None),
    ("Soft", "soft_lambda_0.0000.pt", "soft", None),
    ("Hard 20%", "hard_target_0.2000.pt", "hard", 20.0),
    ("Hard 40%", "hard_target_0.4000.pt", "hard", 40.0),
    ("Hard 60%", "hard_target_0.6000.pt", "hard", 60.0),
    ("Hard 80%", "hard_target_0.8000.pt", "hard", 80.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark validated dense and pruned checkpoints.")
    parser.add_argument("--benchmark-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "final_benchmark")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "efficiency_benchmark")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=None)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[SelfPruningMLP, dict[str, object]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["model_config"]
    model = SelfPruningMLP(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"],
        num_classes=config["num_classes"],
        dropout=config.get("dropout", 0.3),
        use_batchnorm=config.get("use_batchnorm", True),
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    mode = payload.get("pruning", {}).get("mode", "soft")
    model.set_mode(mode if mode in {"dense", "soft", "hard"} else "soft")
    model.to(device)
    model.eval()
    return model, payload


def accuracy_metadata(report_path: Path) -> dict[str, dict[str, float | None]]:
    frame = pd.read_csv(report_path)
    metadata: dict[str, dict[str, float | None]] = {}
    for label, _, model_type, target in TARGETS:
        if model_type == "dense":
            matches = frame[frame["model_type"] == "dense"]
        elif model_type == "soft":
            matches = frame[(frame["model_type"] == "soft") & (frame["lambda"].fillna(-1) == 0.0)]
        else:
            matches = frame[
                (frame["model_type"] == "hard")
                & np.isclose(frame["target_sparsity_percent"].astype(float), target)
            ]
        if matches.empty:
            raise RuntimeError(f"No accuracy record found for {label} in {report_path}.")
        row = matches.iloc[0]
        metadata[label] = {
            "test_accuracy_percent": float(row["test_accuracy"]) * 100.0,
            "validation_accuracy_percent": float(row["validation_accuracy"]) * 100.0,
            "accuracy_drop_percent_points": float(row["accuracy_drop_percent_points"]),
        }
    return metadata


def make_inputs(batch_sizes: list[int], seed: int, device: torch.device) -> dict[int, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return {
        batch_size: torch.randn(batch_size, 3, 32, 32, generator=generator, dtype=torch.float32).to(device)
        for batch_size in batch_sizes
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_latency(
    model: SelfPruningMLP,
    inputs: torch.Tensor,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        synchronize(device)
        samples: list[float] = []
        for _ in range(iterations):
            synchronize(device)
            start = time.perf_counter_ns()
            model(inputs)
            synchronize(device)
            samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return {
        "mean_ms": float(np.mean(samples)),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "min_ms": float(np.min(samples)),
        "max_ms": float(np.max(samples)),
    }


def plot_metric(frame: pd.DataFrame, column: str, path: Path, title: str, y_label: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(frame["sparsity_percent"], frame[column], marker="o", linewidth=2.2, color="#1768ac")
    plt.xlabel("Logical sparsity (%)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def pareto_rows(frame: pd.DataFrame, accuracy_column: str, latency_column: str | None = None) -> list[str]:
    labels: list[str] = []
    for index, candidate in frame.iterrows():
        competitors = frame.drop(index)
        if latency_column is None:
            dominated = (
                (competitors[accuracy_column] >= candidate[accuracy_column])
                & (competitors["sparsity_percent"] >= candidate["sparsity_percent"])
                & (
                    (competitors[accuracy_column] > candidate[accuracy_column])
                    | (competitors["sparsity_percent"] > candidate["sparsity_percent"])
                )
            ).any()
        else:
            dominated = (
                (competitors[accuracy_column] >= candidate[accuracy_column])
                & (competitors[latency_column] <= candidate[latency_column])
                & (
                    (competitors[accuracy_column] > candidate[accuracy_column])
                    | (competitors[latency_column] < candidate[latency_column])
                )
            ).any()
        if not dominated:
            labels.append(str(candidate["model"]))
    return labels


def write_markdown(
    frame: pd.DataFrame,
    layer_frame: pd.DataFrame,
    metadata: dict[str, object],
    path: Path,
) -> None:
    report_frame = frame.copy()
    report_frame["test_accuracy_percent"] = report_frame["test_accuracy_percent"].map(lambda value: f"{value:.4f}")
    report_frame["accuracy_drop_percent_points"] = report_frame["accuracy_drop_percent_points"].map(lambda value: f"{value:.4f}")
    report_frame["sparsity_percent"] = report_frame["sparsity_percent"].map(lambda value: f"{value:.6f}")
    report_frame["density_percent"] = report_frame["density_percent"].map(lambda value: f"{value:.6f}")
    report_frame["logical_connectivity_reduction_percent"] = report_frame["logical_connectivity_reduction_percent"].map(lambda value: f"{value:.6f}")
    report_frame["checkpoint_size_mb"] = report_frame["checkpoint_size_mb"].map(lambda value: f"{value:.6f}")
    report_frame["checkpoint_size_mib"] = report_frame["checkpoint_size_mib"].map(lambda value: f"{value:.6f}")
    for batch_size in metadata["batch_sizes"]:
        for suffix in ("mean_ms", "p50_ms", "p95_ms", "latency_reduction_vs_dense_percent"):
            column = f"batch_{batch_size}_{suffix}"
            report_frame[column] = report_frame[column].map(lambda value: f"{value:.6f}")

    columns = [
        "model", "lambda", "target_sparsity_percent", "sparsity_percent", "density_percent", "test_accuracy_percent",
        "accuracy_drop_percent_points", "active_connections", "pruned_connections",
        "estimated_effective_macs", "theoretical_mac_reduction_percent", "checkpoint_size_bytes", "checkpoint_size_mb", "checkpoint_size_mib",
    ]
    for batch_size in metadata["batch_sizes"]:
        columns.extend(
            [
                f"batch_{batch_size}_mean_ms",
                f"batch_{batch_size}_p50_ms",
                f"batch_{batch_size}_p95_ms",
                f"batch_{batch_size}_latency_reduction_vs_dense_percent",
            ]
        )
    table = report_frame[columns].to_markdown(index=False)
    layer_table = layer_frame.to_markdown(index=False)
    path.write_text(
        "# Efficiency and Compression Benchmark\n\n"
        "This report benchmarks the validated final checkpoints without retraining. Connectivity and MAC values are calculated from checkpoint masks; latency and checkpoint sizes are measured on this execution environment.\n\n"
        "## Environment and protocol\n\n"
        f"```json\n{json.dumps(metadata, indent=2)}\n```\n\n"
        "`estimated_effective_macs` and `theoretical_mac_reduction_percent` assume ideal sparse execution. The benchmark models still use dense PyTorch tensors with binary masks, so these values are not measured runtime speedups.\n\n"
        "## Results\n\n"
        f"{table}\n\n"
        "Latency reduction is measured relative to Dense at the same batch size; negative values mean the masked dense model was slower. Checkpoint size is the actual file size on disk, not a compressed sparse representation.\n\n"
        "## Accuracy/sparsity Pareto frontier\n\n"
        f"{', '.join(metadata['accuracy_sparsity_pareto_frontier']) or 'None'}\n\n"
        "## Latency/accuracy frontier (batch size 1)\n\n"
        f"{', '.join(metadata['latency_accuracy_pareto_frontier_batch_1']) or 'None'}\n\n"
        "## Layer-wise connectivity and MAC accounting\n\n"
        f"{layer_table}\n\n"
        "The complete unrounded layer metrics are also available in `layer_efficiency.csv`.\n\n"
        "## Interpretation\n\n"
        "Logical sparsity and ideal MAC reduction are not physical parameter removal. Dense tensor shapes remain allocated and ordinary dense kernels may not skip zero entries. Actual storage reduction or acceleration requires a sparse representation, structured pruning, sparse kernels, or hardware/runtime support.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    args.benchmark_dir = args.benchmark_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise ValueError("batch sizes must be positive")

    device = choose_device(args.device)
    if args.threads is not None:
        if args.threads <= 0:
            raise ValueError("threads must be positive")
        torch.set_num_threads(args.threads)

    checkpoint_dir = args.benchmark_dir / "checkpoints"
    report_csv = args.benchmark_dir / "reports" / "results.csv"
    if not report_csv.exists():
        raise FileNotFoundError(f"Validated accuracy report not found: {report_csv}")
    accuracy = accuracy_metadata(report_csv)
    inputs = make_inputs(args.batch_sizes, args.seed, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    reports_dir = args.output_dir / "reports"
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    loaded: list[tuple[str, SelfPruningMLP, dict[str, object]]] = []
    layer_rows: list[dict[str, object]] = []
    for label, filename, _, _ in TARGETS:
        checkpoint_path = checkpoint_dir / filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Required validated checkpoint not found: {checkpoint_path}")
        model, payload = load_model(checkpoint_path, device)
        summary = model.efficiency_summary()
        total = int(summary["total_weights"])
        active = int(summary["active_connections"])
        pruned = int(summary["pruned_connections"])
        if active + pruned != total:
            raise AssertionError(f"Connection accounting failed for {label}.")
        loaded.append((label, model, summary))
        experiment_config = payload.get("experiment_config", {})
        if not isinstance(experiment_config, dict):
            experiment_config = {}
        for layer in summary["layers"]:
            layer_rows.append({"model": label, **layer})

        checkpoint_bytes = checkpoint_path.stat().st_size
        row: dict[str, object] = {
            "model": label,
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "lambda": experiment_config.get("lambda"),
            "target_sparsity_percent": (
                float(experiment_config["target_sparsity"]) * 100.0
                if experiment_config.get("target_sparsity") is not None
                else 0.0
            ),
            "checkpoint_size_bytes": checkpoint_bytes,
            "checkpoint_size_mb": checkpoint_bytes / 1_000_000,
            "checkpoint_size_mib": checkpoint_bytes / (1024 * 1024),
            "validation_accuracy_percent": accuracy[label]["validation_accuracy_percent"],
            "test_accuracy_percent": accuracy[label]["test_accuracy_percent"],
            "accuracy_drop_percent_points": accuracy[label]["accuracy_drop_percent_points"],
            "total_connections": total,
            "active_connections": active,
            "pruned_connections": pruned,
            "sparsity_percent": float(summary["sparsity_percent"]),
            "density_percent": float(summary["density_percent"]),
            "logical_connectivity_reduction_percent": float(summary["logical_connectivity_reduction_percent"]),
            "estimated_dense_macs": int(summary["estimated_dense_macs"]),
            "estimated_effective_macs": int(summary["estimated_effective_macs"]),
            "theoretical_mac_reduction_percent": float(summary["theoretical_mac_reduction_percent"]),
        }
        for batch_size in args.batch_sizes:
            latency = measure_latency(model, inputs[batch_size], args.warmup, args.iterations, device)
            for metric, value in latency.items():
                row[f"batch_{batch_size}_{metric}"] = value
        rows.append(row)

    frame = pd.DataFrame(rows)
    layer_frame = pd.DataFrame(layer_rows)
    for batch_size in args.batch_sizes:
        dense_latency = float(frame.loc[frame["model"] == "Dense", f"batch_{batch_size}_mean_ms"].iloc[0])
        frame[f"batch_{batch_size}_latency_reduction_vs_dense_percent"] = (
            (dense_latency - frame[f"batch_{batch_size}_mean_ms"]) / dense_latency * 100.0
        )
    dense_size = float(frame.loc[frame["model"] == "Dense", "checkpoint_size_bytes"].iloc[0])
    frame["checkpoint_size_change_vs_dense_percent"] = (dense_size - frame["checkpoint_size_bytes"]) / dense_size * 100.0

    environment: dict[str, object] = {
        "benchmark_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_dir": str(args.benchmark_dir.relative_to(PROJECT_ROOT)),
        "seed": args.seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_num_threads": torch.get_num_threads(),
        "batch_sizes": args.batch_sizes,
        "warmup_iterations": args.warmup,
        "measurement_iterations": args.iterations,
        "accuracy_source": str(report_csv.relative_to(PROJECT_ROOT)),
    }
    report_view = frame.copy()
    environment["accuracy_sparsity_pareto_frontier"] = pareto_rows(report_view, "test_accuracy_percent")
    environment["latency_accuracy_pareto_frontier_batch_1"] = pareto_rows(
        report_view, "test_accuracy_percent", "batch_1_mean_ms"
    ) if 1 in args.batch_sizes else []
    environment["models"] = [label for label, _, _ in loaded]
    environment["layer_report"] = "layer_efficiency.csv"

    csv_path = reports_dir / "efficiency_results.csv"
    frame.to_csv(csv_path, index=False)
    layer_frame.to_csv(reports_dir / "layer_efficiency.csv", index=False)
    (reports_dir / "summary.json").write_text(
        json.dumps({"metadata": environment, "results": rows}, indent=2),
        encoding="utf-8",
    )
    write_markdown(frame, layer_frame, environment, reports_dir / "efficiency_results.md")
    plot_metric(frame, "test_accuracy_percent", plots_dir / "sparsity_vs_accuracy.png", "Accuracy vs Logical Sparsity", "Test accuracy (%)")
    plot_metric(frame, "theoretical_mac_reduction_percent", plots_dir / "sparsity_vs_macs.png", "Ideal MAC Reduction vs Logical Sparsity", "Theoretical MAC reduction (%)")
    if 1 in args.batch_sizes:
        plot_metric(frame, "batch_1_mean_ms", plots_dir / "sparsity_vs_latency.png", "Measured Batch-1 Latency vs Logical Sparsity", "Mean latency (ms)")
    plot_metric(frame, "checkpoint_size_mib", plots_dir / "sparsity_vs_checkpoint_size.png", "Checkpoint Size vs Logical Sparsity", "Checkpoint size (MiB)")
    print(frame.to_string(index=False))
    print(f"\nWrote efficiency reports to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
