from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


def write_results_table(results: list[dict[str, float | str]], destination: str | Path) -> pd.DataFrame:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(destination, index=False)
    return frame


def write_markdown_report(
    results_frame: pd.DataFrame,
    best_lambda: float,
    best_accuracy: float,
    best_sparsity: float,
    destination: str | Path,
    dense_accuracy: float | None = None,
    pruning_enabled: bool = False,
    experiment_config: dict[str, object] | None = None,
) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    markdown_table = results_frame.to_markdown(index=False)
    hard = results_frame[results_frame["model_type"] == "hard"].copy() if "model_type" in results_frame else pd.DataFrame()
    frontier_rows = []
    if not hard.empty:
        for _, candidate in hard.iterrows():
            dominated = ((hard["test_accuracy"] >= candidate["test_accuracy"]) &
                         (hard["sparsity_percent"] >= candidate["sparsity_percent"]) &
                         ((hard["test_accuracy"] > candidate["test_accuracy"]) |
                          (hard["sparsity_percent"] > candidate["sparsity_percent"]))).any()
            if not dominated:
                frontier_rows.append(candidate)
    frontier = pd.DataFrame(frontier_rows)
    frontier_text = frontier.to_markdown(index=False) if not frontier.empty else "No hard-pruned configurations were evaluated."
    recovery = hard[["target_sparsity_percent", "pre_finetune_test_accuracy", "test_accuracy"]].copy() if not hard.empty else pd.DataFrame()
    recovery_text = recovery.to_markdown(index=False) if not recovery.empty else "No hard-pruned fine-tuning runs were evaluated."
    config_text = ""
    if experiment_config is not None:
        config_text = "## Experiment configuration\n\n```json\n" + pd.Series(experiment_config).to_json(indent=2) + "\n```\n\n"
    dense_section = ""
    if dense_accuracy is not None:
        dense_section = f"## Dense reference\n\nSeparately trained dense-mode test accuracy: **{dense_accuracy:.6f}**. Accuracy drops for soft models are measured against this dense reference; hard-pruned drops are measured against the corresponding unpruned soft model.\n\n"
    report = f"""# Self-Pruning Neural Network Report

## Summary

This project trains a feed-forward classifier with learnable gates attached to every weight.
Each gate is produced by a sigmoid-transformed parameter and multiplied element-wise with the corresponding weight.
An L1 penalty over the gate values encourages the optimizer to reduce the number of active connections.

## Why L1 Regularization Helps

The L1 penalty adds a direct cost for keeping gates open.
Since the gates are bounded in `(0, 1)`, the optimizer can lower the total loss by pushing many gate values toward zero.
That creates a sparse network where only useful connections remain active enough to justify their cost.

## Method

The model learns `W' = W * sigmoid(S)` jointly with classification. When pruning is enabled, gate importance is converted to a persistent binary mask. A hard-pruned forward pass uses `W_hard = W * M`, and fine-tuning reapplies `M` after every optimizer step.

Logical sparsity is the fraction of masked connections. The tensors are still stored densely, so these results do not claim proportional storage savings or hardware speedup.

{dense_section}{config_text}## Results

Best lambda: `{best_lambda}`

Best test accuracy: `{best_accuracy:.4f}`

Best soft-model mask sparsity: `{best_sparsity:.2f}%`

Pruning enabled: `{pruning_enabled}`

{markdown_table}

## Accuracy/sparsity Pareto frontier

The non-dominated hard-pruned configurations are listed below. A configuration is non-dominated when no evaluated hard-pruned result has both higher or equal accuracy and higher or equal sparsity with one strict improvement.

{frontier_text}

## Fine-tuning recovery

The table below records test accuracy immediately after applying the hard mask and after the configured fine-tuning stage. Accuracy-drop values in the main table use the post-fine-tuning accuracy.

{recovery_text}
"""
    destination.write_text(report, encoding="utf-8")


def plot_gate_distribution(gate_values, destination: str | Path, title: str) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(gate_values, bins=50, color="#1768ac", edgecolor="white")
    plt.title(title)
    plt.xlabel("Gate Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(destination, dpi=160)
    plt.close()


def plot_lambda_metric(
    results_frame: pd.DataFrame,
    metric_column: str,
    destination: str | Path,
    title: str,
    y_label: str,
) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    ordered = results_frame.sort_values("lambda")
    plt.figure(figsize=(8, 5))
    plt.plot(ordered["lambda"], ordered[metric_column], marker="o", linewidth=2.2, color="#1768ac")
    if (ordered["lambda"] <= 0).any():
        plt.xscale("symlog", linthresh=1e-5)
    else:
        plt.xscale("log")
    plt.title(title)
    plt.xlabel("Lambda")
    plt.ylabel(y_label)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(destination, dpi=160)
    plt.close()


def plot_accuracy_vs_sparsity(results_frame: pd.DataFrame, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.scatter(
        results_frame["sparsity_percent"],
        results_frame["test_accuracy"],
        s=90,
        color="#c96f3b",
        edgecolors="white",
        linewidths=1.2,
    )
    for _, row in results_frame.iterrows():
        plt.annotate(f"λ={row['lambda']}", (row["sparsity_percent"], row["test_accuracy"]), xytext=(6, 6), textcoords="offset points")
    plt.title("Accuracy vs Sparsity Trade-off")
    plt.xlabel("Sparsity Level (%)")
    plt.ylabel("Test Accuracy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(destination, dpi=160)
    plt.close()


def plot_training_history(history_frame: pd.DataFrame, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    lambdas = history_frame["lambda"].unique().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    palette = ["#1768ac", "#c96f3b", "#2f855a", "#b83280", "#6b46c1"]

    for index, lambda_value in enumerate(lambdas):
        subset = history_frame[history_frame["lambda"] == lambda_value]
        color = palette[index % len(palette)]
        axes[0].plot(subset["epoch"], subset["train_loss"], marker="o", linewidth=1.8, color=color, label=f"λ={lambda_value}")
        axes[0].plot(subset["epoch"], subset["validation_loss"], marker="x", linestyle="--", linewidth=1.5, color=color, alpha=0.85)
        axes[1].plot(subset["epoch"], subset["train_accuracy"], marker="o", linewidth=1.8, color=color, label=f"λ={lambda_value}")
        axes[1].plot(subset["epoch"], subset["validation_accuracy"], marker="x", linestyle="--", linewidth=1.5, color=color, alpha=0.85)

    axes[0].set_title("Loss by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Accuracy by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].grid(alpha=0.25)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(lambdas)))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def plot_layer_metrics(layer_summary: list[dict[str, float | int]], destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    layer_indices = [item["layer_index"] for item in layer_summary]
    mean_gates = [item["mean_gate_value"] for item in layer_summary]
    sparsities = [item["sparsity_percent"] for item in layer_summary]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(layer_indices, mean_gates, color="#1768ac")
    axes[0].set_title("Mean Gate Value per Layer")
    axes[0].set_xlabel("Layer Index")
    axes[0].set_ylabel("Mean Gate Value")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(layer_indices, sparsities, color="#c96f3b")
    axes[1].set_title("Layer-wise Sparsity")
    axes[1].set_xlabel("Layer Index")
    axes[1].set_ylabel("Sparsity (%)")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
