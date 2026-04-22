from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    markdown_table = results_frame.to_markdown(index=False)
    report = f"""# Self-Pruning Neural Network Report

## Summary

This project trains a feed-forward classifier with learnable gates attached to every weight.
Each gate is produced by a sigmoid-transformed parameter and multiplied element-wise with the corresponding weight.
An L1 penalty over the gate values encourages the optimizer to reduce the number of active connections.

## Why L1 Regularization Helps

The L1 penalty adds a direct cost for keeping gates open.
Since the gates are bounded in `(0, 1)`, the optimizer can lower the total loss by pushing many gate values toward zero.
That creates a sparse network where only useful connections remain active enough to justify their cost.

## Results

Best lambda: `{best_lambda}`

Best test accuracy: `{best_accuracy:.4f}`

Best sparsity level: `{best_sparsity:.2f}%`

{markdown_table}
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

