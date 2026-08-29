from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from self_pruning_network.data import build_cifar10_loaders
from self_pruning_network.model import SelfPruningMLP
from self_pruning_network.reporting import plot_layer_metrics
from self_pruning_network.train import (
    _model_config,
    _save_checkpoint,
    accuracy_drop_percent_points,
    evaluate,
    fine_tune,
    reset_loader_seed,
    set_seed,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run learned-vs-random pruning ablations and multi-seed validation.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts/ablation_benchmark")
    parser.add_argument("--reuse-seed42", action="store_true", help="Reuse the validated full-data seed-42 checkpoints.")
    parser.add_argument("--seed42-checkpoint-dir", default="artifacts/final_benchmark/checkpoints")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--targets", type=float, nargs="+", default=[20, 40, 60, 80])
    parser.add_argument("--multi-seed-targets", type=float, nargs="+", default=[60])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[2048, 1024, 512])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--disable-batchnorm", action="store_true")
    parser.add_argument("--fine-tune-epochs", type=int, default=1)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--positive-lambda", type=float, default=0.001)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> tuple[SelfPruningMLP, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["model_config"]
    model = SelfPruningMLP(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"],
        num_classes=config["num_classes"],
        dropout=config.get("dropout", 0.3),
        use_batchnorm=config.get("use_batchnorm", True),
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    pruning = payload.get("pruning", {})
    mode = pruning.get("mode", "soft")
    model.set_mode(mode if mode in {"dense", "soft", "hard"} else "soft")
    return model.to(device), payload


def checkpoint_matches(path: Path, hidden_dims: list[int], dropout: float, use_batchnorm: bool) -> bool:
    if not path.exists():
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("model_config", {})
    return (
        config.get("input_dim") == 3072
        and config.get("hidden_dims") == hidden_dims
        and config.get("num_classes") == 10
        and float(config.get("dropout", -1)) == float(dropout)
        and bool(config.get("use_batchnorm", False)) == use_batchnorm
    )


def model_mask(model: SelfPruningMLP) -> torch.Tensor:
    return torch.cat([layer.mask.detach().reshape(-1).cpu() for layer in model.prunable_layers])


def base_config(args: argparse.Namespace, seed: int, loaders: tuple[object, object, object]) -> dict[str, object]:
    train_loader, validation_loader, test_loader = loaders
    model = SelfPruningMLP(hidden_dims=args.hidden_dims, dropout=args.dropout, use_batchnorm=not args.disable_batchnorm)
    return {
        "dataset": "CIFAR-10",
        "data_dir": str(args.data_dir),
        "train_subset": args.train_subset,
        "test_subset": args.test_subset,
        "validation_ratio": 0.1,
        "seed": seed,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(validation_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "architecture": _model_config(model),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": "AdamW",
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR",
        "label_smoothing": args.label_smoothing,
        "fine_tune_epochs": args.fine_tune_epochs,
        "fine_tune_learning_rate": args.fine_tune_learning_rate,
    }


def make_row(
    *,
    seed: int,
    method: str,
    lambda_value: float | None,
    target: float,
    model: SelfPruningMLP,
    checkpoint: Path,
    reference_label: str | None,
    reference_accuracy: float | None,
    pre_finetune_accuracy: float | None = None,
    validation_accuracy: float | None = None,
) -> dict[str, object]:
    summary = model.gate_summary()
    test_accuracy = float(model._ablation_test_accuracy) if hasattr(model, "_ablation_test_accuracy") else None
    if test_accuracy is None:
        raise RuntimeError("Internal error: test accuracy was not attached before make_row.")
    drop = None
    if reference_accuracy is not None:
        drop = accuracy_drop_percent_points(reference_accuracy, test_accuracy)
    return {
        "seed": seed,
        "method": method,
        "lambda": lambda_value,
        "target_sparsity_percent": target,
        "actual_sparsity_percent": summary.sparsity_percent,
        "density_percent": summary.density_percent,
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
        "validation_accuracy_percent": validation_accuracy * 100.0 if validation_accuracy is not None else None,
        "test_accuracy_percent": test_accuracy * 100.0,
        "accuracy_drop_percent_points": drop,
        "pre_finetune_test_accuracy": pre_finetune_accuracy,
        "pre_finetune_test_accuracy_percent": (
            pre_finetune_accuracy * 100.0 if pre_finetune_accuracy is not None else None
        ),
        "fine_tune_recovery_percent_points": (
            (test_accuracy - pre_finetune_accuracy) * 100.0 if pre_finetune_accuracy is not None else None
        ),
        "total_connections": summary.total_weights,
        "active_connections": summary.active_weights,
        "pruned_connections": summary.pruned_weights,
        "mean_gate_value": summary.mean_gate_value,
        "reference_label": reference_label,
        "reference_test_accuracy": reference_accuracy,
        "reference_test_accuracy_percent": reference_accuracy * 100.0 if reference_accuracy is not None else None,
        "checkpoint_path": str(checkpoint.relative_to(PROJECT_ROOT)),
    }


def attach_accuracy(model: SelfPruningMLP, loaders: tuple[object, object, object], device: torch.device) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    validation = evaluate(model, loaders[1], device, criterion)
    test = evaluate(model, loaders[2], device, criterion)
    model._ablation_test_accuracy = float(test["accuracy"])
    return float(validation["accuracy"]), float(test["accuracy"])


def layer_rows(seed: int, method: str, target: float, model: SelfPruningMLP) -> list[dict[str, object]]:
    rows = []
    gate_summary = model.gate_summary()
    for layer in gate_summary.layers:
        rows.append({
            "seed": seed,
            "method": method,
            "target_sparsity_percent": target,
            "layer_index": layer["layer_index"],
            "total_connections": layer["weight_count"],
            "active_connections": layer["active_count"],
            "pruned_connections": layer["pruned_count"],
            "sparsity_percent": layer["sparsity_percent"],
            "mean_gate_value": layer["mean_gate_value"],
        })
    return rows


def run_pruned_variant(
    *,
    seed: int,
    method: str,
    target: float,
    base_model: SelfPruningMLP,
    base_result: dict[str, object],
    loaders: tuple[object, object, object],
    device: torch.device,
    args: argparse.Namespace,
    config: dict[str, object],
    output_dir: Path,
) -> tuple[dict[str, object], list[dict[str, object]], torch.Tensor]:
    variant = copy.deepcopy(base_model).to(device)
    if method == "learned":
        variant.hard_prune_target_sparsity(target / 100.0)
    elif method == "random":
        variant.random_prune_target_sparsity(target / 100.0, seed=seed)
    else:
        raise ValueError(f"Unsupported pruning method: {method}")
    mask = model_mask(variant)
    set_seed(seed + 10_000 + int(target) + (0 if method == "learned" else 1))
    reset_loader_seed(loaders[0], seed + 10_000 + int(target))
    validation_accuracy, test_accuracy, before_accuracy, history = fine_tune(
        variant, loaders[0], loaders[1], loaders[2], device,
        args.fine_tune_epochs, args.fine_tune_learning_rate, args.weight_decay,
        args.label_smoothing, float(base_result["lambda"] or 0.0),
    )
    variant._ablation_test_accuracy = test_accuracy
    checkpoint = output_dir / "checkpoints" / f"{method}_seed{seed}_target{int(target)}.pt"
    metrics = evaluate(variant, loaders[2], device, nn.CrossEntropyLoss())
    pruning = {
        "mode": "hard",
        "strategy": method,
        "threshold": None,
        "target_sparsity": target / 100.0,
        "random_seed": seed if method == "random" else None,
    }
    _save_checkpoint(
        checkpoint, variant, lambda_value=float(base_result["lambda"] or 0.0), metrics=metrics,
        history=history, seed=seed,
        experiment_config={**config, "method": method, "lambda": float(base_result["lambda"] or 0.0),
                           "pruning_strategy": method, "target_sparsity": target / 100.0,
                           "random_mask_seed": seed if method == "random" else None},
        pruning=pruning,
    )
    row = make_row(seed=seed, method=method, lambda_value=float(base_result["lambda"] or 0.0),
                   target=target, model=variant, checkpoint=checkpoint,
                   reference_label=f"soft_seed{seed}", reference_accuracy=float(base_result["test_accuracy"]),
                   pre_finetune_accuracy=before_accuracy, validation_accuracy=validation_accuracy)
    return row, layer_rows(seed, method, target, variant), mask


def pareto_labels(frame: pd.DataFrame) -> list[str]:
    hard = frame[frame["method"].isin(["learned", "random"])].copy()
    labels = []
    for index, candidate in hard.iterrows():
        competitors = hard.drop(index)
        dominated = (
            (competitors["test_accuracy"] >= candidate["test_accuracy"])
            & (competitors["actual_sparsity_percent"] >= candidate["actual_sparsity_percent"])
            & (
                (competitors["test_accuracy"] > candidate["test_accuracy"])
                | (competitors["actual_sparsity_percent"] > candidate["actual_sparsity_percent"])
            )
        ).any()
        if not dominated:
            labels.append(f"{candidate['method']} seed{int(candidate['seed'])} {candidate['target_sparsity_percent']:.0f}%")
    return labels


def plot_learned_random(aggregate: pd.DataFrame, column: str, destination: Path, title: str, y_label: str) -> None:
    plt.figure(figsize=(8, 5))
    for method, color in [("learned", "#1768ac"), ("random", "#c96f3b")]:
        subset = aggregate[aggregate["method"] == method].sort_values("target_sparsity_percent")
        if subset.empty:
            continue
        plt.errorbar(subset["target_sparsity_percent"], subset[f"mean_{column}"], yerr=subset[f"std_{column}"],
                     marker="o", capsize=4, linewidth=2, label=method.title(), color=color)
    plt.title(title)
    plt.xlabel("Target sparsity (%)")
    plt.ylabel(y_label)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=160)
    plt.close()


def write_reports(results: list[dict[str, object]], layers: list[dict[str, object]], overlaps: list[dict[str, object]],
                  output_dir: Path, config: dict[str, object]) -> None:
    reports_dir = output_dir / "reports"
    plots_dir = output_dir / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    if "validation_accuracy_percent" not in frame:
        frame["validation_accuracy_percent"] = frame["validation_accuracy"] * 100.0
    if "test_accuracy_percent" not in frame:
        frame["test_accuracy_percent"] = frame["test_accuracy"] * 100.0
    if "reference_test_accuracy_percent" not in frame:
        frame["reference_test_accuracy_percent"] = frame["reference_test_accuracy"] * 100.0
    if "pre_finetune_test_accuracy_percent" not in frame:
        frame["pre_finetune_test_accuracy_percent"] = frame["pre_finetune_test_accuracy"] * 100.0
    if {"test_accuracy", "pre_finetune_test_accuracy"}.issubset(frame.columns):
        has_pre_finetune = frame["pre_finetune_test_accuracy"].notna()
        frame.loc[has_pre_finetune, "fine_tune_recovery_percent_points"] = (
            (frame.loc[has_pre_finetune, "test_accuracy"] -
             frame.loc[has_pre_finetune, "pre_finetune_test_accuracy"]) * 100.0
        )
    layer_frame = pd.DataFrame(layers)
    overlap_frame = pd.DataFrame(overlaps)
    hard = frame[frame["method"].isin(["learned", "random"])].copy()
    aggregate_source = frame[frame["method"].isin(["dense", "soft_lambda_0", "learned", "random"])]
    aggregate = aggregate_source.groupby(["method", "target_sparsity_percent"], as_index=False).agg(
        mean_test_accuracy=("test_accuracy_percent", "mean"), std_test_accuracy=("test_accuracy_percent", "std"),
        mean_accuracy_drop=("accuracy_drop_percent_points", "mean"), std_accuracy_drop=("accuracy_drop_percent_points", "std"),
        mean_actual_sparsity=("actual_sparsity_percent", "mean"), std_actual_sparsity=("actual_sparsity_percent", "std"),
        seed_count=("seed", "count"),
    )
    aggregate = aggregate.fillna(0.0)
    multi_seed_targets = config["multi_seed_targets"]
    multi_seed = aggregate[
        ((aggregate["method"].isin(["dense", "soft_lambda_0"])) & (aggregate["target_sparsity_percent"] == 0.0))
        | ((aggregate["method"].isin(["learned", "random"])) &
           (aggregate["target_sparsity_percent"].isin(multi_seed_targets)))
    ].copy()
    advantages = []
    for target in sorted(hard["target_sparsity_percent"].unique()):
        learned = hard[(hard["method"] == "learned") & (hard["target_sparsity_percent"] == target)]
        random_rows = hard[(hard["method"] == "random") & (hard["target_sparsity_percent"] == target)]
        merged = learned[["seed", "test_accuracy_percent"]].merge(random_rows[["seed", "test_accuracy_percent"]], on="seed", suffixes=("_learned", "_random"))
        for _, row in merged.iterrows():
            advantages.append({"seed": int(row["seed"]), "target_sparsity_percent": target,
                               "learned_test_accuracy_percent": row["test_accuracy_percent_learned"],
                               "random_test_accuracy_percent": row["test_accuracy_percent_random"],
                               "learned_advantage_percent_points": row["test_accuracy_percent_learned"] - row["test_accuracy_percent_random"]})
    advantage_frame = pd.DataFrame(advantages)
    if not advantage_frame.empty:
        advantage_summary = advantage_frame.groupby("target_sparsity_percent", as_index=False).agg(
            mean_learned_advantage_percent_points=("learned_advantage_percent_points", "mean"),
            std_learned_advantage_percent_points=("learned_advantage_percent_points", "std"),
            seed_count=("seed", "count"),
        ).fillna(0.0)
    else:
        advantage_summary = pd.DataFrame(columns=[
            "target_sparsity_percent", "mean_learned_advantage_percent_points",
            "std_learned_advantage_percent_points", "seed_count",
        ])

    frame.to_csv(reports_dir / "ablation_results.csv", index=False)
    multi_seed.to_csv(reports_dir / "multi_seed_results.csv", index=False)
    layer_frame.to_csv(reports_dir / "layerwise_results.csv", index=False)
    overlap_frame.to_csv(reports_dir / "mask_overlap.csv", index=False)
    advantage_frame.to_csv(reports_dir / "learned_advantage.csv", index=False)
    advantage_summary.to_csv(reports_dir / "learned_advantage_summary.csv", index=False)

    plot_learned_random(aggregate, "test_accuracy", plots_dir / "learned_vs_random_accuracy.png",
                        "Learned vs Random Accuracy", "Test accuracy (%)")
    plot_learned_random(aggregate, "accuracy_drop", plots_dir / "learned_vs_random_accuracy_drop.png",
                        "Learned vs Random Accuracy Drop", "Accuracy drop (percentage points)")

    key = frame[
        ((frame["method"].isin(["dense", "soft_lambda_0"])) & (frame["target_sparsity_percent"] == 0.0))
        | ((frame["method"].isin(["learned", "random"])) &
           (frame["target_sparsity_percent"].isin(multi_seed_targets)))
    ]
    key_aggregate = key.groupby("method", as_index=False).agg(mean_test_accuracy=("test_accuracy_percent", "mean"),
                                                                std_test_accuracy=("test_accuracy_percent", "std"))
    key_aggregate = key_aggregate.fillna(0.0)
    plt.figure(figsize=(8, 5))
    if not key_aggregate.empty:
        x = np.arange(len(key_aggregate))
        plt.bar(x, key_aggregate["mean_test_accuracy"], yerr=key_aggregate["std_test_accuracy"], capsize=5,
                color=["#1768ac" if method in {"dense", "learned"} else "#c96f3b" for method in key_aggregate["method"]])
        plt.xticks(x, key_aggregate["method"].str.replace("_", " ").str.title())
    plt.title("Multi-Seed Test Accuracy Variability")
    plt.ylabel("Mean test accuracy (%)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots_dir / "seed_variance.png", dpi=160)
    plt.close()

    layer_key = layer_frame[(layer_frame["method"].isin(["learned", "random"])) &
                            (layer_frame["target_sparsity_percent"] == config["layer_plot_target"]) &
                            (layer_frame["seed"] == config["layer_plot_seed"])]
    if not layer_key.empty:
        pivot = layer_key.pivot(index="layer_index", columns="method", values="sparsity_percent")
        pivot.plot(kind="bar", figsize=(8, 5), color=["#1768ac", "#c96f3b"])
        plt.title(f"Layer-wise Sparsity at {config['layer_plot_target']:.0f}% (seed {config['layer_plot_seed']})")
        plt.xlabel("Layer index")
        plt.ylabel("Sparsity (%)")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(plots_dir / "layerwise_sparsity.png", dpi=160)
        plt.close()
    else:
        plot_layer_metrics([], plots_dir / "layerwise_sparsity.png")

    plt.figure(figsize=(8, 5))
    if not overlap_frame.empty:
        plt.bar(overlap_frame["target_sparsity_percent"].astype(str), overlap_frame["pruned_overlap_percent"], color="#6b46c1")
    plt.title("Learned/Random Pruned-Mask Overlap")
    plt.xlabel("Target sparsity (%)")
    plt.ylabel("Overlap of learned-pruned set (%)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots_dir / "mask_overlap.png", dpi=160)
    plt.close()

    results_table = frame.sort_values(["seed", "method", "target_sparsity_percent"]).copy()
    display_columns = ["seed", "method", "lambda", "target_sparsity_percent", "actual_sparsity_percent",
                       "validation_accuracy_percent", "test_accuracy_percent", "accuracy_drop_percent_points",
                       "pre_finetune_test_accuracy_percent", "fine_tune_recovery_percent_points",
                       "active_connections", "pruned_connections"]
    summary_text = (
        "# Ablation Benchmark\n\n"
        "Learned and random pruning start from the same unpruned soft-gated checkpoint for each seed. They use the same global target sparsity, one fixed mask during fine-tuning, and the same data/evaluation protocol.\n\n"
        "## Per-run results\n\n" + results_table[display_columns].to_markdown(index=False) + "\n\n"
        "## Learned advantage over random\n\n" + (advantage_frame.to_markdown(index=False) if not advantage_frame.empty else "No paired results.") + "\n\n"
        "## Mean learned advantage over random\n\n" + (advantage_summary.to_markdown(index=False) if not advantage_summary.empty else "No paired results.") + "\n\n"
        "## Accuracy/sparsity Pareto configurations\n\n" + ", ".join(pareto_labels(frame)) + "\n\n"
        "## Interpretation\n\n"
        "The learned-vs-random difference is descriptive for the evaluated seeds. Three seeds are not sufficient for a strong statistical-significance claim. Logical sparsity and identical active counts are enforced by construction; accuracy differences measure the selection strategy and fine-tuning outcome.\n"
    )
    (reports_dir / "ablation_results.md").write_text(summary_text, encoding="utf-8")
    multi_text = (
        "# Multi-Seed Summary\n\n"
        "Mean and sample standard deviation are computed from the available seeds. A standard deviation of zero indicates only one seed was available for that target.\n\n"
        + (multi_seed.to_markdown(index=False) if not multi_seed.empty else "No multi-seed hard-pruning results.") + "\n"
    )
    (reports_dir / "multi_seed_summary.md").write_text(multi_text, encoding="utf-8")
    summary = {
        "experiment": "learned_vs_random_pruning_ablation",
        "configuration": config,
        "results": json.loads(frame.to_json(orient="records")),
        "multi_seed_summary": json.loads(multi_seed.to_json(orient="records")),
        "learned_advantage": json.loads(advantage_frame.to_json(orient="records")) if not advantage_frame.empty else [],
        "learned_advantage_summary": json.loads(advantage_summary.to_json(orient="records")) if not advantage_summary.empty else [],
        "mask_overlap": json.loads(overlap_frame.to_json(orient="records")) if not overlap_frame.empty else [],
        "report_files": sorted(path.name for path in reports_dir.iterdir() if path.is_file()),
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_seed(args.seeds[0])
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    seed42_dir = Path(args.seed42_checkpoint_dir).resolve()
    all_results: list[dict[str, object]] = []
    all_layers: list[dict[str, object]] = []
    overlaps: list[dict[str, object]] = []
    seed42_reused = False

    for seed in args.seeds:
        set_seed(seed)
        loaders = build_cifar10_loaders(args.data_dir, args.batch_size, args.num_workers,
                                         train_subset=args.train_subset, test_subset=args.test_subset, seed=seed)
        config = base_config(args, seed, loaders)
        use_reuse = (
            args.reuse_seed42 and seed == 42 and args.train_subset is None and args.test_subset is None
            and checkpoint_matches(seed42_dir / "dense_lambda_0.0000.pt", args.hidden_dims, args.dropout, not args.disable_batchnorm)
        )
        seed_output = output_dir / f"seed{seed}"
        if use_reuse:
            dense_path = seed42_dir / "dense_lambda_0.0000.pt"
            soft_path = seed42_dir / "soft_lambda_0.0000.pt"
            dense_model, _ = load_checkpoint(dense_path, device)
            soft_model, soft_payload = load_checkpoint(soft_path, device)
            dense_val, dense_test = attach_accuracy(dense_model, loaders, device)
            soft_val, soft_test = attach_accuracy(soft_model, loaders, device)
            seed42_reused = True
        else:
            dense_result, dense_model, _ = train_model(
                "dense", 0.0, loaders[0], loaders[1], loaders[2], device, args.epochs,
                args.learning_rate, args.weight_decay, args.label_smoothing, args.hidden_dims,
                args.dropout, not args.disable_batchnorm, seed_output, seed, config,
            )
            soft_result, soft_model, _ = train_model(
                "soft", 0.0, loaders[0], loaders[1], loaders[2], device, args.epochs,
                args.learning_rate, args.weight_decay, args.label_smoothing, args.hidden_dims,
                args.dropout, not args.disable_batchnorm, seed_output, seed, {**config, "lambda": 0.0},
            )
            dense_val, dense_test = dense_result.validation_accuracy, dense_result.test_accuracy
            soft_val, soft_test = soft_result.validation_accuracy, soft_result.test_accuracy
            soft_payload = {"lambda_value": 0.0}

        dense_model.set_mode("dense")
        dense_model._ablation_test_accuracy = dense_test
        all_results.append(make_row(seed=seed, method="dense", lambda_value=None, target=0.0,
                                    model=dense_model, checkpoint=(seed42_dir if use_reuse else seed_output / "checkpoints") / "dense_lambda_0.0000.pt" if use_reuse else seed_output / "checkpoints" / "dense_lambda_0.0000.pt",
                                    reference_label="dense", reference_accuracy=dense_test,
                                    validation_accuracy=dense_val))
        soft_model.set_mode("soft")
        soft_model._ablation_test_accuracy = soft_test
        soft_checkpoint = (seed42_dir / "soft_lambda_0.0000.pt") if use_reuse else seed_output / "checkpoints" / "soft_lambda_0.0000.pt"
        all_results.append(make_row(seed=seed, method="soft_lambda_0", lambda_value=0.0, target=0.0,
                                    model=soft_model, checkpoint=soft_checkpoint,
                                    reference_label=f"dense_seed{seed}", reference_accuracy=dense_test,
                                    validation_accuracy=soft_val))

        if use_reuse:
            positive_path = seed42_dir / f"soft_lambda_{args.positive_lambda:.4f}.pt"
            if positive_path.exists():
                positive_model, _ = load_checkpoint(positive_path, device)
                positive_val, positive_test = attach_accuracy(positive_model, loaders, device)
                positive_model._ablation_test_accuracy = positive_test
                all_results.append(make_row(seed=seed, method="soft_lambda_positive", lambda_value=args.positive_lambda,
                                            target=0.0, model=positive_model, checkpoint=positive_path,
                                            reference_label=f"dense_seed{seed}", reference_accuracy=dense_test,
                                            validation_accuracy=positive_val))

        base_result = {"lambda": 0.0, "test_accuracy": soft_test}
        selected_targets = sorted(set(args.targets if seed == 42 else args.multi_seed_targets))
        run_masks: dict[tuple[float, str], torch.Tensor] = {}
        for target in selected_targets:
            for method in ("learned", "random"):
                row, layer_result, mask = run_pruned_variant(
                    seed=seed, method=method, target=target, base_model=soft_model,
                    base_result=base_result, loaders=loaders, device=device, args=args,
                    config={**config, "lambda": 0.0}, output_dir=seed_output,
                )
                all_results.append(row)
                all_layers.extend(layer_result)
                run_masks[(target, method)] = mask
            learned_mask = run_masks[(target, "learned")]
            random_mask = run_masks[(target, "random")]
            learned_pruned = learned_mask == 0
            random_pruned = random_mask == 0
            intersection = int((learned_pruned & random_pruned).sum().item())
            union = int((learned_pruned | random_pruned).sum().item())
            pruned_count = int(learned_pruned.sum().item())
            overlaps.append({
                "seed": seed,
                "target_sparsity_percent": target,
                "learned_pruned_connections": pruned_count,
                "random_pruned_connections": int(random_pruned.sum().item()),
                "intersection": intersection,
                "union": union,
                "pruned_overlap_percent": intersection / pruned_count * 100.0 if pruned_count else 0.0,
                "jaccard_percent": intersection / union * 100.0 if union else 0.0,
            })

    reporting_config = {
        **base_config(args, args.seeds[0], build_cifar10_loaders(args.data_dir, args.batch_size, args.num_workers,
                                                                  train_subset=args.train_subset, test_subset=args.test_subset,
                                                                  seed=args.seeds[0])),
        "seeds": args.seeds,
        "targets_full_seed42": args.targets,
        "multi_seed_targets": args.multi_seed_targets,
        "positive_lambda": args.positive_lambda,
        "reuse_seed42": seed42_reused,
        "random_pruning": "global random permutation with a local generator seeded by the run seed",
        "layer_plot_seed": 42 if 42 in args.seeds else args.seeds[0],
        "layer_plot_target": 60.0 if 60.0 in args.targets or 60.0 in args.multi_seed_targets else args.targets[0],
    }
    write_reports(all_results, all_layers, overlaps, output_dir, reporting_config)
    print(pd.DataFrame(all_results).to_string(index=False))
    print(f"\nWrote ablation reports to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
