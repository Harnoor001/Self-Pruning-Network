"""Experiment orchestration for the comparative pruning study.

Layout of an experiment root (``config['output_root']``)::

    experiment_config.yaml           resolved configuration (last invocation)
    environments/<utc>.json          environment metadata of every invocation
    seed<S>/sources/<name>/          source.pt, training_history.csv, info.json, last.pt
    seed<S>/comparison/rows/*.json   one result row per (method, ratio, ft budget)
    seed<S>/comparison/histories/    fine-tuning histories
    seed<S>/reliability/             scores.pt, damage.csv, correlations.csv, stability.csv
    results.csv / results.json       all rows of all seeds (rebuilt by ``collect_results``)

Every expensive unit of work writes its own file and is skipped when that file
exists, so interrupted runs resume where they stopped.
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import yaml

from self_pruning_network.architectures import build_model, model_config, set_source_mode, to_plain, unit_widths
from self_pruning_network.evaluation import (
    count_macs, deployable_parameter_count, deployable_size_bytes, environment_metadata, evaluate,
)
from self_pruning_network.importance import SCORERS, ScoringContext, compute_scores
from self_pruning_network.research_data import (
    Loaders, build_research_loaders, load_cifar10_arrays,
)
from self_pruning_network.structured_pruning import (
    compact_model, find_unit_groups, global_keep, pruning_summary, uniform_keep, zeroed_units,
)
from self_pruning_network.training import TrainConfig, fine_tune, set_determinism, train_model

Log = Callable[[str], None]


# --------------------------------------------------------------------------- configuration
def _deep_merge(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for parent in reversed(config.pop("extends", []) or []):
        config = _deep_merge(load_config(path.parent / parent), config)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return config


def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def record_invocation(config: dict, log: Log = print) -> dict:
    root = Path(config["output_root"])
    (root / "environments").mkdir(parents=True, exist_ok=True)
    env = environment_metadata()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    (root / "environments" / f"{stamp}.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (root / "experiment_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    log(f"environment: torch {env['torch_version']} cuda={env['cuda_available']} "
        f"gpu={env.get('gpu_name', 'none')} python {env['python_version']}")
    return env


# --------------------------------------------------------------------------- data
_ARRAY_CACHE: dict[str, tuple] = {}


def make_loaders(config: dict, seed: int, loader_seed: int | None = None) -> Loaders:
    data = config["data"]
    key = str(Path(data.get("data_dir", "data")).resolve())
    if key not in _ARRAY_CACHE:
        _ARRAY_CACHE[key] = load_cifar10_arrays(data.get("data_dir", "data"))
    split_seed = seed if data.get("split_mode", "fixed") == "legacy" else int(data.get("split_seed", 0))
    loaders = build_research_loaders(
        data_dir=data.get("data_dir", "data"), batch_size=int(data.get("batch_size", 128)),
        num_workers=int(data.get("num_workers", 0)), augmentation=data.get("augmentation", "standard"),
        split_seed=split_seed, loader_seed=seed if loader_seed is None else loader_seed,
        calibration_size=int(data.get("calibration_size", 2048)), eval_batch_size=data.get("eval_batch_size"),
        train_limit=data.get("train_limit"), validation_limit=data.get("validation_limit"),
        test_limit=data.get("test_limit"), arrays=_ARRAY_CACHE[key],
        legacy_split=data.get("split_mode", "fixed") == "legacy")
    return loaders


def calibration_loader(config: dict, loaders: Loaders, offset: int = 0):
    """Calibration subset; ``offset=1`` returns the disjoint next block (stability check)."""
    if offset == 0:
        return loaders.calibration
    from torch.utils.data import DataLoader, Subset
    n = len(loaders.calibration.dataset)
    base = loaders.train.dataset
    start = n * offset
    from self_pruning_network.research_data import TensorImageDataset
    ds = TensorImageDataset(base.images[start:start + n], base.labels[start:start + n], "none")
    return DataLoader(ds, batch_size=loaders.calibration.batch_size, shuffle=False)


# --------------------------------------------------------------------------- sources
def seed_root(config: dict, seed: int) -> Path:
    return Path(config["output_root"]) / f"seed{seed}"


def source_spec(config: dict, name: str) -> dict:
    if name not in config.get("sources", {}):
        raise KeyError(f"source {name!r} not defined in config")
    return config["sources"][name] or {}


def source_train_config(config: dict, name: str, seed: int) -> TrainConfig:
    spec = source_spec(config, name)
    values = _deep_merge(config.get("train", {}), spec.get("train", {}))
    objective = spec.get("objective", name)
    values.setdefault("gate_lambda", 0.0)
    if objective != "gated":
        values["gate_lambda"] = 0.0
        values["train_gates"] = False
    if objective != "slimming":
        values["slimming_lambda"] = 0.0
    values["loader_seed"] = seed
    return TrainConfig(**values)


def new_source_model(config: dict, name: str) -> torch.nn.Module:
    """All sources share the gated architecture (identical initialisation for a seed);
    non-gated objectives run it in dense mode with frozen gates."""
    objective = source_spec(config, name).get("objective", name)
    model = build_model({**config["model"], "gated": True})
    set_source_mode(model, "soft" if objective == "gated" else "dense")
    return model


def load_legacy_checkpoint(path: str | Path) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["model_config"]
    model = build_model({"family": "mlp", "gated": True, **{k: cfg[k] for k in
                         ("input_dim", "hidden_dims", "num_classes", "dropout", "use_batchnorm")}})
    model.load_state_dict(payload["model_state_dict"], strict=False)
    mode = payload.get("pruning", {}).get("mode", "soft")
    set_source_mode(model, mode if mode in {"soft", "dense", "hard"} else "soft")
    return model, payload


def get_source(config: dict, seed: int, name: str, loaders: Loaders, device: torch.device,
               dry_run: bool = False, log: Log = print) -> tuple[torch.nn.Module | None, dict]:
    spec = source_spec(config, name)
    directory = seed_root(config, seed) / "sources" / name
    path = directory / "source.pt"
    if spec.get("checkpoint"):
        template = spec["checkpoint"]
        if isinstance(template, dict):
            template = template.get(seed, template.get(str(seed)))
            if template is None:
                raise KeyError(f"source {name!r} has no legacy checkpoint for seed {seed}")
        ckpt = Path(str(template).format(seed=seed))
        if dry_run:
            log(f"[dry-run] would load legacy source {name} from {ckpt}")
            return None, {}
        model, payload = load_legacy_checkpoint(ckpt)
        info = {"source": name, "legacy_checkpoint": str(ckpt), "train_time_s": None,
                "legacy_experiment_config": payload.get("experiment_config")}
        return model.to(device).eval(), info
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(payload["model_config"])
        model.load_state_dict(payload["state_dict"])
        set_source_mode(model, payload["mode"])
        return model.to(device).eval(), payload["info"]
    cfg = source_train_config(config, name, seed)
    if dry_run:
        log(f"[dry-run] would train source {name} seed {seed}: {cfg.to_dict()}")
        return None, {}
    log(f"training source {name} seed {seed} ({cfg.epochs} epochs)")
    set_determinism(seed, bool(config.get("deterministic", True)))
    model = new_source_model(config, name)
    model, history, info = train_model(model, loaders.train, loaders.validation, cfg, device,
                                       checkpoint_dir=directory, log=log, tag=f"{name}/s{seed}")
    validation = evaluate(model, loaders.validation, device)
    test = evaluate(model, loaders.test, device)
    mode = "soft" if source_spec(config, name).get("objective", name) == "gated" else "dense"
    info = {**info, "source": name, "seed": seed, "validation": validation, "test": test,
            "train_config": cfg.to_dict(), "deployable_parameters": deployable_parameter_count(model)}
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(directory / "training_history.csv", index=False)
    (directory / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    torch.save({"model_config": model_config(model), "state_dict": model.state_dict(), "mode": mode,
                "info": info, "history": history}, path)
    (directory / "last.pt").unlink(missing_ok=True)
    return model.eval(), info


# --------------------------------------------------------------------------- methods
@dataclass(frozen=True)
class MethodSpec:
    criterion: str
    source: str
    allocation: str = "uniform"

    @classmethod
    def parse(cls, text: str) -> "MethodSpec":
        body, _, allocation = text.partition(":")
        criterion, _, source = body.partition("@")
        if criterion not in SCORERS:
            raise ValueError(f"unknown criterion in method spec {text!r}")
        return cls(criterion, source or "gated", allocation or "uniform")

    @property
    def id(self) -> str:
        suffix = "" if self.allocation == "uniform" else f":{self.allocation}"
        return f"{self.criterion}@{self.source}{suffix}"

    @property
    def slug(self) -> str:
        return self.id.replace("@", "_at_").replace(":", "_")


def _fmt_ratio(ratio: float) -> str:
    return f"{int(round(ratio * 100)):02d}"


def score_source(config: dict, seed: int, spec: MethodSpec, model, loaders: Loaders, device) -> tuple[list, float]:
    start = time.perf_counter()
    ctx = ScoringContext(model, device, seed=seed, calibration_loader=loaders.calibration)
    scores = compute_scores(spec.criterion, ctx)
    return scores, time.perf_counter() - start


def finetune_config(config: dict, epochs: int, seed: int) -> TrainConfig:
    values = dict(config["fine_tune"])
    values["epochs"] = int(epochs)
    values["loader_seed"] = seed + 1
    return TrainConfig(**values)


def _row_base(config: dict, seed: int, spec_id: str, source: str, criterion: str, allocation: str,
              ratio: float, epochs: int) -> dict:
    return {"experiment": config.get("name"), "architecture": config["model"]["family"], "seed": seed,
            "method": spec_id, "criterion": criterion, "source": source, "allocation": allocation,
            "target_ratio": ratio, "ft_epochs": epochs}


def evaluate_compact(config: dict, seed: int, compact, loaders: Loaders, device, epochs: int,
                     log: Log, tag: str) -> tuple[dict, list[dict], torch.nn.Module]:
    pre_val = evaluate(compact, loaders.validation, device)
    pre_test = evaluate(compact, loaders.test, device)
    set_determinism(seed + 1, bool(config.get("deterministic", True)))
    start = time.perf_counter()
    compact, history, info = fine_tune(compact, loaders.train, loaders.validation,
                                       finetune_config(config, epochs, seed), device, log=log, tag=tag)
    ft_time = time.perf_counter() - start
    val = evaluate(compact, loaders.validation, device)
    test = evaluate(compact, loaders.test, device)
    metrics = {
        "pre_ft_validation_accuracy": pre_val["accuracy"], "pre_ft_validation_loss": pre_val["loss"],
        "pre_ft_test_accuracy": pre_test["accuracy"],
        "validation_accuracy": val["accuracy"], "validation_loss": val["loss"],
        "test_accuracy": test["accuracy"], "test_loss": test["loss"],
        "best_ft_epoch": info["best_epoch"], "fine_tune_time_s": ft_time,
        "deployable_parameters": deployable_parameter_count(compact), "macs": count_macs(compact),
        "deployable_size_bytes": deployable_size_bytes(compact), "compact_widths": unit_widths(compact),
    }
    return metrics, history, compact


def run_comparison(config: dict, seed: int, device: torch.device, dry_run: bool = False, log: Log = print,
                   methods: list[str] | None = None, ratios: list[float] | None = None,
                   ft_epochs: list[int] | None = None, controls: bool | None = None) -> list[dict]:
    pruning = config["pruning"]
    methods = [MethodSpec.parse(m) for m in (methods or pruning["methods"])]
    ratios = [float(r) for r in (ratios or pruning["ratios"])]
    epochs_list = [int(e) for e in (ft_epochs or [config["fine_tune"]["epochs"]])]
    controls = pruning.get("controls", True) if controls is None else controls
    root = seed_root(config, seed) / "comparison"
    rows_dir, hist_dir = root / "rows", root / "histories"
    save_ckpt = bool(pruning.get("save_compact_checkpoints", False))
    loaders = None if dry_run else make_loaders(config, seed)
    needed_sources = sorted({m.source for m in methods} | ({m.source for m in methods} if controls else set()))
    sources: dict[str, tuple] = {}
    rows: list[dict] = []

    def todo(path: Path) -> bool:
        return not path.exists()

    plan = []
    for spec in methods:
        for ratio in ratios:
            for e in epochs_list:
                plan.append((spec, ratio, e, rows_dir / f"{spec.slug}__r{_fmt_ratio(ratio)}__ft{e}.json"))
    control_plan = [(src, e, rows_dir / f"control_at_{src}__r00__ft{e}.json")
                    for src in (needed_sources if controls else []) for e in epochs_list]
    pending = [p for p in plan if todo(p[3])] + [c for c in control_plan if todo(c[2])]
    log(f"seed {seed}: {len(plan) + len(control_plan)} result rows planned, {len(pending)} pending")
    if dry_run:
        for name in needed_sources:
            get_source(config, seed, name, None, device, dry_run=True, log=log)
        for item in pending:
            log(f"[dry-run] would run {item[-1].name}")
        return []

    for name in needed_sources:
        sources[name] = get_source(config, seed, name, loaders, device, log=log)
        model, info = sources[name]
        info.setdefault("validation", evaluate(model, loaders.validation, device))
        info.setdefault("test", evaluate(model, loaders.test, device))

    def source_fields(name: str, model) -> dict:
        info = sources[name][1]
        plain = to_plain(model)
        return {"source_validation_accuracy": info["validation"]["accuracy"],
                "source_test_accuracy": info["test"]["accuracy"],
                "source_train_time_s": info.get("train_time_s"),
                "dense_deployable_parameters": deployable_parameter_count(plain),
                "dense_macs": count_macs(plain), "dense_deployable_size_bytes": deployable_size_bytes(plain),
                "source_widths": unit_widths(plain)}

    def finish(row: dict, path: Path, history: list[dict], compact) -> None:
        row["parameter_reduction_percent"] = 100.0 * (1 - row["deployable_parameters"] / row["dense_deployable_parameters"])
        row["mac_reduction_percent"] = 100.0 * (1 - row["macs"] / row["dense_macs"])
        row["size_reduction_percent"] = 100.0 * (1 - row["deployable_size_bytes"] / row["dense_deployable_size_bytes"])
        row["accuracy_change_vs_source_pp"] = 100.0 * (row["test_accuracy"] - row["source_test_accuracy"])
        row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        rows_dir.mkdir(parents=True, exist_ok=True)
        hist_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(hist_dir / path.with_suffix(".csv").name, index=False)
        if save_ckpt:
            (root / "checkpoints").mkdir(parents=True, exist_ok=True)
            torch.save({"model_config": model_config(compact), "state_dict": compact.state_dict(), "row": row},
                       root / "checkpoints" / path.with_suffix(".pt").name)
        path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)

    for name, e, path in control_plan:
        if not todo(path):
            continue
        model = sources[name][0]
        compact = to_plain(model)
        row = _row_base(config, seed, f"control@{name}", name, "none", "none", 0.0, e)
        row.update(source_fields(name, model))
        metrics, history, compact = evaluate_compact(config, seed, compact, loaders, device, e, log, f"control@{name}")
        row.update(metrics)
        row.update({"units_removed_percent": 0.0, "scoring_time_s": 0.0, "pruning_time_s": 0.0,
                    "layer_removed_percent": [0.0] * len(row["source_widths"])})
        finish(row, path, history, compact)

    score_cache: dict[str, tuple[list, float]] = {}
    for spec, ratio, e, path in plan:
        if not todo(path):
            continue
        model = sources[spec.source][0]
        key = f"{spec.criterion}@{spec.source}"
        if key not in score_cache:
            score_cache[key] = score_source(config, seed, spec, model, loaders, device)
        scores, score_time = score_cache[key]
        start = time.perf_counter()
        min_keep = int(pruning.get("min_keep", 1))
        keep = (uniform_keep if spec.allocation == "uniform" else global_keep)(scores, ratio, min_keep)
        compact = compact_model(model, keep).to(device)
        prune_time = time.perf_counter() - start
        row = _row_base(config, seed, spec.id, spec.source, spec.criterion, spec.allocation, ratio, e)
        row.update(source_fields(spec.source, model))
        summary = pruning_summary(model, keep)
        row.update({"units_removed_percent": summary["units_removed_percent"],
                    "layer_removed_percent": summary["layer_removed_percent"],
                    "scoring_time_s": score_time, "pruning_time_s": prune_time,
                    "keep_indices_sha": _hash_keep(keep)})
        metrics, history, compact = evaluate_compact(config, seed, compact, loaders, device, e, log,
                                                     f"{spec.id} r={ratio} ft={e}")
        row.update(metrics)
        _save_keep(root / "masks" / f"{spec.slug}__r{_fmt_ratio(ratio)}.pt", keep)
        finish(row, path, history, compact)
        log(f"seed {seed} {spec.id:28s} r={ratio:.2f} ft={e}: pre {row['pre_ft_validation_accuracy']:.4f} "
            f"-> val {row['validation_accuracy']:.4f} test {row['test_accuracy']:.4f}")
    return rows


def _hash_keep(keep: list[torch.Tensor]) -> str:
    import hashlib
    digest = hashlib.sha1()
    for k in keep:
        digest.update(np.asarray(k.tolist(), dtype=np.int64).tobytes())
        digest.update(b"|")
    return digest.hexdigest()[:16]


def _save_keep(path: Path, keep: list[torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        torch.save([k.clone() for k in keep], path)


def collect_results(config: dict) -> pd.DataFrame:
    root = Path(config["output_root"])
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("seed*/comparison/rows/*.json"))]
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame.to_csv(root / "results.csv", index=False)
        (root / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return frame


# --------------------------------------------------------------------------- reliability (RQ5)
def run_reliability(config: dict, seed: int, device: torch.device, dry_run: bool = False, log: Log = print) -> None:
    rel = config.get("reliability", {})
    out = seed_root(config, seed) / "reliability"
    if (out / "correlations.csv").exists():
        log(f"seed {seed}: reliability already computed")
        return
    if dry_run:
        log(f"[dry-run] would run reliability for seed {seed}: sources={rel.get('sources')} "
            f"criteria={rel.get('criteria')} units/layer={rel.get('units_per_layer', 64)}")
        return
    loaders = make_loaders(config, seed)
    damage_loader = loaders.validation
    corr_rows, damage_rows, stability_rows = [], [], []
    all_scores: dict[str, list[torch.Tensor]] = {}
    for source in rel.get("sources", ["gated"]):
        model, _ = get_source(config, seed, source, loaders, device, log=log)
        model.eval()
        groups = find_unit_groups(model)
        base = evaluate(model, damage_loader, device)
        generator = torch.Generator().manual_seed(seed)
        per_layer = int(rel.get("units_per_layer", 64))
        sampled = [torch.randperm(g.n_units, generator=generator)[:min(per_layer, g.n_units)].sort().values
                   for g in groups]
        damage: list[torch.Tensor] = []
        start = time.perf_counter()
        for g, units in zip(groups, sampled):
            values = []
            for unit in units.tolist():
                with zeroed_units(model, g, [unit]):
                    metrics = evaluate(model, damage_loader, device)
                values.append(metrics["loss"] - base["loss"])
                damage_rows.append({"seed": seed, "source": source, "group": g.index, "unit": unit,
                                    "loss_increase": metrics["loss"] - base["loss"],
                                    "accuracy_drop": base["accuracy"] - metrics["accuracy"]})
            damage.append(torch.tensor(values, dtype=torch.float64))
        log(f"seed {seed} source {source}: measured removal damage for {sum(len(s) for s in sampled)} units "
            f"in {time.perf_counter() - start:.0f}s")
        gated_source = source_spec(config, source).get("objective", source) == "gated"
        criteria = [c for c in rel.get("criteria", list(SCORERS)) if gated_source or not SCORERS[c].needs_gates]
        for criterion in criteria:
            ctx = ScoringContext(model, device, seed=seed, calibration_loader=loaders.calibration)
            scores = compute_scores(criterion, ctx)
            all_scores[f"{criterion}@{source}"] = scores
            rhos = []
            for g, units, d, s in zip(groups, sampled, damage, scores):
                rho = spearman(s[units], d)
                rhos.append(rho)
                corr_rows.append({"seed": seed, "source": source, "criterion": criterion, "group": g.index,
                                  "n_units": len(units), "spearman": rho})
            corr_rows.append({"seed": seed, "source": source, "criterion": criterion, "group": "mean",
                              "n_units": sum(len(u) for u in sampled),
                              "spearman": float(np.nanmean(rhos)) if np.isfinite(rhos).any() else math.nan})
            if SCORERS[criterion].needs_data:
                ctx2 = ScoringContext(model, device, seed=seed, calibration_loader=calibration_loader(config, loaders, 1))
                second = compute_scores(criterion, ctx2)
                for g, a, b in zip(groups, scores, second):
                    stability_rows.append({"seed": seed, "source": source, "criterion": criterion, "group": g.index,
                                           "calibration_resample_spearman": spearman(a, b)})
    out.mkdir(parents=True, exist_ok=True)
    torch.save(all_scores, out / "scores.pt")
    pd.DataFrame(damage_rows).to_csv(out / "damage.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(out / "stability.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(out / "correlations.csv", index=False)


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    from scipy.stats import spearmanr
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return math.nan
    return float(spearmanr(a, b).statistic)
