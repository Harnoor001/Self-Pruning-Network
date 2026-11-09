"""Efficiency benchmark of the pruned architectures (RQ4).

Under uniform allocation every criterion produces the same compact architecture at a
given ratio, so latency/memory are properties of the architecture and are measured
once per unique width vector (also covering global-allocation architectures found in
results.csv). Weights do not affect dense-kernel latency; models are built with
random weights. The gated training-time model (soft mode) is benchmarked too, to
quantify the overhead of evaluating W*sigmoid(S) at inference.

Example:
    python scripts/research/benchmark.py --config configs/mlp.yaml
"""
import ast
import json
from pathlib import Path

import pandas as pd
import torch
from _common import base_parser, setup

from self_pruning_network.architectures import build_model, set_source_mode, with_widths
from self_pruning_network.evaluation import (
    count_macs, deployable_parameter_count, deployable_size_bytes, environment_metadata, measure_latency,
    peak_gpu_memory_bytes,
)
from self_pruning_network.structured_pruning import keep_count


def main() -> int:
    args = base_parser(__doc__).parse_args()
    config, _, device, log = setup(args)
    bench = config.get("benchmark", {})
    base = {**config["model"], "gated": False}
    dense = build_model(base)
    from self_pruning_network.architectures import unit_widths
    dense_widths = unit_widths(dense)
    min_keep = int(config["pruning"].get("min_keep", 1))
    candidates = {("dense", 0.0): dense_widths}
    for r in config["pruning"]["ratios"]:
        candidates[("uniform", float(r))] = [keep_count(w, float(r), min_keep) for w in dense_widths]
    results_csv = Path(config["output_root"]) / "results.csv"
    if results_csv.exists():
        frame = pd.read_csv(results_csv)
        for _, row in frame[frame["allocation"] == "global"].drop_duplicates("compact_widths").iterrows():
            candidates[(f"global:{row['method']}:seed{row['seed']}", float(row["target_ratio"]))] = \
                ast.literal_eval(str(row["compact_widths"]))
    if args.dry_run:
        for key, widths in candidates.items():
            log(f"[dry-run] would benchmark {key}: {widths}")
        return 0
    input_shape = (3, 32, 32)
    rows = []
    torch.set_num_threads(int(bench.get("cpu_threads", 1)))
    entries = [(k, build_model(with_widths(base, w)), w, "compact (deployable)") for k, w in candidates.items()]
    gated = build_model({**config["model"], "gated": True})
    set_source_mode(gated, "soft")
    entries.append((("gated_soft_training_model", 0.0), gated, dense_widths, "gated W*sigmoid(S), training-time"))
    for (label, ratio), model, widths, kind in entries:
        model.eval()
        row = {"label": label, "target_ratio": ratio, "kind": kind, "widths": widths,
               "deployable_parameters": deployable_parameter_count(model), "macs": count_macs(model, (1, *input_shape)),
               "deployable_size_bytes": deployable_size_bytes(model),
               "full_parameter_count_incl_gates": sum(p.numel() for p in model.parameters())}
        for bs in bench.get("cpu_batch_sizes", [1, 64]):
            lat = measure_latency(model, bs, torch.device("cpu"), int(bench.get("warmup", 20)),
                                  int(bench.get("iterations", 100)), input_shape)
            row.update({f"cpu_bs{bs}_{k}": v for k, v in lat.items() if k.endswith("_ms")})
        if device.type == "cuda":
            for bs in bench.get("gpu_batch_sizes", [1, 256]):
                lat = measure_latency(model, bs, device, int(bench.get("warmup", 20)), int(bench.get("iterations", 100)),
                                      input_shape)
                row.update({f"gpu_bs{bs}_{k}": v for k, v in lat.items() if k.endswith("_ms")})
                row[f"gpu_bs{bs}_peak_memory_bytes"] = peak_gpu_memory_bytes(model, bs, device, input_shape)
            model.to("cpu")
        rows.append(row)
        log(f"{label} r={ratio}: params={row['deployable_parameters']:,} macs={row['macs']:,} "
            f"cpu bs1 median={row.get('cpu_bs1_median_ms', float('nan')):.3f} ms")
    out = Path(config["output_root"]) / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "benchmark_results.csv", index=False)
    (out / "benchmark_environment.json").write_text(json.dumps(
        {**environment_metadata(), "cpu_threads_used": torch.get_num_threads(), "benchmark_config": bench}, indent=2),
        encoding="utf-8")
    log(f"wrote {out / 'benchmark_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
