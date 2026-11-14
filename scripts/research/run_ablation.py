"""Ablations A (gate lambda / penalty normalisation) and C (fine-tuning budget).

B (gate aggregation), D (ratios) and E (seeds) are part of run_comparison.py.

Example:
    python scripts/research/run_ablation.py --config configs/mlp.yaml --pilot --which A C
"""
from pathlib import Path

import pandas as pd
from _common import base_parser, setup

from self_pruning_network.experiment import collect_results, get_source, make_loaders, run_comparison
from self_pruning_network.gates import gated_modules
from self_pruning_network.importance import ScoringContext, compute_scores


def gate_statistics(model, source: str, seed: int) -> list[dict]:
    device = next(model.parameters()).device
    scores = compute_scores("gate_in", ScoringContext(model, device))
    gates = [m.gates().detach().flatten().double().cpu() for m in gated_modules(model)]
    return [{"seed": seed, "source": source, "group": i, "neuron_score_mean": float(s.mean()),
             "neuron_score_std": float(s.std()), "neuron_score_range": float(s.max() - s.min()),
             "connection_gate_mean": float(gates[i].mean()), "connection_gate_std": float(gates[i].std())}
            for i, s in enumerate(scores)]


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--which", nargs="+", default=["A", "C"], choices=["A", "C"])
    args = parser.parse_args()
    config, seeds, device, log = setup(args)
    ablation = config["ablation"]
    if "A" in args.which:
        spec = ablation["gate_lambda"]
        config["sources"].update(spec["sources"])
        methods = [f"{c}@{s}" for s in spec["sources"] for c in spec["criteria"]]
        stats = []
        for seed in seeds:
            run_comparison(config, seed, device, dry_run=args.dry_run, log=log, methods=methods,
                           ratios=spec["ratios"], controls=True)
            if args.dry_run:
                continue
            loaders = make_loaders(config, seed)
            for name in [*spec["sources"], "gated"]:
                model, _ = get_source(config, seed, name, loaders, device, log=log)
                stats += gate_statistics(model, name, seed)
        if stats:
            out = Path(config["output_root"]) / "ablation_gate_statistics.csv"
            pd.DataFrame(stats).to_csv(out, index=False)
            log(f"wrote {out}")
    if "C" in args.which:
        spec = ablation["fine_tune"]
        for seed in seeds:
            run_comparison(config, seed, device, dry_run=args.dry_run, log=log, methods=spec["methods"],
                           ratios=spec["ratios"], ft_epochs=spec["epochs"], controls=True)
    if not args.dry_run:
        collect_results(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
