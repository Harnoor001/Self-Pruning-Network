"""Importance-score reliability (RQ5) and calibration stability (RQ3).

For units sampled per layer, removes one unit at a time (exactly equivalent to
zeroing its activation in eval mode) and measures the validation-loss increase,
then correlates it (Spearman) with every criterion's score.

Example:
    python scripts/research/run_importance_analysis.py --config configs/mlp.yaml --pilot
"""
from _common import base_parser, setup

from self_pruning_network.experiment import run_reliability


def main() -> int:
    args = base_parser(__doc__).parse_args()
    config, seeds, device, log = setup(args)
    for seed in seeds:
        run_reliability(config, seed, device, dry_run=args.dry_run, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
