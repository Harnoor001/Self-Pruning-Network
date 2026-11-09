"""Structured pruning comparison (RQ1, RQ2, RQ4).

Trains/loads sources, scores units with every configured criterion, builds compact
models at every ratio, fine-tunes them with the identical policy and records metrics.

Examples:
    python scripts/research/run_comparison.py --config configs/mlp.yaml --pilot
    python scripts/research/run_comparison.py --config configs/vgg_cifar10.yaml --dry-run
"""
from _common import base_parser, setup

from self_pruning_network.experiment import collect_results, run_comparison


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--methods", nargs="+", default=None,
                        help="e.g. gate_in@gated l1@dense slimming@slimming:global")
    parser.add_argument("--ratios", type=float, nargs="+", default=None)
    parser.add_argument("--ft-epochs", type=int, nargs="+", default=None)
    parser.add_argument("--no-controls", action="store_true")
    args = parser.parse_args()
    config, seeds, device, log = setup(args)
    for seed in seeds:
        run_comparison(config, seed, device, dry_run=args.dry_run, log=log, methods=args.methods,
                       ratios=args.ratios, ft_epochs=args.ft_epochs, controls=False if args.no_controls else None)
    if not args.dry_run:
        frame = collect_results(config)
        log(f"collected {len(frame)} rows into {config['output_root']}/results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
