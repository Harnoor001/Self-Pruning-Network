"""Train (or load cached) source models: dense, gated (existing method), slimming.

Examples:
    python scripts/research/train_sources.py --config configs/mlp.yaml --pilot
    python scripts/research/train_sources.py --config configs/vgg_cifar10.yaml --seeds 42 --sources dense gated
"""
from _common import base_parser, setup

from self_pruning_network.experiment import get_source, make_loaders


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--sources", nargs="+", default=None)
    args = parser.parse_args()
    config, seeds, device, log = setup(args)
    names = args.sources or list(config["sources"])
    for seed in seeds:
        loaders = None if args.dry_run else make_loaders(config, seed)
        for name in names:
            _, info = get_source(config, seed, name, loaders, device, dry_run=args.dry_run, log=log)
            if info.get("test"):
                log(f"seed {seed} {name}: val {info['validation']['accuracy']:.4f} test {info['test']['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
