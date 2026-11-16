"""Shared CLI plumbing for research scripts."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_pruning_network.experiment import load_config, record_invocation, resolve_device  # noqa: E402


def base_parser(description: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML config, e.g. configs/mlp.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config entries, e.g. fine_tune.epochs=3 data.num_workers=0")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="defaults to config 'seeds'")
    parser.add_argument("--pilot", action="store_true", help="use config 'pilot_seeds'")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true", help="print the plan without training anything")
    return parser


def setup(args: argparse.Namespace):
    os.chdir(PROJECT_ROOT)
    config = load_config(args.config, args.set)
    seeds = args.seeds or (config.get("pilot_seeds") if args.pilot else None) or config.get("seeds", [42])
    device = resolve_device(args.device or config.get("device", "auto"))
    log_dir = Path(config["output_root"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{Path(sys.argv[0]).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    def log(message: str) -> None:
        line = f"{datetime.now().strftime('%H:%M:%S')} {message}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    if not args.dry_run:
        record_invocation(config, log)
    log(f"config={args.config} seeds={seeds} device={device} dry_run={args.dry_run}")
    if device.type == "cpu":
        log("NOTE: running on CPU. No GPU results are produced by this invocation.")
    return config, seeds, device, log
