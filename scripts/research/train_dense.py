"""Alias of train_sources.py restricted to the dense reference source."""
import sys

if "--sources" not in sys.argv:
    sys.argv += ["--sources", "dense"]
from train_sources import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
