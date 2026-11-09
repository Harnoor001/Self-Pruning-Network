"""Importance criteria. Importing this package registers every criterion in
``self_pruning_network.importance.SCORERS``.

The spec places baselines in a top-level ``baselines/`` folder; they live inside
the installed package instead so that they are importable everywhere without
path manipulation (documented deviation).
"""
from self_pruning_network.baselines import (  # noqa: F401
    activation, gates, l1_norm, l2_norm, network_slimming, random, taylor,
)
