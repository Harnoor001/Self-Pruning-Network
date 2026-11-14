import math

import pytest
import torch

from self_pruning_network.experiment import MethodSpec, _deep_merge, load_config
from self_pruning_network.research_data import fixed_split_indices, legacy_split_indices
from self_pruning_network.stats import describe, holm, minimum_wilcoxon_pvalue, paired_comparison


def test_describe_matches_manual_t_interval():
    d = describe([1.0, 2.0, 3.0, 4.0, 5.0])
    assert d["mean"] == 3.0 and d["sd"] == pytest.approx(math.sqrt(2.5))
    assert d["ci95_high"] - d["mean"] == pytest.approx(2.7764451 * math.sqrt(2.5) / math.sqrt(5), rel=1e-6)


def test_paired_comparison_and_holm():
    out = paired_comparison([0.52, 0.53, 0.51, 0.54, 0.50], [0.50, 0.51, 0.50, 0.51, 0.49])
    assert out["diff_mean"] == pytest.approx(0.018) and out["wins"] == 5
    assert out["wilcoxon_pvalue"] == pytest.approx(minimum_wilcoxon_pvalue(5))
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert math.isnan(holm([float("nan"), 0.01])[0])


def test_fixed_split_is_seed_independent_and_disjoint():
    train, val = fixed_split_indices(50_000, 0)
    assert len(train) == 45_000 and len(val) == 5_000
    assert not set(train.tolist()) & set(val.tolist())
    assert torch.equal(fixed_split_indices(50_000, 0)[1], val)
    assert not torch.equal(legacy_split_indices(50_000, 42)[1], legacy_split_indices(50_000, 123)[1])


def test_method_spec_parsing():
    spec = MethodSpec.parse("slimming@slimming:global")
    assert (spec.criterion, spec.source, spec.allocation) == ("slimming", "slimming", "global")
    assert MethodSpec.parse("gate_in").source == "gated"
    with pytest.raises(ValueError):
        MethodSpec.parse("unknown@gated")


def test_config_inheritance_and_overrides():
    config = load_config("configs/smoke_vgg.yaml", ["fine_tune.epochs=3"])
    assert config["model"]["cfg"] == "vgg_tiny" and config["model"]["family"] == "vgg"
    assert config["pruning"]["ratios"] == [0.2, 0.4, 0.6] and config["pruning"]["min_keep"] == 2
    assert config["fine_tune"]["epochs"] == 3 and config["fine_tune"]["optimizer"] == "sgd"
    assert _deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}
