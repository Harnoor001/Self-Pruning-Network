"""Statistical helpers for multi-seed comparisons.

* ``describe`` — mean, sample SD (ddof=1), 95 % t-interval.
* ``paired_comparison`` — matched-seed paired t-test and exact Wilcoxon signed-rank test,
  mean paired difference with its 95 % t-interval.
* ``holm`` — Holm step-down adjustment of a family of p-values.

Nothing here decides "significance" by itself; the analysis script labels a result
"statistically supported" only when the Holm-adjusted paired t-test p < alpha and
reports the normality assumption alongside.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps


def describe(values) -> dict[str, float]:
    x = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": math.nan, "sd": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
    mean = float(x.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "sd": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
    sd = float(x.std(ddof=1))
    half = float(sps.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    return {"n": n, "mean": mean, "sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}


def paired_comparison(a, b) -> dict[str, float]:
    """Compare matched samples ``a`` and ``b`` (difference = a - b)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have equal length")
    d = a - b
    out = {"n_pairs": int(len(d)), **{f"diff_{k}": v for k, v in describe(d).items() if k != "n"}}
    if len(d) < 2 or np.allclose(d, d[0]):
        out.update({"t_stat": math.nan, "t_pvalue": math.nan, "wilcoxon_pvalue": math.nan,
                    "shapiro_pvalue": math.nan, "wins": int((d > 0).sum()), "losses": int((d < 0).sum())})
        return out
    t = sps.ttest_rel(a, b)
    try:
        w = sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method="exact").pvalue
    except ValueError:
        w = math.nan
    shapiro = sps.shapiro(d).pvalue if len(d) >= 3 else math.nan
    out.update({"t_stat": float(t.statistic), "t_pvalue": float(t.pvalue), "wilcoxon_pvalue": float(w),
                "shapiro_pvalue": float(shapiro), "wins": int((d > 0).sum()), "losses": int((d < 0).sum())})
    return out


def holm(pvalues) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.where(~np.isnan(p))[0]
    m = len(valid)
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[index]))
        adjusted[index] = running
    return adjusted.tolist()


def minimum_wilcoxon_pvalue(n: int) -> float:
    """Smallest attainable two-sided exact Wilcoxon p-value with ``n`` non-zero pairs."""
    return 2.0 / (2 ** n) if n > 0 else math.nan
