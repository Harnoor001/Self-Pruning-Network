"""Statistical analysis, tables and publication figures from stored results.

Every number, table and plot is computed from files written by the experiment
scripts (results.csv, reliability/*.csv, benchmark/benchmark_results.csv,
ablation_gate_statistics.csv). Nothing is typed in by hand.

Outputs (under <output_root>/analysis/ and research/paper/{tables,figures}/<name>/):
    summary.csv, paired_tests.csv, reliability_summary.csv, stability_summary.csv,
    findings.md, *.tex tables, fig_*.pdf/png

Example:
    python scripts/research/analyze_results.py --config configs/mlp_legacy_sources.yaml
"""
import ast
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from _common import PROJECT_ROOT, base_parser, setup  # noqa: E402

from self_pruning_network.experiment import collect_results  # noqa: E402
from self_pruning_network.stats import describe, holm, minimum_wilcoxon_pvalue, paired_comparison  # noqa: E402

PRINCIPAL = "gate_in@gated"
# Fixed categorical order (validated reference palette, light mode); entity -> colour never changes.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STYLE = {  # method -> (label, colour, marker, linestyle)
    "gate_in@gated": ("Gate-in (existing)", PALETTE[0], "o", "-"),
    "taylor@gated": ("Taylor", PALETTE[1], "s", "-"),
    "activation@gated": ("Activation", PALETTE[2], "^", "-"),
    "l1@gated": ("L1 in+out", PALETTE[3], "D", "-"),
    "l2@gated": ("L2 in+out", PALETTE[4], "v", "-"),
    "slimming@slimming": ("Net. Slimming", PALETTE[5], "P", "-"),
    "gate_combined@gated": ("Gate-combined (V2)", PALETTE[6], "X", "--"),
    "gate_out@gated": ("Gate-out (V1)", PALETTE[7], "h", "--"),
    "random@gated": ("Random", "#7a7974", "x", ":"),
}
CRITERION_LABEL = {"gate_in": "Gate-in", "gate_out": "Gate-out", "gate_combined": "Gate-comb.", "taylor": "Taylor",
                   "activation": "Activation", "l1": "L1 in+out", "l2": "L2 in+out", "l1_in": "L1 in",
                   "bn_gamma": "BN |γ|", "random": "Random", "slimming": "Slimming"}


def label(method: str) -> str:
    return STYLE.get(method, (method,))[0]


def setup_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "lines.linewidth": 1.6, "lines.markersize": 5,
        "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def save(fig, directories: list[Path], name: str) -> None:
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.pdf")
        fig.savefig(d / f"{name}.png")
    plt.close(fig)


# --------------------------------------------------------------------------- tables
def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["method", "criterion", "source", "allocation", "target_ratio", "ft_epochs"]
    for key, group in frame.groupby(keys, dropna=False):
        row = dict(zip(keys, key))
        for metric in ["test_accuracy", "validation_accuracy", "pre_ft_test_accuracy", "validation_loss",
                       "accuracy_change_vs_source_pp", "parameter_reduction_percent", "mac_reduction_percent",
                       "size_reduction_percent", "units_removed_percent", "fine_tune_time_s", "scoring_time_s"]:
            if metric in group:
                d = describe(group[metric].astype(float))
                row.update({f"{metric}_mean": d["mean"], f"{metric}_sd": d["sd"],
                            f"{metric}_ci95_low": d["ci95_low"], f"{metric}_ci95_high": d["ci95_high"]})
        row["n_seeds"] = int(group["seed"].nunique())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["ft_epochs", "target_ratio", "test_accuracy_mean"],
                                          ascending=[True, True, False])


def paired_tests(frame: pd.DataFrame, primary_ft: int) -> pd.DataFrame:
    data = frame[(frame["ft_epochs"] == primary_ft) & (frame["target_ratio"] > 0)]
    out = []
    for ratio, group in data.groupby("target_ratio"):
        pivot = group.pivot_table(index="seed", columns="method", values="test_accuracy")
        family = []
        if PRINCIPAL in pivot:
            for method in pivot.columns:
                if method == PRINCIPAL:
                    continue
                pair = pivot[[method, PRINCIPAL]].dropna()
                if len(pair) >= 2:
                    family.append((method, PRINCIPAL, pair))
        for method, reference, pair in family:
            result = paired_comparison(pair[method] * 100, pair[reference] * 100)
            out.append({"target_ratio": ratio, "method": method, "reference": reference, **result})
    tests = pd.DataFrame(out)
    if tests.empty:
        return tests
    tests["t_pvalue_holm"] = np.nan
    tests["wilcoxon_pvalue_holm"] = np.nan
    for ratio, idx in tests.groupby("target_ratio").groups.items():
        tests.loc[idx, "t_pvalue_holm"] = holm(tests.loc[idx, "t_pvalue"])
        tests.loc[idx, "wilcoxon_pvalue_holm"] = holm(tests.loc[idx, "wilcoxon_pvalue"])
    tests["min_attainable_wilcoxon_p"] = tests["n_pairs"].map(minimum_wilcoxon_pvalue)
    tests["statistically_supported_t_holm_0.05"] = tests["t_pvalue_holm"] < 0.05
    return tests


def fmt(mean, sd, scale=100.0, digits=2) -> str:
    if pd.isna(mean):
        return "--"
    return f"{mean * scale:.{digits}f}" + ("" if pd.isna(sd) else f" $\\pm$ {sd * scale:.{digits}f}")


def latex_main_table(summary: pd.DataFrame, primary_ft: int, methods: list[str], caption: str, tag: str) -> str:
    data = summary[summary["ft_epochs"] == primary_ft]
    ratios = sorted(r for r in data["target_ratio"].unique() if r > 0)
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{tab:" + tag + "}",
             r"\footnotesize", r"\begin{tabular}{l" + "c" * len(ratios) + "}", r"\toprule",
             "Method & " + " & ".join(f"{int(r * 100)}\\% units removed" for r in ratios) + r" \\", r"\midrule"]
    for method in methods:
        cells = []
        for r in ratios:
            row = data[(data["method"] == method) & (data["target_ratio"] == r)]
            cells.append(fmt(row["test_accuracy_mean"].iloc[0], row["test_accuracy_sd"].iloc[0]) if len(row) else "--")
        lines.append(label(method).replace("%", r"\%") + " & " + " & ".join(cells) + r" \\")
    controls = data[data["method"].str.startswith("control@")]
    for _, row in controls.iterrows():
        lines.append(r"\midrule" if _ == controls.index[0] else "")
        lines.append(f"Unpruned {row['source']} + FT & \\multicolumn{{{len(ratios)}}}{{c}}"
                     f"{{{fmt(row['test_accuracy_mean'], row['test_accuracy_sd'])}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(line for line in lines if line != "")


# --------------------------------------------------------------------------- figures
def fig_accuracy_curves(summary, primary_ft, x_metric, xlabel, name, dirs, methods):
    """Mean test accuracy with +/- 1 sample SD bars (CIs are in the tables)."""
    data = summary[(summary["ft_epochs"] == primary_ft) & (summary["allocation"] != "none")]
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    present = [m for m in methods if m in set(data["method"])]
    span = 1.0 if x_metric == "target_ratio" else 0.6
    for i, method in enumerate(present):
        rows = data[data["method"] == method].sort_values("target_ratio")
        lab, colour, marker, ls = STYLE[method]
        x = rows["target_ratio"] * 100 if x_metric == "target_ratio" else rows[f"{x_metric}_mean"]
        x = x + (i - (len(present) - 1) / 2) * span * 0.25
        y = rows["test_accuracy_mean"] * 100
        ax.errorbar(x, y, yerr=np.nan_to_num(rows["test_accuracy_sd"] * 100), label=lab, color=colour, marker=marker,
                    linestyle=ls, capsize=1.5, elinewidth=0.7, markersize=3.5, linewidth=1.2)
    ctl = summary[(summary["method"] == "control@gated") & (summary["ft_epochs"] == primary_ft)]
    if len(ctl):
        ax.axhline(ctl["test_accuracy_mean"].iloc[0] * 100, color="#52514e", linewidth=0.9, linestyle="--",
                   label="Unpruned gated + FT")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=6.5)
    save(fig, dirs, name)


def fig_seed_strip(frame, primary_ft, ratio, dirs, methods):
    data = frame[(frame["ft_epochs"] == primary_ft) & np.isclose(frame["target_ratio"], ratio)]
    present = [m for m in methods if m in set(data["method"])]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    seeds = sorted(data["seed"].unique())
    for i, method in enumerate(present):
        vals = data[data["method"] == method].set_index("seed")["test_accuracy"] * 100
        lab, colour, marker, _ = STYLE[method]
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), vals.values, color=colour, marker=marker,
                   s=18, zorder=3)
        ax.hlines(vals.mean(), i - 0.25, i + 0.25, color="#0b0b0b", linewidth=1.0)
    # matched-seed lines
    for seed in seeds:
        ys = [data[(data["method"] == m) & (data["seed"] == seed)]["test_accuracy"].mean() * 100 for m in present]
        ax.plot(range(len(present)), ys, color="#9a9994", linewidth=0.4, zorder=1)
    ax.set_xticks(range(len(present)), [label(m) for m in present], rotation=40, ha="right")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"{int(ratio * 100)}% units removed, per seed (bar = mean)")
    save(fig, dirs, f"fig_multiseed_r{int(ratio * 100)}")


def fig_recovery(frame, dirs, methods):
    ft = sorted(frame["ft_epochs"].unique())
    data = frame[frame["target_ratio"] > 0]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    if len(ft) > 1:
        r = sorted(data["target_ratio"].unique())[-1]
        for method in methods:
            rows = data[(data["method"] == method) & np.isclose(data["target_ratio"], r)]
            if rows["ft_epochs"].nunique() < 2:
                continue
            g = rows.groupby("ft_epochs")["test_accuracy"].mean() * 100
            lab, colour, marker, ls = STYLE[method]
            ax.plot(g.index, g.values, label=lab, color=colour, marker=marker, linestyle=ls)
        ax.set_xlabel("Fine-tuning epochs")
        ax.set_title(f"Recovery at {int(r * 100)}% units removed")
    else:
        rows = data[data["method"].isin(methods)].groupby(["method", "target_ratio"])[
            ["pre_ft_test_accuracy", "test_accuracy"]].mean().reset_index()
        for method in methods:
            sub = rows[rows["method"] == method]
            if sub.empty:
                continue
            lab, colour, marker, ls = STYLE[method]
            ax.plot(sub["target_ratio"] * 100, sub["pre_ft_test_accuracy"] * 100, color=colour, marker=marker,
                    linestyle=":", linewidth=1.0)
            ax.plot(sub["target_ratio"] * 100, sub["test_accuracy"] * 100, color=colour, marker=marker, linestyle=ls,
                    label=lab)
        ax.set_xlabel("Units removed (%)")
        ax.set_title(f"Before (dotted) and after (solid) {ft[0]} FT epoch(s)")
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=6.5)
    save(fig, dirs, "fig_finetune_recovery")


def fig_layerwise(frame, primary_ft, dirs):
    data = frame[(frame["ft_epochs"] == primary_ft) & (frame["target_ratio"] > 0)]
    r = sorted(data["target_ratio"].unique())[len(data["target_ratio"].unique()) // 2]
    data = data[np.isclose(data["target_ratio"], r)]
    chosen = data.drop_duplicates("method")
    chosen = chosen[chosen["method"].isin([PRINCIPAL, "slimming@slimming", "slimming@slimming:global"]) |
                    (chosen["allocation"] == "global")]
    if chosen.empty:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    n = len(chosen)
    for i, (_, row) in enumerate(chosen.iterrows()):
        values = ast.literal_eval(str(row["layer_removed_percent"]))
        x = np.arange(len(values)) + (i - (n - 1) / 2) * (0.8 / n)
        name = row["method"]
        colour = STYLE.get(name, (None, PALETTE[(i + 2) % 8]))[1]
        ax.bar(x, values, width=0.8 / n - 0.02, color=colour,
               label=STYLE.get(name, (name.replace("@slimming:global", " (global)"),))[0])
    ax.set_xlabel("Prunable layer index")
    ax.set_ylabel("Units removed (%)")
    ax.set_title(f"Layer-wise removal, target {int(r * 100)}% (seed {int(chosen['seed'].iloc[0])})")
    ax.legend(frameon=False)
    save(fig, dirs, "fig_layerwise_distribution")


def reliability(root: Path, dirs: list[Path]):
    paths = sorted(root.glob("seed*/reliability/correlations.csv"))
    if not paths:
        return None, None
    corr = pd.concat(pd.read_csv(p) for p in paths)
    mean_rows = corr[corr["group"] == "mean"].copy()
    summary = mean_rows.groupby(["source", "criterion"])["spearman"].agg(["mean", "std", "count"]).reset_index()
    stab_paths = sorted(root.glob("seed*/reliability/stability.csv"))
    stability = None
    if stab_paths:
        frames = [pd.read_csv(p) for p in stab_paths if p.stat().st_size > 2]
        frames = [f for f in frames if not f.empty]
        if frames:
            stability = pd.concat(frames).groupby(["source", "criterion"])["calibration_resample_spearman"].agg(
                ["mean", "std", "count"]).reset_index()
    # bar chart of mean Spearman
    sources = list(summary["source"].unique())
    # union of criteria over all sources (gate criteria exist only on gated-objective sources)
    order = summary.groupby("criterion")["mean"].max().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(max(3.4, 0.38 * len(order) + 0.8), 2.6))
    width = 0.8 / len(sources)
    for i, source in enumerate(sources):
        sub = summary[summary["source"] == source].set_index("criterion").reindex(order)
        x = np.arange(len(order)) + (i - (len(sources) - 1) / 2) * width
        ax.bar(x, sub["mean"], width=width - 0.02, yerr=sub["std"].fillna(0), capsize=1.5, color=PALETTE[i],
               label=f"{source} source", error_kw={"elinewidth": 0.7})
    ax.axhline(0, color="#0b0b0b", linewidth=0.6)
    ax.set_xticks(range(len(order)), [CRITERION_LABEL.get(c, c) for c in order], rotation=40, ha="right")
    ax.set_ylabel("Spearman ρ vs. damage")
    ax.legend(frameon=False)
    save(fig, dirs, "fig_reliability_spearman")
    # scatter for first seed, first source, layer 0
    first = paths[0].parent
    damage = pd.read_csv(first / "damage.csv")
    scores = torch.load(first / "scores.pt", weights_only=False)
    source = damage["source"].iloc[0]
    crits = [c for c in ["gate_in", "gate_out", "taylor", "activation", "l1", "random"] if f"{c}@{source}" in scores]
    fig, axes = plt.subplots(1, len(crits), figsize=(1.15 * len(crits) + 0.5, 1.9), sharey=True)
    axes = np.atleast_1d(axes)
    sub = damage[(damage["source"] == source) & (damage["group"] == 0)]
    for ax, crit in zip(axes, crits):
        s = scores[f"{crit}@{source}"][0][torch.tensor(sub["unit"].tolist())].numpy()
        ranks = pd.Series(s).rank(method="average").values  # Spearman is rank-based
        ax.scatter(ranks, sub["loss_increase"], s=6, color=PALETTE[0], alpha=0.8)
        rho = corr[(corr["seed"] == sub["seed"].iloc[0]) & (corr["source"] == source) & (corr["criterion"] == crit) &
                   (corr["group"].astype(str) == "0")]["spearman"]
        title = CRITERION_LABEL[crit] + (f" (ρ={rho.iloc[0]:.2f})" if len(rho) else "")
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("score rank", fontsize=6.5)
    axes[0].set_ylabel("Δ val. loss", fontsize=7)
    fig.suptitle(f"Score rank vs. single-unit removal damage — layer 1, {source} source, seed {sub['seed'].iloc[0]}",
                 fontsize=7.5, y=1.04)
    save(fig, dirs, "fig_score_vs_damage")
    # score distributions (layer 1, z-scored within criterion so shapes are comparable)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for i, crit in enumerate([c for c in ["gate_in", "gate_out", "taylor", "activation", "l1"] if f"{c}@{source}" in scores]):
        s = scores[f"{crit}@{source}"][0].numpy()
        cv = s.std() / abs(s.mean()) if s.mean() else np.nan
        z = (s - s.mean()) / (s.std() if s.std() > 0 else 1)
        ax.hist(z, bins=40, histtype="step", linewidth=1.2, color=PALETTE[i], label=f"{CRITERION_LABEL[crit]} (CV={cv:.1e})")
    ax.set_xlabel("Standardised score (layer 1)")
    ax.set_ylabel("Units")
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    save(fig, dirs, "fig_score_distribution")
    return summary, stability


def fig_benchmark(root: Path, dirs: list[Path]):
    path = root / "benchmark" / "benchmark_results.csv"
    if not path.exists():
        return None
    bench = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4))
    labels = [("Dense" if l == "dense" else "Gated (train-time)" if l.startswith("gated") else
               f"{int(r * 100)}% ({l.split(':')[0]})") for l, r in zip(bench["label"], bench["target_ratio"])]
    x = np.arange(len(bench))
    cols = [c for c in ["cpu_bs1_median_ms", "cpu_bs64_median_ms", "gpu_bs1_median_ms", "gpu_bs256_median_ms"] if c in bench]
    width = 0.8 / max(1, len(cols))
    for i, col in enumerate(cols):
        axes[0].bar(x + (i - (len(cols) - 1) / 2) * width, bench[col], width=width - 0.02, color=PALETTE[i],
                    label=col.replace("_median_ms", "").replace("_", " ").upper())
    axes[0].set_xticks(x, labels, rotation=40, ha="right")
    axes[0].set_ylabel("Median latency (ms)")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False)
    axes[1].bar(x, bench["deployable_size_bytes"] / 1e6, color=PALETTE[0], width=0.6)
    axes[1].set_xticks(x, labels, rotation=40, ha="right")
    axes[1].set_ylabel("Deployable state dict (MB)")
    save(fig, dirs, "fig_latency_and_size")
    return bench


def main() -> int:
    parser = base_parser(__doc__)
    args = parser.parse_args()
    config, _, _, log = setup(args)
    setup_matplotlib()
    root = Path(config["output_root"])
    name = config.get("name", root.name)
    out = root / "analysis"
    paper_fig = PROJECT_ROOT / "research" / "paper" / "figures" / name
    paper_tab = PROJECT_ROOT / "research" / "paper" / "tables" / name
    publish = not str(name).startswith("smoke")  # smoke outputs never enter the paper folder
    if not publish:
        paper_tab = out / "tables"
    dirs = [out / "plots", paper_fig] if publish else [out / "plots"]
    for d in [out, paper_tab]:
        d.mkdir(parents=True, exist_ok=True)
    frame = collect_results(config)
    if frame.empty:
        log("no results found")
        return 1
    primary_ft = int(config["fine_tune"]["epochs"])
    if primary_ft not in set(frame["ft_epochs"]):
        primary_ft = int(frame["ft_epochs"].mode().iloc[0])
    summary = summarize(frame)
    summary.to_csv(out / "summary.csv", index=False)
    tests = paired_tests(frame, primary_ft)
    tests.to_csv(out / "paired_tests.csv", index=False)
    methods = [m for m in STYLE if m in set(frame["method"])]
    n_seeds = frame["seed"].nunique()
    caption = (f"\\texttt{{{name.replace('_', chr(92) + '_')}}}: test accuracy (\\%, mean $\\pm$ sample SD over "
               f"{n_seeds} seeds) after {primary_ft} fine-tuning epoch(s). All criteria score the gated source; "
               f"identical compact architectures per column.")
    (paper_tab / "main_accuracy.tex").write_text(latex_main_table(summary, primary_ft, methods, caption,
                                                                  f"{name}_main"), encoding="utf-8")
    fig_accuracy_curves(summary, primary_ft, "target_ratio", "Hidden units removed (%)", "fig_accuracy_vs_ratio", dirs, methods)
    fig_accuracy_curves(summary, primary_ft, "parameter_reduction_percent", "Deployable parameter reduction (%)",
                        "fig_accuracy_vs_params", dirs, methods)
    fig_accuracy_curves(summary, primary_ft, "mac_reduction_percent", "MAC reduction (%)", "fig_accuracy_vs_macs",
                        dirs, methods)
    for ratio in sorted(r for r in frame["target_ratio"].unique() if r > 0):
        fig_seed_strip(frame, primary_ft, ratio, dirs, methods)
    fig_recovery(frame, dirs, [m for m in methods if m in {PRINCIPAL, "taylor@gated", "random@gated", "l1@gated",
                                                           "slimming@slimming", "gate_combined@gated"}])
    fig_layerwise(frame, primary_ft, dirs)
    rel, stab = reliability(root, dirs)
    if rel is not None:
        rel.to_csv(out / "reliability_summary.csv", index=False)
    if stab is not None:
        stab.to_csv(out / "stability_summary.csv", index=False)
    bench = fig_benchmark(root, dirs)
    gate_stats_path = root / "ablation_gate_statistics.csv"

    # ---------------- findings (descriptive + tested), all values computed above
    lines = [f"# Findings — {name}", "", f"Generated by `scripts/research/analyze_results.py` from `{root}/results.csv` "
             f"({len(frame)} rows, seeds {sorted(frame['seed'].unique().tolist())}, primary fine-tuning budget "
             f"{primary_ft} epoch(s)).", "",
             "## Test accuracy (mean ± SD, %)", ""]
    view = summary[summary["ft_epochs"] == primary_ft][["method", "target_ratio", "n_seeds", "test_accuracy_mean",
                                                        "test_accuracy_sd", "pre_ft_test_accuracy_mean",
                                                        "parameter_reduction_percent_mean", "mac_reduction_percent_mean"]]
    view = view.assign(**{c: view[c] * 100 for c in ["test_accuracy_mean", "test_accuracy_sd", "pre_ft_test_accuracy_mean"]})
    lines.append(view.round(3).to_markdown(index=False))
    if not tests.empty:
        lines += ["", f"## Paired comparisons vs `{PRINCIPAL}` (difference = method − gate-in, pp)", "",
                  f"Paired t-test and exact Wilcoxon signed-rank on {n_seeds} matched seeds; Holm correction within each "
                  f"ratio. Minimum attainable two-sided Wilcoxon p with n={n_seeds}: "
                  f"{minimum_wilcoxon_pvalue(n_seeds):.4f}. Shapiro–Wilk p on the differences is reported as a weak "
                  "check of the t-test normality assumption (low power at this n).", "",
                  tests[["target_ratio", "method", "n_pairs", "diff_mean", "diff_ci95_low", "diff_ci95_high", "wins",
                         "losses", "t_pvalue", "t_pvalue_holm", "wilcoxon_pvalue", "shapiro_pvalue",
                         "statistically_supported_t_holm_0.05"]].round(4).to_markdown(index=False)]
    if rel is not None:
        lines += ["", "## Score reliability: Spearman ρ between score and single-unit removal damage", "",
                  "Mean over layers per seed, then mean ± SD over seeds.", "", rel.round(3).to_markdown(index=False)]
    if stab is not None:
        lines += ["", "## Calibration-resampling stability (Spearman between scores from two disjoint calibration "
                  "subsets)", "", stab.round(4).to_markdown(index=False)]
    if gate_stats_path.exists():
        lines += ["", "## Gate statistics (ablation A)", "",
                  pd.read_csv(gate_stats_path).round(6).to_markdown(index=False)]
    if bench is not None:
        cols = [c for c in bench.columns if c.endswith("median_ms") or c in
                ("label", "target_ratio", "deployable_parameters", "macs", "deployable_size_bytes")]
        lines += ["", "## Measured efficiency (benchmark/benchmark_results.csv)", "", bench[cols].round(4).to_markdown(index=False)]
    lines += ["", "Statements of superiority require `statistically_supported_t_holm_0.05 == True`; otherwise differences "
              "are descriptive."]
    (out / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy(out / "findings.md", paper_tab / "findings.md")
    (out / "analysis_inputs.json").write_text(json.dumps({"results_rows": len(frame), "primary_ft": primary_ft,
                                                          "methods": sorted(frame["method"].unique().tolist())}, indent=2))
    log(f"analysis written to {out} and {paper_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
