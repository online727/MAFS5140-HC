from __future__ import annotations

import argparse
import itertools
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.append(str(TOOLS_DIR))

from utils import build_dataset_paths
from utils import finite_corr
from utils import load_feature_panel
from utils import resolve_dataset_split


HORIZONS = ("5min", "15min", "30min", "60min")
COMPOSITE_WEIGHTS = {
    "sharpe_score": 0.35,
    "return_score": 0.20,
    "avg_rankicir_score": 0.25,
    "avg_icir_score": 0.10,
    "drawdown_score": 0.10,
}


def load_summary(summary_path: str | Path) -> pd.DataFrame:
    summary_path = Path(summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Cannot find summary file: {summary_path}")

    df = pd.read_csv(summary_path)
    required = {
        "feature",
        "variant",
        "direction",
        "family",
        "feature_path",
        "weights_path",
        "cumulative_return",
        "sharpe_ratio",
        "max_drawdown",
    }
    for horizon in HORIZONS:
        required.update(
            {
                f"icir_{horizon}",
                f"rankicir_{horizon}",
                f"ic_{horizon}",
                f"rankic_{horizon}",
            }
        )

    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Summary is missing required columns: {missing}")

    return df.replace([np.inf, -np.inf], np.nan)


def percentile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan)
    if clean.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)

    score = clean.rank(pct=True, ascending=higher_is_better)
    return score.fillna(0.0)


def add_derived_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    icir_cols = [f"icir_{horizon}" for horizon in HORIZONS]
    rankicir_cols = [f"rankicir_{horizon}" for horizon in HORIZONS]
    ic_cols = [f"ic_{horizon}" for horizon in HORIZONS]
    rankic_cols = [f"rankic_{horizon}" for horizon in HORIZONS]

    out["avg_icir"] = out[icir_cols].mean(axis=1)
    out["min_icir"] = out[icir_cols].min(axis=1)
    out["positive_icir_horizons"] = (out[icir_cols] > 0.0).sum(axis=1)
    out["avg_rankicir"] = out[rankicir_cols].mean(axis=1)
    out["min_rankicir"] = out[rankicir_cols].min(axis=1)
    out["positive_rankicir_horizons"] = (out[rankicir_cols] > 0.0).sum(axis=1)
    out["avg_ic"] = out[ic_cols].mean(axis=1)
    out["avg_rankic"] = out[rankic_cols].mean(axis=1)
    out["abs_max_drawdown"] = out["max_drawdown"].abs()

    out["sharpe_score"] = percentile_score(out["sharpe_ratio"])
    out["return_score"] = percentile_score(out["cumulative_return"])
    out["avg_rankicir_score"] = percentile_score(out["avg_rankicir"])
    out["avg_icir_score"] = percentile_score(out["avg_icir"])
    out["drawdown_score"] = percentile_score(out["max_drawdown"])
    out["risk_adjusted_score"] = 0.70 * out["sharpe_score"] + 0.30 * out["drawdown_score"]

    out["composite_score"] = sum(out[col] * weight for col, weight in COMPOSITE_WEIGHTS.items())
    out["rank_composite"] = out["composite_score"].rank(ascending=False, method="min").astype(int)
    out["rank_sharpe"] = out["sharpe_ratio"].rank(ascending=False, method="min").astype(int)
    out["rank_return"] = out["cumulative_return"].rank(ascending=False, method="min").astype(int)
    out["rank_avg_rankicir"] = out["avg_rankicir"].rank(ascending=False, method="min").astype(int)
    out["rank_avg_icir"] = out["avg_icir"].rank(ascending=False, method="min").astype(int)
    out["rank_drawdown"] = out["max_drawdown"].rank(ascending=False, method="min").astype(int)

    return out.sort_values("composite_score", ascending=False).reset_index(drop=True)


def write_top_by_metric(scores: pd.DataFrame, output_path: Path, top_n: int) -> pd.DataFrame:
    metric_specs = [
        ("sharpe_ratio", False, "best_by_sharpe"),
        ("cumulative_return", False, "best_by_return"),
        ("avg_icir", False, "best_by_avg_icir"),
        ("avg_rankicir", False, "best_by_avg_rankicir"),
        ("max_drawdown", False, "best_by_drawdown"),
        ("composite_score", False, "best_by_composite"),
    ]

    rows = []
    for metric, ascending, label in metric_specs:
        ranked = scores.sort_values(metric, ascending=ascending).head(top_n)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            rows.append(
                {
                    "metric_set": label,
                    "selection_rank": rank,
                    "metric": metric,
                    "metric_value": row[metric],
                    "variant": row["variant"],
                    "feature": row["feature"],
                    "direction": row["direction"],
                    "family": row["family"],
                    "composite_score": row["composite_score"],
                    "sharpe_ratio": row["sharpe_ratio"],
                    "cumulative_return": row["cumulative_return"],
                    "max_drawdown": row["max_drawdown"],
                    "avg_icir": row["avg_icir"],
                    "avg_rankicir": row["avg_rankicir"],
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv(output_path, index=False)
    return table


def add_selection(rows: list[dict[str, object]], set_name: str, frame: pd.DataFrame, reason: str) -> None:
    for rank, (_, row) in enumerate(frame.iterrows(), start=1):
        rows.append(
            {
                "set_name": set_name,
                "selection_rank": rank,
                "reason": reason,
                "variant": row["variant"],
                "feature": row["feature"],
                "direction": row["direction"],
                "family": row["family"],
                "composite_score": row["composite_score"],
                "sharpe_ratio": row["sharpe_ratio"],
                "cumulative_return": row["cumulative_return"],
                "max_drawdown": row["max_drawdown"],
                "avg_icir": row["avg_icir"],
                "avg_rankicir": row["avg_rankicir"],
                "positive_rankicir_horizons": row["positive_rankicir_horizons"],
                "feature_path": row["feature_path"],
                "weights_path": row["weights_path"],
            }
        )


def load_flattened_signal(path: str | Path, direction: str = "raw") -> np.ndarray | None:
    path = Path(path)
    if not path.exists():
        return None

    try:
        frame = load_feature_panel(path, direction=direction)
    except Exception:
        return None

    arr = frame.to_numpy(dtype=float, copy=False).reshape(-1)
    return arr


def find_redundancy_pairs(
    candidates: pd.DataFrame,
    feature_corr_threshold: float,
    weights_corr_threshold: float,
    output_path: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    feature_arrays: dict[str, np.ndarray] = {}
    weight_arrays: dict[str, np.ndarray] = {}

    for _, row in candidates.iterrows():
        variant = row["variant"]
        feature_arr = load_flattened_signal(row["feature_path"], direction=row["direction"])
        weight_arr = load_flattened_signal(row["weights_path"], direction="raw")
        if feature_arr is not None:
            feature_arrays[variant] = feature_arr
        if weight_arr is not None:
            weight_arrays[variant] = weight_arr

    rows = []
    for left, right in itertools.combinations(candidates["variant"], 2):
        feature_corr = np.nan
        weight_corr = np.nan
        if left in feature_arrays and right in feature_arrays:
            feature_corr = finite_corr(feature_arrays[left], feature_arrays[right])
        if left in weight_arrays and right in weight_arrays:
            weight_corr = finite_corr(weight_arrays[left], weight_arrays[right])

        feature_hit = pd.notna(feature_corr) and abs(feature_corr) >= feature_corr_threshold
        weight_hit = pd.notna(weight_corr) and abs(weight_corr) >= weights_corr_threshold
        if feature_hit or weight_hit:
            rows.append(
                {
                    "left_variant": left,
                    "right_variant": right,
                    "feature_corr": feature_corr,
                    "weight_corr": weight_corr,
                    "reason": ",".join(
                        reason
                        for reason, hit in (
                            ("feature_corr", feature_hit),
                            ("weight_corr", weight_hit),
                        )
                        if hit
                    ),
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv(output_path, index=False)
    return table, feature_arrays, weight_arrays


def too_correlated(
    candidate: pd.Series,
    selected: pd.DataFrame,
    feature_arrays: dict[str, np.ndarray],
    weight_arrays: dict[str, np.ndarray],
    feature_corr_threshold: float,
    weights_corr_threshold: float,
) -> bool:
    candidate_name = candidate["variant"]
    for _, existing in selected.iterrows():
        existing_name = existing["variant"]

        if candidate["feature"] == existing["feature"]:
            return True

        if candidate_name in feature_arrays and existing_name in feature_arrays:
            corr = finite_corr(feature_arrays[candidate_name], feature_arrays[existing_name])
            if pd.notna(corr) and abs(corr) >= feature_corr_threshold:
                return True

        if candidate_name in weight_arrays and existing_name in weight_arrays:
            corr = finite_corr(weight_arrays[candidate_name], weight_arrays[existing_name])
            if pd.notna(corr) and abs(corr) >= weights_corr_threshold:
                return True

    return False


def build_diversified_set(
    scores: pd.DataFrame,
    feature_arrays: dict[str, np.ndarray],
    weight_arrays: dict[str, np.ndarray],
    feature_corr_threshold: float,
    weights_corr_threshold: float,
    top_n: int,
) -> pd.DataFrame:
    eligible = scores[
        (scores["rankicir_5min"] > 0.0)
        & (scores["avg_rankicir"] > 0.0)
        & (scores["cumulative_return"] >= 0.0)
    ].sort_values("composite_score", ascending=False)

    selected_rows = []
    selected = pd.DataFrame()

    covered_families: set[str] = set()
    target_families = set(eligible["family"].dropna().unique())

    for _, row in eligible.iterrows():
        if len(selected_rows) >= top_n or covered_families == target_families:
            break
        if row["family"] in covered_families:
            continue
        if not selected.empty and too_correlated(
            row,
            selected,
            feature_arrays,
            weight_arrays,
            feature_corr_threshold,
            weights_corr_threshold,
        ):
            continue
        selected_rows.append(row)
        covered_families.add(row["family"])
        selected = pd.DataFrame(selected_rows)

    for _, row in eligible.iterrows():
        if len(selected_rows) >= top_n:
            break
        if row["variant"] in set(selected["variant"]):
            continue
        if too_correlated(
            row,
            selected,
            feature_arrays,
            weight_arrays,
            feature_corr_threshold,
            weights_corr_threshold,
        ):
            continue
        selected_rows.append(row)
        selected = pd.DataFrame(selected_rows)

    return pd.DataFrame(selected_rows).head(top_n)


def build_selected_sets(
    scores: pd.DataFrame,
    diversified: pd.DataFrame,
    output_path: Path,
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    add_selection(
        rows,
        "best_by_sharpe",
        scores.sort_values("sharpe_ratio", ascending=False).head(top_n),
        "Highest Sharpe ratio.",
    )
    add_selection(
        rows,
        "best_by_rankicir",
        scores.sort_values("avg_rankicir", ascending=False).head(top_n),
        "Highest average RankICIR across 5/15/30/60min horizons.",
    )
    add_selection(
        rows,
        "best_by_icir",
        scores.sort_values("avg_icir", ascending=False).head(top_n),
        "Highest average Pearson ICIR across horizons.",
    )

    robust = scores[
        (scores["positive_rankicir_horizons"] >= 3)
        & (scores["rankicir_5min"] > 0.0)
        & (scores["avg_rankicir"] > 0.0)
        & (scores["cumulative_return"] >= 0.0)
    ].sort_values("composite_score", ascending=False)
    add_selection(
        rows,
        "robust_multi_horizon",
        robust.head(top_n),
        "Positive short-horizon RankICIR and positive average multi-horizon RankICIR.",
    )

    low_drawdown = scores[
        (scores["sharpe_ratio"] > 0.0) & (scores["cumulative_return"] >= 0.0)
    ].sort_values(["risk_adjusted_score", "composite_score"], ascending=False)
    add_selection(
        rows,
        "low_drawdown",
        low_drawdown.head(top_n),
        "Strong risk-adjusted score using Sharpe and drawdown.",
    )

    per_family = max(1, int(np.ceil(top_n / max(scores["family"].nunique(), 1))))
    family_balanced = (
        scores.sort_values("composite_score", ascending=False)
        .groupby("family", group_keys=False)
        .head(per_family)
        .sort_values("composite_score", ascending=False)
        .head(top_n)
    )
    add_selection(
        rows,
        "family_balanced",
        family_balanced,
        "Top composite candidates with explicit family diversification.",
    )

    add_selection(
        rows,
        "composite_best",
        scores.sort_values("composite_score", ascending=False).head(top_n),
        "Weighted composite score using PnL, IC, RankIC, and drawdown.",
    )
    add_selection(
        rows,
        "diversified_best",
        diversified,
        "Composite-ranked candidates after duplicate and correlation pruning.",
    )

    table = pd.DataFrame(rows)
    table.to_csv(output_path, index=False)
    return table


def write_family_summary(scores: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    table = (
        scores.groupby("family")
        .agg(
            count=("variant", "count"),
            mean_sharpe=("sharpe_ratio", "mean"),
            median_sharpe=("sharpe_ratio", "median"),
            max_sharpe=("sharpe_ratio", "max"),
            mean_return=("cumulative_return", "mean"),
            mean_avg_icir=("avg_icir", "mean"),
            mean_avg_rankicir=("avg_rankicir", "mean"),
            max_composite=("composite_score", "max"),
        )
        .sort_values("max_composite", ascending=False)
        .reset_index()
    )
    table.to_csv(output_path, index=False)
    return table


def write_horizon_summary(scores: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        rows.append(
            {
                "horizon": horizon,
                "mean_ic": scores[f"ic_{horizon}"].mean(),
                "median_ic": scores[f"ic_{horizon}"].median(),
                "mean_icir": scores[f"icir_{horizon}"].mean(),
                "median_icir": scores[f"icir_{horizon}"].median(),
                "mean_rankic": scores[f"rankic_{horizon}"].mean(),
                "median_rankic": scores[f"rankic_{horizon}"].median(),
                "mean_rankicir": scores[f"rankicir_{horizon}"].mean(),
                "median_rankicir": scores[f"rankicir_{horizon}"].median(),
                "positive_rankicir_ratio": (scores[f"rankicir_{horizon}"] > 0.0).mean(),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(output_path, index=False)
    return table


def annotate_top(ax: plt.Axes, data: pd.DataFrame, x_col: str, y_col: str, n: int = 8) -> None:
    for _, row in data.sort_values("composite_score", ascending=False).head(n).iterrows():
        ax.annotate(
            row["variant"],
            (row[x_col], row[y_col]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )


def save_scatter(scores: pd.DataFrame, output_path: Path, x_col: str, y_col: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    families = sorted(scores["family"].dropna().unique())
    cmap = plt.get_cmap("tab10")
    for idx, family in enumerate(families):
        part = scores[scores["family"] == family]
        ax.scatter(part[x_col], part[y_col], s=35, alpha=0.75, label=family, color=cmap(idx % 10))
    annotate_top(ax, scores, x_col, y_col)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_family_boxplot(scores: pd.DataFrame, output_path: Path, metric: str, title: str) -> None:
    families = sorted(scores["family"].dropna().unique())
    data = [scores.loc[scores["family"] == family, metric].dropna().to_numpy() for family in families]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, tick_labels=families, showfliers=False)
    ax.set_title(title)
    ax.set_xlabel("family")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_heatmap(
    frame: pd.DataFrame,
    output_path: Path,
    title: str,
    x_labels: Iterable[str] | None = None,
    y_labels: Iterable[str] | None = None,
    cmap: str = "RdYlGn",
) -> None:
    values = frame.to_numpy(dtype=float)
    fig_width = max(8, min(18, 0.35 * values.shape[1] + 5))
    fig_height = max(6, min(18, 0.28 * values.shape[0] + 3))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_xticklabels(list(x_labels) if x_labels is not None else frame.columns, rotation=45, ha="right")
    ax.set_yticklabels(list(y_labels) if y_labels is not None else frame.index, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_top_composite_bar(scores: pd.DataFrame, output_path: Path, top_n: int) -> None:
    top = scores.sort_values("composite_score", ascending=False).head(top_n).iloc[::-1]
    components = list(COMPOSITE_WEIGHTS)
    weighted = top[components].mul(pd.Series(COMPOSITE_WEIGHTS), axis=1)

    fig, ax = plt.subplots(figsize=(12, max(7, 0.32 * len(top))))
    left = np.zeros(len(top))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]
    y = np.arange(len(top))
    for idx, component in enumerate(components):
        ax.barh(y, weighted[component], left=left, label=component, color=colors[idx], alpha=0.85)
        left += weighted[component].to_numpy()
    ax.set_yticks(y)
    ax.set_yticklabels(top["variant"], fontsize=8)
    ax.set_xlabel("weighted composite score")
    ax.set_title("Top Composite Score Breakdown")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_plots(scores: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    save_scatter(
        scores,
        output_dir / "sharpe_vs_rankicir.png",
        x_col="sharpe_ratio",
        y_col="avg_rankicir",
        title="Sharpe vs Average RankICIR",
    )
    save_scatter(
        scores,
        output_dir / "return_vs_drawdown.png",
        x_col="max_drawdown",
        y_col="cumulative_return",
        title="Return vs Drawdown",
    )
    save_family_boxplot(
        scores,
        output_dir / "family_boxplot_sharpe.png",
        metric="sharpe_ratio",
        title="Sharpe Distribution by Family",
    )
    save_family_boxplot(
        scores,
        output_dir / "family_boxplot_rankicir.png",
        metric="avg_rankicir",
        title="Average RankICIR Distribution by Family",
    )

    top = scores.sort_values("composite_score", ascending=False).head(top_n)
    rankicir_frame = top.set_index("variant")[[f"rankicir_{horizon}" for horizon in HORIZONS]]
    rankicir_frame.columns = list(HORIZONS)
    save_heatmap(
        rankicir_frame,
        output_dir / "horizon_rankicir_heatmap.png",
        title="Top Variants RankICIR by Horizon",
    )

    metric_cols = [
        "cumulative_return",
        "sharpe_ratio",
        "max_drawdown",
        "avg_icir",
        "avg_rankicir",
        "min_rankicir",
        "positive_rankicir_horizons",
        "composite_score",
    ]
    metric_corr = scores[metric_cols].corr()
    save_heatmap(
        metric_corr,
        output_dir / "metric_corr_heatmap.png",
        title="Metric Correlation Heatmap",
        cmap="coolwarm",
    )
    save_top_composite_bar(scores, output_dir / "top_composite_bar.png", top_n=min(top_n, 30))


def write_report(
    scores: pd.DataFrame,
    selected_sets: pd.DataFrame,
    family_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    redundancy_pairs: pd.DataFrame,
    output_path: Path,
    top_n: int,
) -> None:
    top_composite = scores.sort_values("composite_score", ascending=False).head(10)
    diversified = selected_sets[selected_sets["set_name"] == "diversified_best"].head(top_n)

    lines = [
        "# Feature Analysis Report",
        "",
        "## Data",
        f"- Variants analyzed: {len(scores)}",
        f"- Unique raw features: {scores['feature'].nunique()}",
        f"- Families: {', '.join(sorted(scores['family'].dropna().unique()))}",
        "",
        "## Best Composite Candidates",
        top_composite[
            [
                "variant",
                "family",
                "composite_score",
                "sharpe_ratio",
                "cumulative_return",
                "max_drawdown",
                "avg_rankicir",
                "positive_rankicir_horizons",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Recommended Diversified Set",
        diversified[
            [
                "selection_rank",
                "variant",
                "family",
                "composite_score",
                "sharpe_ratio",
                "cumulative_return",
                "avg_rankicir",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Family Summary",
        family_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Horizon Summary",
        horizon_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Redundancy",
        f"- Highly correlated candidate pairs found: {len(redundancy_pairs)}",
        "- Redundancy is computed on the top composite candidate pool to keep analysis tractable.",
        "",
        "## Output Files",
        "- `feature_scores.csv`",
        "- `top_by_metric.csv`",
        "- `selected_feature_sets.csv`",
        "- `family_summary.csv`",
        "- `horizon_summary.csv`",
        "- `redundancy_pairs.csv`",
        "- `*.png` visualizations",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(
    data_path: str | Path | None,
    dataset: str | None,
    split: str | None,
    top_n: int,
    feature_corr_threshold: float,
    weights_corr_threshold: float,
    redundancy_pool_size: int | None = None,
) -> dict[str, Path]:
    dataset, split = resolve_dataset_split(data_path=data_path, dataset=dataset, split=split)
    paths = build_dataset_paths(dataset=dataset, split=split)
    summary_path = paths.mom_summary_path
    output_dir = paths.feat_ana_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(summary_path)
    scores = add_derived_scores(summary)

    feature_scores_path = output_dir / "feature_scores.csv"
    scores.to_csv(feature_scores_path, index=False)

    top_by_metric_path = output_dir / "top_by_metric.csv"
    write_top_by_metric(scores, top_by_metric_path, top_n=top_n)

    pool_size = redundancy_pool_size or max(top_n * 3, 60)
    redundancy_candidates = scores.sort_values("composite_score", ascending=False).head(pool_size)
    redundancy_pairs_path = output_dir / "redundancy_pairs.csv"
    redundancy_pairs, feature_arrays, weight_arrays = find_redundancy_pairs(
        redundancy_candidates,
        feature_corr_threshold=feature_corr_threshold,
        weights_corr_threshold=weights_corr_threshold,
        output_path=redundancy_pairs_path,
    )

    diversified = build_diversified_set(
        scores,
        feature_arrays=feature_arrays,
        weight_arrays=weight_arrays,
        feature_corr_threshold=feature_corr_threshold,
        weights_corr_threshold=weights_corr_threshold,
        top_n=top_n,
    )

    selected_sets_path = output_dir / "selected_feature_sets.csv"
    selected_sets = build_selected_sets(scores, diversified, selected_sets_path, top_n=top_n)

    family_summary_path = output_dir / "family_summary.csv"
    family_summary = write_family_summary(scores, family_summary_path)

    horizon_summary_path = output_dir / "horizon_summary.csv"
    horizon_summary = write_horizon_summary(scores, horizon_summary_path)

    write_plots(scores, output_dir=output_dir, top_n=top_n)

    report_path = output_dir / "analysis_report.md"
    write_report(
        scores=scores,
        selected_sets=selected_sets,
        family_summary=family_summary,
        horizon_summary=horizon_summary,
        redundancy_pairs=redundancy_pairs,
        output_path=report_path,
        top_n=top_n,
    )

    return {
        "feature_scores": feature_scores_path,
        "top_by_metric": top_by_metric_path,
        "selected_feature_sets": selected_sets_path,
        "family_summary": family_summary_path,
        "horizon_summary": horizon_summary_path,
        "redundancy_pairs": redundancy_pairs_path,
        "analysis_report": report_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze momentum feature evaluation summaries.")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to source data; used to infer dataset/split and default input/output dirs.",
    )
    parser.add_argument("--dataset", default=None, help="Dataset name. Defaults to parent dir of --data-path.")
    parser.add_argument("--split", default=None, help="Split name. Defaults to stem of --data-path.")
    parser.add_argument("--top-n", type=int, default=15, help="Number of rows per ranked feature set.")
    parser.add_argument("--feature-corr-threshold", type=float, default=0.85)
    parser.add_argument("--weights-corr-threshold", type=float, default=0.85)
    parser.add_argument(
        "--redundancy-pool-size",
        type=int,
        default=None,
        help="Number of top composite variants used for pairwise redundancy checks.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = run_analysis(
        data_path=args.data_path,
        dataset=args.dataset,
        split=args.split,
        top_n=args.top_n,
        feature_corr_threshold=args.feature_corr_threshold,
        weights_corr_threshold=args.weights_corr_threshold,
        redundancy_pool_size=args.redundancy_pool_size,
    )

    print("Feature analysis outputs:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
