from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.append(str(TOOLS_DIR))

from utils import build_dataset_paths
from utils import build_top_score_weights
from utils import cross_sectional_zscore
from utils import extract_close_prices
from utils import load_feature_panel
from utils import load_selected_feature_set
from utils import load_tabular_data
from utils import resolve_dataset_split
from utils import save_tabular_data
from utils import validate_weight_frame

from backtest import run_backtest


DEFAULT_TOP_SELECTOR = 10.0
DEFAULT_MAX_WEIGHT = 0.1
DEFAULT_PERIODS_PER_YEAR = 252 * 78


def available_set_names(selected_features_path: str | Path) -> list[str]:
    path = Path(selected_features_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find selected feature sets file: {path}")
    selected = pd.read_csv(path)
    if "set_name" not in selected.columns:
        raise ValueError("selected_feature_sets.csv must contain a set_name column.")
    return sorted(selected["set_name"].dropna().unique())


def build_combined_score(
    selected_features: pd.DataFrame,
    target_index: pd.Index | None = None,
    target_columns: pd.Index | None = None,
) -> pd.DataFrame:
    components: list[pd.DataFrame] = []

    for _, row in selected_features.iterrows():
        panel = load_feature_panel(
            row["feature_path"],
            direction=row["direction"],
            target_index=target_index,
            target_columns=target_columns,
        )
        components.append(cross_sectional_zscore(panel))

    if not components:
        raise ValueError("No feature panels loaded.")

    base_index = components[0].index
    base_columns = components[0].columns
    aligned = [
        component.reindex(index=base_index, columns=base_columns).fillna(0.0)
        for component in components
    ]
    score = sum(aligned) / len(aligned)
    return score.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def generate_set_weights(
    selected_features_path: str | Path,
    set_name: str,
    top_selector: float,
    max_weight: float,
    feature_limit: int | None,
    target_index: pd.Index | None,
    target_columns: pd.Index | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = load_selected_feature_set(
        selected_features_path=selected_features_path,
        set_name=set_name,
        feature_limit=feature_limit,
    )
    score = build_combined_score(
        selected_features=selected,
        target_index=target_index,
        target_columns=target_columns,
    )
    weights = build_top_score_weights(
        scores=score,
        top_selector=top_selector,
        max_weight=max_weight,
    )
    validate_weight_frame(weights, max_weight=max_weight)
    return score, weights, selected


def run_strategy_ver2(
    data_path: str | Path | None,
    dataset: str | None,
    split: str | None,
    set_name: str,
    top_selector: float,
    max_weight: float,
    feature_limit: int | None,
    run_backtests: bool = True,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> pd.DataFrame:
    dataset, split = resolve_dataset_split(data_path=data_path, dataset=dataset, split=split)
    paths = build_dataset_paths(dataset=dataset, split=split)

    target_index = target_columns = None
    if data_path is not None:
        market_data = load_tabular_data(data_path)
        close = extract_close_prices(market_data)
        target_index = close.index
        target_columns = close.columns

    if set_name == "all":
        set_names = available_set_names(paths.selected_feature_sets_path)
    else:
        set_names = [set_name]

    rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for current_set in set_names:
        print(f"Generating strategy_ver2 weights for set: {current_set}")
        score, weights, selected = generate_set_weights(
            selected_features_path=paths.selected_feature_sets_path,
            set_name=current_set,
            top_selector=top_selector,
            max_weight=max_weight,
            feature_limit=feature_limit,
            target_index=target_index,
            target_columns=target_columns,
        )

        score_path = paths.strategy_ver2_score_dir / f"{current_set}.parquet"
        weights_path = paths.strategy_ver2_weights_dir / f"{current_set}.parquet"
        save_tabular_data(score, score_path)
        save_tabular_data(weights, weights_path)

        backtest_metrics_path = ""
        returns_path = ""
        equity_curve_path = ""
        if run_backtests:
            if data_path is None:
                raise ValueError("--data-path is required when backtests are enabled.")
            print(f"Backtesting strategy_ver2 weights for set: {current_set}")
            equity_curve_path = str(paths.strategy_ver2_backtest_dir / f"equity_curve_{current_set}.png")
            returns_path = str(paths.strategy_ver2_backtest_dir / f"portfolio_returns_{current_set}.csv")
            backtest_metrics_path = str(paths.strategy_ver2_backtest_dir / f"metrics_{current_set}.csv")

            result = run_backtest(
                data_path=data_path,
                signal=weights,
                periods_per_year=periods_per_year,
                plot_path=equity_curve_path,
            )
            save_tabular_data(result["portfolio_returns"].to_frame(), returns_path)
            metric_row = {
                "dataset": dataset,
                "split": split,
                "set_name": current_set,
                **result["metrics"],
                "weights_path": str(weights_path),
                "score_path": str(score_path),
                "portfolio_returns_path": returns_path,
                "equity_curve_path": equity_curve_path,
            }
            metric_rows.append(metric_row)
            save_tabular_data(pd.DataFrame([metric_row]), backtest_metrics_path)

        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "set_name": current_set,
                "feature_count": len(selected),
                "top_selector": top_selector,
                "max_weight": max_weight,
                "score_path": str(score_path),
                "weights_path": str(weights_path),
                "first_timestamp": score.index.min(),
                "last_timestamp": score.index.max(),
                "n_timestamps": len(score),
                "n_assets": len(score.columns),
                "max_row_weight_sum": float(weights.sum(axis=1).max()),
                "max_single_weight": float(weights.max(axis=1).max()),
                "portfolio_returns_path": returns_path,
                "equity_curve_path": equity_curve_path,
                "backtest_metrics_path": backtest_metrics_path,
            }
        )

    manifest = pd.DataFrame(rows)
    save_tabular_data(manifest, paths.strategy_ver2_manifest_path)
    if run_backtests and metric_rows:
        metric_rows = pd.DataFrame(metric_rows).sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
        print(metric_rows[["set_name", "cumulative_return", "sharpe_ratio", "annualized_return", "annualized_volatility", "max_drawdown"]])
        save_tabular_data(
            metric_rows,
            paths.strategy_ver2_backtest_dir / "metrics_summary.csv",
        )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate v2 feature-set momentum weights.")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to market data; used to infer dataset/split and align output panels.",
    )
    parser.add_argument("--dataset", default=None, help="Dataset name. Defaults to parent dir of --data-path.")
    parser.add_argument("--split", default=None, help="Split name. Defaults to stem of --data-path.")
    parser.add_argument("--set-name", default="diversified_best", help="Feature set name, or 'all'.")
    parser.add_argument("--top-selector", type=float, default=DEFAULT_TOP_SELECTOR)
    parser.add_argument("--max-weight", type=float, default=DEFAULT_MAX_WEIGHT)
    parser.add_argument("--feature-limit", type=int, default=None)
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Only generate scores/weights; do not run vectorized backtests.",
    )
    parser.add_argument(
        "--periods-per-year",
        type=int,
        default=DEFAULT_PERIODS_PER_YEAR,
        help="Annualization factor used for integrated backtests.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = run_strategy_ver2(
        data_path=args.data_path,
        dataset=args.dataset,
        split=args.split,
        set_name=args.set_name,
        top_selector=args.top_selector,
        max_weight=args.max_weight,
        feature_limit=args.feature_limit,
        run_backtests=not args.skip_backtest,
        periods_per_year=args.periods_per_year,
    )
    paths = build_dataset_paths(manifest["dataset"].iloc[0], manifest["split"].iloc[0])
    print(f"Saved strategy_ver2 manifest with {len(manifest)} rows to {paths.strategy_ver2_manifest_path}")
    if not args.skip_backtest:
        print(f"Saved strategy_ver2 backtest metrics to {paths.strategy_ver2_backtest_dir / 'metrics_summary.csv'}")


if __name__ == "__main__":
    main()
