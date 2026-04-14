from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPSILON = 1e-6


def load_tabular_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_parquet(path, engine="fastparquet")

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)

    raise ValueError(f"Unsupported file format: {path}")


def extract_close_prices(market_data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(market_data.index, pd.DatetimeIndex):
        if "datetime" in market_data.columns:
            market_data = market_data.copy()
            market_data["datetime"] = pd.to_datetime(market_data["datetime"])
            market_data = market_data.set_index("datetime")
        else:
            raise ValueError("Market data must have a DatetimeIndex or a 'datetime' column.")

    if isinstance(market_data.columns, pd.MultiIndex):
        fields = market_data.columns.get_level_values(-1)
        if "close" not in fields:
            raise ValueError("MultiIndex market data must include a 'close' field.")
        close_prices = market_data.xs("close", axis=1, level=-1)
    else:
        close_prices = market_data.copy()

    close_prices = close_prices.sort_index().astype(float)

    if close_prices.isna().any().any():
        raise ValueError("Market close prices contain NaN values.")

    return close_prices


def prepare_signal_frame(
    signal: pd.DataFrame | str | Path,
    target_index: pd.DatetimeIndex,
    target_columns: pd.Index,
) -> pd.DataFrame:
    if isinstance(signal, (str, Path)):
        signal_df = load_tabular_data(signal)
    else:
        signal_df = signal.copy()

    if isinstance(signal_df, pd.Series):
        signal_df = signal_df.to_frame().T

    if not isinstance(signal_df, pd.DataFrame):
        raise TypeError("Signal must be a pandas DataFrame, Series, or a file path.")

    if not isinstance(signal_df.index, pd.DatetimeIndex):
        if "datetime" in signal_df.columns:
            signal_df = signal_df.copy()
            signal_df["datetime"] = pd.to_datetime(signal_df["datetime"])
            signal_df = signal_df.set_index("datetime")
        else:
            raise ValueError("Signal must have a DatetimeIndex or a 'datetime' column.")

    if isinstance(signal_df.columns, pd.MultiIndex):
        raise ValueError("Signal columns must be ticker names, not a MultiIndex.")

    signal_df = signal_df.sort_index()

    if signal_df.index.has_duplicates:
        raise ValueError("Signal index contains duplicate timestamps.")

    signal_df.columns = pd.Index(signal_df.columns.astype(str))
    target_columns = pd.Index(target_columns.astype(str))

    unknown_tickers = signal_df.columns.difference(target_columns)
    if len(unknown_tickers) > 0:
        raise ValueError(
            f"Signal contains unknown tickers: {unknown_tickers.tolist()[:10]}"
        )

    signal_df = signal_df.reindex(columns=target_columns, fill_value=0.0)
    signal_df = signal_df.reindex(target_index).ffill().fillna(0.0)
    signal_df = signal_df.astype(float)

    return signal_df


def validate_weights(weights: pd.DataFrame) -> None:
    negative_mask = weights < -EPSILON
    if negative_mask.any().any():
        timestamp = negative_mask.any(axis=1).idxmax()
        violators = weights.loc[timestamp][weights.loc[timestamp] < -EPSILON].to_dict()
        raise ValueError(
            f"Validation Error at {timestamp}: Negative weights are not allowed. "
            f"Violating assets: {violators}"
        )

    weight_sums = weights.sum(axis=1)
    overweight_mask = weight_sums > 1.0 + EPSILON
    if overweight_mask.any():
        timestamp = overweight_mask.idxmax()
        raise ValueError(
            f"Validation Error at {timestamp}: Sum of weights exceeds 1.0. "
            f"Total weight sum: {weight_sums.loc[timestamp]:.6f}"
        )


def compute_metrics(returns: pd.Series, periods_per_year: int) -> dict[str, float]:
    if returns.empty:
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }

    cumulative_return = float((1.0 + returns).prod() - 1.0)
    annualized_return = float((1.0 + cumulative_return) ** (periods_per_year / len(returns)) - 1.0)
    annualized_volatility = float(returns.std() * np.sqrt(periods_per_year))
    sharpe_ratio = 0.0 if annualized_volatility == 0 else annualized_return / annualized_volatility

    equity_curve = (1.0 + returns).cumprod()
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
    }


def plot_equity_curve(equity_curve: pd.Series, plot_path: str | Path) -> Path:
    if isinstance(equity_curve.index, pd.DatetimeIndex):
        equity_curve.index = equity_curve.index.strftime("%Y-%m-%d %H:%M")
    
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity_curve.index, equity_curve.values, linewidth=1.5, color="#0B6E4F")
    ax.set_title("Portfolio Equity Curve")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Net Value")

    sep = max(1, len(equity_curve) // 10)
    xticks = np.arange(0, len(equity_curve), sep)
    ax.set_xticks(xticks)
    ax.set_xticklabels(equity_curve.index[xticks], rotation=45, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return plot_path


def run_backtest(
    data_path: str | Path,
    signal: pd.DataFrame | str | Path,
    periods_per_year: int = 252 * 78,
    plot_path: str | Path | None = None,
) -> dict[str, Any]:
    market_data = load_tabular_data(data_path)
    close_prices = extract_close_prices(market_data)

    weights = prepare_signal_frame(
        signal=signal,
        target_index=close_prices.index,
        target_columns=close_prices.columns,
    )
    validate_weights(weights)

    asset_returns = close_prices.pct_change()
    portfolio_returns = (weights.shift(1).fillna(0.0) * asset_returns).sum(axis=1)
    portfolio_returns = portfolio_returns.iloc[1:].rename("Portfolio_Return")
    equity_curve = (1.0 + portfolio_returns).cumprod().rename("Equity_Curve")
    metrics = compute_metrics(portfolio_returns, periods_per_year=periods_per_year)

    saved_plot_path = None
    if plot_path is not None:
        saved_plot_path = plot_equity_curve(equity_curve, plot_path)

    return {
        "weights": weights,
        "asset_returns": asset_returns,
        "portfolio_returns": portfolio_returns,
        "equity_curve": equity_curve,
        "metrics": metrics,
        "plot_path": saved_plot_path,
    }


def print_report(metrics: dict[str, float]) -> None:
    print("\n--- Vectorized Backtest Report ---")
    print(f"Cumulative Return        : {metrics['cumulative_return'] * 100:.2f}%")
    print(f"Annualized Return        : {metrics['annualized_return'] * 100:.2f}%")
    print(f"Annualized Volatility    : {metrics['annualized_volatility'] * 100:.2f}%")
    print(f"Sharpe Ratio             : {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown             : {metrics['max_drawdown'] * 100:.2f}%")
    print("----------------------------------")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vectorized research backtest tool.")
    parser.add_argument("--data-path", required=True, help="Path to market data.")
    parser.add_argument("--signal-path", required=True, help="Path to signal weights.")
    parser.add_argument(
        "--periods-per-year",
        type=int,
        default=252 * 78,
        help="Annualization factor for 5-minute US equity data.",
    )
    parser.add_argument(
        "--plot-path",
        default="tools/output/equity_curve.png",
        help="Path to save the equity curve plot.",
    )
    parser.add_argument(
        "--returns-path",
        default=None,
        help="Optional path to save portfolio returns as CSV.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    result = run_backtest(
        data_path=args.data_path,
        signal=args.signal_path,
        periods_per_year=args.periods_per_year,
        plot_path=args.plot_path,
    )

    print_report(result["metrics"])

    if result["plot_path"] is not None:
        print(f"Equity curve saved to: {result['plot_path']}")

    if args.returns_path:
        returns_path = Path(args.returns_path)
        returns_path.parent.mkdir(parents=True, exist_ok=True)
        result["portfolio_returns"].to_csv(returns_path, header=True)
        print(f"Portfolio returns saved to: {returns_path}")


if __name__ == "__main__":
    main()
