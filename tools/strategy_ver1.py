from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable
from tqdm import tqdm

import numpy as np
import pandas as pd
from utils import load_tabular_data


DEFAULT_MOMENTUM_WINDOWS = (6, 12, 24, 78)
DEFAULT_VOLUME_WINDOW = 78
DEFAULT_TOP_SELECTOR = 20.0
DEFAULT_MAX_WEIGHT = 0.05
DEFAULT_MIN_RELATIVE_VOLUME = 0.8
DEFAULT_VOLUME_CAP = 2.0
EPSILON = 1e-12


def _parse_tuple_column(column: object) -> tuple[str, str] | None:
    if isinstance(column, tuple) and len(column) == 2:
        ticker, field = column
        return str(ticker), str(field)

    if not isinstance(column, str):
        return None

    try:
        value = ast.literal_eval(column)
    except (ValueError, SyntaxError):
        return None

    if isinstance(value, tuple) and len(value) == 2:
        ticker, field = value
        return str(ticker), str(field)

    return None


def extract_close_volume_frames(market_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(market_data.index, pd.DatetimeIndex):
        if "datetime" not in market_data.columns:
            raise ValueError("Market data must have a DatetimeIndex or a 'datetime' column.")
        market_data = market_data.copy()
        market_data["datetime"] = pd.to_datetime(market_data["datetime"])
        market_data = market_data.set_index("datetime")

    market_data = market_data.sort_index()

    if isinstance(market_data.columns, pd.MultiIndex):
        close_prices = market_data.xs("close", axis=1, level=-1)
        volume = market_data.xs("volume", axis=1, level=-1)
    else:
        parsed = {_parse_tuple_column(col): col for col in market_data.columns}
        pairs = [pair for pair in parsed if pair is not None]
        if not pairs:
            raise ValueError("Expected MultiIndex columns or stringified '(ticker, field)' columns.")

        tickers = sorted({ticker for ticker, _ in pairs})
        close_columns = []
        volume_columns = []
        for ticker in tickers:
            close_key = (ticker, "close")
            volume_key = (ticker, "volume")
            if close_key not in parsed or volume_key not in parsed:
                raise ValueError(f"Missing close/volume fields for ticker: {ticker}")
            close_columns.append(parsed[close_key])
            volume_columns.append(parsed[volume_key])

        close_prices = market_data[close_columns].copy()
        volume = market_data[volume_columns].copy()
        close_prices.columns = tickers
        volume.columns = tickers

    close_prices = close_prices.astype(float)
    volume = volume.astype(float)

    if close_prices.isna().any().any():
        raise ValueError("Close price panel contains NaN values.")
    if volume.isna().any().any():
        raise ValueError("Volume panel contains NaN values.")

    if not close_prices.columns.equals(volume.columns):
        raise ValueError("Close and volume columns are not aligned.")

    return close_prices, volume


def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    demeaned = frame.sub(frame.mean(axis=1), axis=0)
    std = frame.std(axis=1).replace(0.0, np.nan)
    zscore = demeaned.div(std, axis=0)
    return zscore.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _normalize_windows(momentum_windows: Iterable[int]) -> tuple[int, ...]:
    windows = tuple(sorted({int(window) for window in momentum_windows if int(window) > 0}))
    if not windows:
        raise ValueError("momentum_windows must contain at least one positive integer.")
    return windows


def build_signal_scores(
    market_data: pd.DataFrame,
    momentum_windows: Iterable[int] = DEFAULT_MOMENTUM_WINDOWS,
    volume_window: int = DEFAULT_VOLUME_WINDOW,
    min_relative_volume: float = DEFAULT_MIN_RELATIVE_VOLUME,
    volume_cap: float = DEFAULT_VOLUME_CAP,
) -> pd.DataFrame:
    close_prices, volume = extract_close_volume_frames(market_data)
    momentum_windows = _normalize_windows(momentum_windows)
    max_momentum_window = max(momentum_windows)

    if volume_window <= 0:
        raise ValueError("volume_window must be positive.")
    if volume_cap <= 0:
        raise ValueError("volume_cap must be positive.")
    
    # concat a sub dataframe with values equal to close_prices.iloc[0] and length max_momentum_window - 1 for momentum features calculation
    initial_row = close_prices.iloc[0].to_frame().T
    close_prices = pd.concat([initial_row] * (max_momentum_window - 1) + [close_prices])

    momentum_components = []
    for window in momentum_windows:
        raw_return = close_prices.pct_change(window)
        momentum_components.append(_cross_sectional_zscore(raw_return))

    momentum_score = sum(momentum_components) / len(momentum_components)

    # remove the concat-ed initial rows
    momentum_score = momentum_score.iloc[max_momentum_window - 1:]

    avg_volume = volume.rolling(window=volume_window, min_periods=1).mean()
    relative_volume = volume.div(avg_volume.replace(0.0, np.nan))
    relative_volume = relative_volume.replace([np.inf, -np.inf], np.nan)

    volume_multiplier = relative_volume.clip(lower=0.0, upper=volume_cap) / volume_cap
    volume_multiplier = volume_multiplier.where(relative_volume >= min_relative_volume, 0.0)
    volume_multiplier = volume_multiplier.fillna(0.0)

    scores = momentum_score.mul(volume_multiplier)
    scores = scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scores = scores.reindex(columns=close_prices.columns).astype(float)
    return - scores


def _resolve_selection_count(top_selector: float, n_assets: int) -> int:
    if n_assets <= 0:
        raise ValueError("n_assets must be positive.")

    selector = float(top_selector)
    if np.isclose(selector, 1.0) or np.isclose(selector, float(n_assets)):
        return n_assets

    if selector < 1.0:
        count = int(np.ceil(selector * n_assets))
        return min(max(count, 1), n_assets)

    count = int(round(selector))
    return min(max(count, 1), n_assets)


def build_weights(
    scores: pd.DataFrame,
    top_selector: float = DEFAULT_TOP_SELECTOR,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> pd.DataFrame:
    if scores.empty:
        raise ValueError("scores must not be empty.")
    if max_weight <= 0:
        raise ValueError("max_weight must be positive.")

    tickers = scores.columns
    n_assets = len(tickers)
    selection_count = _resolve_selection_count(top_selector=top_selector, n_assets=n_assets)
    min_live_position = 0

    weights = pd.DataFrame(0.0, index=scores.index, columns=tickers, dtype=float)
    nonzero_rows = scores.ne(0.0).any(axis=1).to_numpy()
    if not nonzero_rows.any():
        return weights
    min_live_position = int(np.flatnonzero(nonzero_rows)[0])

    for row_position, (timestamp, row) in enumerate(
        tqdm(scores.iterrows(), total=len(scores), desc="Building weights", unit="timestamp")
    ):
        if row_position < min_live_position:
            continue
        clean_row = row.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if selection_count >= n_assets:
            selected = clean_row.index
        else:
            selected = clean_row.nlargest(selection_count).index

        selected_scores = clean_row.loc[selected]
        positive_scores = selected_scores[selected_scores > 0.0]

        if positive_scores.empty:
            raw_weights = pd.Series(1.0 / len(selected), index=selected, dtype=float)
        else:
            raw_weights = positive_scores / positive_scores.sum()

        clipped_weights = raw_weights.clip(upper=max_weight)
        weights.loc[timestamp, clipped_weights.index] = clipped_weights

    weight_sums = weights.sum(axis=1)
    if (weights < -EPSILON).any().any():
        raise ValueError("Generated negative weights, which violates long-only constraints.")
    if (weight_sums > 1.0 + EPSILON).any():
        raise ValueError("Generated weights exceed total portfolio weight of 1.0.")

    return weights


def generate_weights_matrix(
    data_path: str | Path,
    momentum_windows: Iterable[int] = DEFAULT_MOMENTUM_WINDOWS,
    volume_window: int = DEFAULT_VOLUME_WINDOW,
    min_relative_volume: float = DEFAULT_MIN_RELATIVE_VOLUME,
    volume_cap: float = DEFAULT_VOLUME_CAP,
    top_selector: float = DEFAULT_TOP_SELECTOR,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> pd.DataFrame:
    market_data = load_tabular_data(data_path)
    scores = build_signal_scores(
        market_data=market_data,
        momentum_windows=momentum_windows,
        volume_window=volume_window,
        min_relative_volume=min_relative_volume,
        volume_cap=volume_cap,
    )
    return build_weights(scores=scores, top_selector=top_selector, max_weight=max_weight)


def save_weights(weights: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        weights.to_parquet(output_path)
    elif suffix == ".csv":
        weights.to_csv(output_path)
    else:
        raise ValueError("output_path must end with .parquet or .csv")

    return output_path


def _parse_windows(raw_value: str) -> tuple[int, ...]:
    return _normalize_windows(int(part.strip()) for part in raw_value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate weights for the v1 volume-confirmed momentum strategy.")
    parser.add_argument("--data-path", required=True, help="Path to market data.")
    parser.add_argument("--output-path", required=True, help="Path to save generated weights (.parquet or .csv).")
    parser.add_argument(
        "--momentum-windows",
        default="6,12,24,78",
        help="Comma-separated momentum windows in bars.",
    )
    parser.add_argument(
        "--volume-window",
        type=int,
        default=DEFAULT_VOLUME_WINDOW,
        help="Rolling window for relative volume.",
    )
    parser.add_argument(
        "--min-relative-volume",
        type=float,
        default=DEFAULT_MIN_RELATIVE_VOLUME,
        help="Minimum relative volume required to keep a score active.",
    )
    parser.add_argument(
        "--volume-cap",
        type=float,
        default=DEFAULT_VOLUME_CAP,
        help="Upper cap for relative volume multiplier scaling.",
    )
    parser.add_argument(
        "--top-selector",
        type=float,
        default=DEFAULT_TOP_SELECTOR,
        help="If <1 uses top q%%, if >1 uses top N, if ==1 or ==438 uses all assets.",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=DEFAULT_MAX_WEIGHT,
        help="Maximum weight allowed for a single stock.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    weights = generate_weights_matrix(
        data_path=args.data_path,
        momentum_windows=_parse_windows(args.momentum_windows),
        volume_window=args.volume_window,
        min_relative_volume=args.min_relative_volume,
        volume_cap=args.volume_cap,
        top_selector=args.top_selector,
        max_weight=args.max_weight,
    )
    output_path = save_weights(weights, args.output_path)
    print(f"Generated weights with shape {weights.shape} -> {output_path}")


if __name__ == "__main__":
    main()
