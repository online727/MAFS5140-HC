from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.append(str(TOOLS_DIR))

from utils import load_tabular_data
from utils import build_dataset_paths
from utils import build_top_score_weights
from utils import clean_numeric_frame
from utils import cross_sectional_zscore
from utils import extract_close_volume_frames
from utils import resolve_dataset_split
from utils import save_tabular_data
from utils import time_series_zscore


EPSILON = 1e-12
DEFAULT_FORWARD_HORIZONS = {"5min": 1, "15min": 3, "30min": 6, "60min": 12}
DEFAULT_CLOSE_WINDOWS = (1, 3, 6, 12, 24, 39, 78)
DEFAULT_MEDIUM_WINDOWS = (6, 12, 24, 39, 78)
DEFAULT_LONG_WINDOWS = (6, 12, 24, 39, 78, 156)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    params: dict[str, int | str]


def safe_divide(numerator: pd.DataFrame, denominator: pd.DataFrame | pd.Series | float) -> pd.DataFrame:
    result = numerator.div(denominator)
    return result.replace([np.inf, -np.inf], np.nan)


def clean_feature(feature: pd.DataFrame) -> pd.DataFrame:
    return clean_numeric_frame(feature)


def rsi(close: pd.DataFrame, window: int) -> pd.DataFrame:
    diff = close.diff()
    gain = diff.clip(lower=0.0)
    loss = -diff.clip(upper=0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain.div(avg_loss.replace(0.0, np.nan))
    value = 100.0 - (100.0 / (1.0 + rs))
    value = value.where(avg_loss != 0.0, 100.0)
    value = value.where(avg_gain != 0.0, 0.0)
    return clean_feature((value - 50.0) / 50.0)


def macd_hist(close: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
    fast_ema = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return clean_feature((macd_line - signal_line).div(close.replace(0.0, np.nan)))


def obv_proxy(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    direction = np.sign(close.diff()).replace(0.0, np.nan).ffill().fillna(0.0)
    return (direction * volume).cumsum()


def build_feature_specs() -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []

    for window in DEFAULT_CLOSE_WINDOWS:
        specs.append(FeatureSpec(f"ret_{window}", "return", {"window": window}))
        specs.append(FeatureSpec(f"logret_{window}", "return", {"window": window}))
        specs.append(FeatureSpec(f"roc_{window}", "oscillator", {"window": window}))
        specs.append(FeatureSpec(f"volume_change_{window}", "volume", {"window": window}))

    for window in DEFAULT_MEDIUM_WINDOWS:
        specs.extend(
            [
                FeatureSpec(f"rolling_mean_ret_{window}", "return", {"window": window}),
                FeatureSpec(f"rolling_sum_ret_{window}", "return", {"window": window}),
                FeatureSpec(f"ret_zscore_ts_{window}", "return", {"window": window}),
                FeatureSpec(f"price_zscore_ts_{window}", "trend", {"window": window}),
                FeatureSpec(f"sma_slope_{window}", "trend", {"window": window}),
                FeatureSpec(f"ema_slope_{window}", "trend", {"window": window}),
                FeatureSpec(f"rsi_{window}", "oscillator", {"window": window}),
                FeatureSpec(f"realized_vol_{window}", "volatility", {"window": window}),
                FeatureSpec(f"downside_vol_{window}", "volatility", {"window": window}),
                FeatureSpec(f"vol_adjusted_ret_{window}", "volatility", {"window": window}),
            ]
        )

    for window in DEFAULT_LONG_WINDOWS:
        specs.extend(
            [
                FeatureSpec(f"sma_dist_{window}", "trend", {"window": window}),
                FeatureSpec(f"ema_dist_{window}", "trend", {"window": window}),
                FeatureSpec(f"relative_volume_{window}", "volume", {"window": window}),
                FeatureSpec(f"volume_zscore_{window}", "volume", {"window": window}),
                FeatureSpec(f"dollar_volume_zscore_{window}", "volume", {"window": window}),
            ]
        )

    for window in (12, 24, 39, 78):
        specs.extend(
            [
                FeatureSpec(f"stoch_close_{window}", "oscillator", {"window": window}),
                FeatureSpec(f"price_volume_corr_{window}", "volume", {"window": window}),
                FeatureSpec(f"obv_slope_{window}", "volume", {"window": window}),
                FeatureSpec(f"return_range_{window}", "volatility", {"window": window}),
            ]
        )

    for fast, slow in ((3, 12), (6, 24), (12, 39), (12, 78), (24, 78)):
        specs.append(FeatureSpec(f"ma_cross_{fast}_{slow}", "trend", {"fast": fast, "slow": slow}))

    for fast, slow, signal in ((6, 24, 9), (12, 26, 9), (12, 39, 9)):
        specs.append(
            FeatureSpec(
                f"macd_{fast}_{slow}_{signal}",
                "oscillator",
                {"fast": fast, "slow": slow, "signal": signal},
            )
        )

    return specs


def compute_feature(spec: FeatureSpec, close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    returns_1 = close.pct_change()
    log_close = np.log(close.replace(0.0, np.nan))
    dollar_volume = close * volume
    name = spec.name
    params = spec.params

    if name.startswith("ret_"):
        return clean_feature(close.pct_change(int(params["window"])))
    if name.startswith("logret_"):
        return clean_feature(log_close.diff(int(params["window"])))
    if name.startswith("roc_"):
        window = int(params["window"])
        return clean_feature(close.diff(window).div(close.shift(window).replace(0.0, np.nan)))
    if name.startswith("volume_change_"):
        return clean_feature(volume.pct_change(int(params["window"])))

    if name.startswith("rolling_mean_ret_"):
        window = int(params["window"])
        return clean_feature(returns_1.rolling(window=window, min_periods=window).mean())
    if name.startswith("rolling_sum_ret_"):
        window = int(params["window"])
        return clean_feature(returns_1.rolling(window=window, min_periods=window).sum())
    if name.startswith("ret_zscore_ts_"):
        return time_series_zscore(returns_1, int(params["window"]))
    if name.startswith("price_zscore_ts_"):
        return time_series_zscore(close, int(params["window"]))

    if name.startswith("sma_dist_"):
        window = int(params["window"])
        sma = close.rolling(window=window, min_periods=window).mean()
        return clean_feature(close.div(sma.replace(0.0, np.nan)) - 1.0)
    if name.startswith("ema_dist_"):
        window = int(params["window"])
        ema = close.ewm(span=window, adjust=False, min_periods=window).mean()
        return clean_feature(close.div(ema.replace(0.0, np.nan)) - 1.0)
    if name.startswith("sma_slope_"):
        window = int(params["window"])
        sma = close.rolling(window=window, min_periods=window).mean()
        return clean_feature(sma.pct_change(window))
    if name.startswith("ema_slope_"):
        window = int(params["window"])
        ema = close.ewm(span=window, adjust=False, min_periods=window).mean()
        return clean_feature(ema.pct_change(window))
    if name.startswith("ma_cross_"):
        fast = int(params["fast"])
        slow = int(params["slow"])
        fast_sma = close.rolling(window=fast, min_periods=fast).mean()
        slow_sma = close.rolling(window=slow, min_periods=slow).mean()
        return clean_feature(fast_sma.div(slow_sma.replace(0.0, np.nan)) - 1.0)

    if name.startswith("rsi_"):
        return rsi(close, int(params["window"]))
    if name.startswith("stoch_close_"):
        window = int(params["window"])
        roll_min = close.rolling(window=window, min_periods=window).min()
        roll_max = close.rolling(window=window, min_periods=window).max()
        return clean_feature((close - roll_min).div((roll_max - roll_min).replace(0.0, np.nan)) - 0.5)
    if name.startswith("macd_"):
        return macd_hist(
            close=close,
            fast=int(params["fast"]),
            slow=int(params["slow"]),
            signal=int(params["signal"]),
        )

    if name.startswith("relative_volume_"):
        window = int(params["window"])
        avg_volume = volume.rolling(window=window, min_periods=window).mean()
        return clean_feature(volume.div(avg_volume.replace(0.0, np.nan)))
    if name.startswith("volume_zscore_"):
        return time_series_zscore(volume, int(params["window"]))
    if name.startswith("dollar_volume_zscore_"):
        return time_series_zscore(dollar_volume, int(params["window"]))
    if name.startswith("price_volume_corr_"):
        window = int(params["window"])
        volume_change = volume.pct_change()
        return clean_feature(returns_1.rolling(window=window, min_periods=window).corr(volume_change))
    if name.startswith("obv_slope_"):
        window = int(params["window"])
        obv = obv_proxy(close, volume)
        scale = volume.rolling(window=window, min_periods=window).sum().replace(0.0, np.nan)
        return clean_feature(obv.diff(window).div(scale))

    if name.startswith("realized_vol_"):
        window = int(params["window"])
        return clean_feature(returns_1.rolling(window=window, min_periods=window).std())
    if name.startswith("downside_vol_"):
        window = int(params["window"])
        downside = returns_1.where(returns_1 < 0.0, 0.0)
        return clean_feature(downside.rolling(window=window, min_periods=window).std())
    if name.startswith("return_range_"):
        window = int(params["window"])
        roll_min = close.rolling(window=window, min_periods=window).min()
        roll_max = close.rolling(window=window, min_periods=window).max()
        return clean_feature(roll_max.div(roll_min.replace(0.0, np.nan)) - 1.0)
    if name.startswith("vol_adjusted_ret_"):
        window = int(params["window"])
        ret = close.pct_change(window)
        vol = returns_1.rolling(window=window, min_periods=window).std()
        return clean_feature(ret.div(vol.replace(0.0, np.nan)))

    raise ValueError(f"Unsupported feature spec: {spec}")


def build_feature_weights(
    feature: pd.DataFrame,
    direction: str,
    top_selector: float,
    max_weight: float,
) -> pd.DataFrame:
    if direction not in {"raw", "neg"}:
        raise ValueError("direction must be 'raw' or 'neg'.")
    if max_weight <= 0.0:
        raise ValueError("max_weight must be positive.")

    scores = cross_sectional_zscore(feature)
    if direction == "neg":
        scores = -scores
    return build_top_score_weights(scores=scores, top_selector=top_selector, max_weight=max_weight)


def compute_pnl_metrics(
    close: pd.DataFrame,
    weights: pd.DataFrame,
    periods_per_year: int,
) -> dict[str, float]:
    asset_returns = close.pct_change()
    portfolio_returns = (weights.shift(1).fillna(0.0) * asset_returns).sum(axis=1).iloc[1:]
    if portfolio_returns.empty:
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }

    cumulative_return = float((1.0 + portfolio_returns).prod() - 1.0)
    annualized_return = float(
        (1.0 + cumulative_return) ** (periods_per_year / len(portfolio_returns)) - 1.0
    )
    annualized_volatility = float(portfolio_returns.std() * np.sqrt(periods_per_year))
    sharpe_ratio = 0.0 if annualized_volatility == 0.0 else annualized_return / annualized_volatility
    equity_curve = (1.0 + portfolio_returns).cumprod()
    max_drawdown = float(((equity_curve - equity_curve.cummax()) / equity_curve.cummax()).min())

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
    }


def rowwise_pearson(left: pd.DataFrame, right: pd.DataFrame, min_assets: int = 5) -> pd.Series:
    valid = left.notna() & right.notna()
    count = valid.sum(axis=1)
    left_masked = left.where(valid)
    right_masked = right.where(valid)

    left_dm = left_masked.sub(left_masked.mean(axis=1), axis=0)
    right_dm = right_masked.sub(right_masked.mean(axis=1), axis=0)
    numerator = (left_dm * right_dm).sum(axis=1)
    denominator = np.sqrt((left_dm.pow(2).sum(axis=1)) * (right_dm.pow(2).sum(axis=1)))
    corr = numerator.div(denominator.replace(0.0, np.nan))
    corr = corr.where(count >= min_assets)
    return corr.replace([np.inf, -np.inf], np.nan)


def compute_ic_metrics(
    feature: pd.DataFrame,
    close: pd.DataFrame,
    horizons: dict[str, int] = DEFAULT_FORWARD_HORIZONS,
    min_assets: int = 5,
) -> tuple[dict[str, float], pd.DataFrame]:
    metric_values: dict[str, float] = {}
    series_by_name: dict[str, pd.Series] = {}

    signal = feature.replace([np.inf, -np.inf], np.nan)
    signal_rank = signal.rank(axis=1, method="average")

    for label, horizon in horizons.items():
        future_return = close.shift(-horizon).div(close) - 1.0
        ic_series = rowwise_pearson(signal, future_return, min_assets=min_assets).dropna()
        rank_ic_series = rowwise_pearson(
            signal_rank,
            future_return.rank(axis=1, method="average"),
            min_assets=min_assets,
        ).dropna()

        ic_mean = float(ic_series.mean()) if not ic_series.empty else np.nan
        ic_std = float(ic_series.std()) if not ic_series.empty else np.nan
        rank_ic_mean = float(rank_ic_series.mean()) if not rank_ic_series.empty else np.nan
        rank_ic_std = float(rank_ic_series.std()) if not rank_ic_series.empty else np.nan

        metric_values[f"ic_{label}"] = ic_mean
        metric_values[f"icir_{label}"] = ic_mean / ic_std if ic_std and not np.isnan(ic_std) else np.nan
        metric_values[f"rankic_{label}"] = rank_ic_mean
        metric_values[f"rankicir_{label}"] = (
            rank_ic_mean / rank_ic_std if rank_ic_std and not np.isnan(rank_ic_std) else np.nan
        )
        metric_values[f"ic_obs_{label}"] = float(len(ic_series))
        metric_values[f"rankic_obs_{label}"] = float(len(rank_ic_series))

        series_by_name[f"ic_{label}"] = ic_series
        series_by_name[f"rankic_{label}"] = rank_ic_series

    ic_frame = pd.DataFrame(series_by_name)
    return metric_values, ic_frame


def save_frame(frame: pd.DataFrame, output_path: Path) -> Path:
    return save_tabular_data(frame, output_path)


def _parse_directions(raw: str) -> tuple[str, ...]:
    directions = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = sorted(set(directions).difference({"raw", "neg"}))
    if unknown:
        raise ValueError(f"Unknown directions: {unknown}")
    return directions


def _select_specs(specs: list[FeatureSpec], names: str | None, max_features: int | None) -> list[FeatureSpec]:
    if names:
        requested = [name.strip() for name in names.split(",") if name.strip()]
        spec_by_name = {spec.name: spec for spec in specs}
        missing = sorted(set(requested).difference(spec_by_name))
        if missing:
            raise ValueError(f"Unknown feature names: {missing}")
        specs = [spec_by_name[name] for name in requested]

    if max_features is not None:
        specs = specs[:max_features]
    return specs


def run_feature_research(
    data_path: str | Path,
    dataset: str | None = None,
    split: str | None = None,
    feature_format: str = "parquet",
    weights_format: str = "parquet",
    directions: Iterable[str] = ("raw", "neg"),
    top_selector: float = 20.0,
    max_weight: float = 0.05,
    periods_per_year: int = 252 * 78,
    save_features: bool = True,
    save_weights: bool = True,
    save_ic_series: bool = True,
    feature_names: str | None = None,
    max_features: int | None = None,
) -> pd.DataFrame:
    dataset, split = resolve_dataset_split(data_path=data_path, dataset=dataset, split=split)
    paths = build_dataset_paths(dataset=dataset, split=split)
    market_data = load_tabular_data(data_path)
    close, volume = extract_close_volume_frames(market_data)

    specs = _select_specs(build_feature_specs(), names=feature_names, max_features=max_features)
    directions = tuple(directions)
    rows: list[dict[str, object]] = []

    feature_dir = paths.mom_feature_dir
    weights_dir = paths.mom_weights_dir
    ic_series_dir = paths.mom_ic_series_dir
    summary_path = paths.mom_summary_path

    for spec in specs:
        print(f"Computing feature: {spec.name}")
        feature = compute_feature(spec, close=close, volume=volume).reindex_like(close)

        feature_path = None
        if save_features:
            feature_path = feature_dir / f"{spec.name}.{feature_format}"
            save_frame(feature, feature_path)

        ic_metrics, ic_frame = compute_ic_metrics(feature=feature, close=close)

        for direction in directions:
            variant_name = f"{spec.name}__{direction}"
            print(f"Evaluating variant: {variant_name}")
            weights = build_feature_weights(
                feature=feature,
                direction=direction,
                top_selector=top_selector,
                max_weight=max_weight,
            )
            pnl_metrics = compute_pnl_metrics(
                close=close,
                weights=weights,
                periods_per_year=periods_per_year,
            )

            weights_path = None
            if save_weights:
                weights_path = weights_dir / f"{variant_name}.{weights_format}"
                save_frame(weights, weights_path)

            ic_path = None
            if save_ic_series:
                ic_path = ic_series_dir / f"{variant_name}.csv"
                signed_ic_frame = ic_frame if direction == "raw" else -ic_frame
                save_frame(signed_ic_frame, ic_path)

            signed_ic_metrics = ic_metrics.copy()
            if direction == "neg":
                for key in list(signed_ic_metrics):
                    if key.startswith(("ic_", "icir_", "rankic_", "rankicir_")) and "_obs_" not in key:
                        signed_ic_metrics[key] = -signed_ic_metrics[key]

            rows.append(
                {
                    "feature": spec.name,
                    "variant": variant_name,
                    "direction": direction,
                    "family": spec.family,
                    "params": json.dumps(spec.params, sort_keys=True),
                    "feature_path": str(feature_path) if feature_path is not None else "",
                    "weights_path": str(weights_path) if weights_path is not None else "",
                    "ic_series_path": str(ic_path) if ic_path is not None else "",
                    **pnl_metrics,
                    **signed_ic_metrics,
                }
            )

    summary = pd.DataFrame(rows)
    save_tabular_data(summary, summary_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and evaluate close/volume momentum features.")
    parser.add_argument("--data-path", required=True, help="Path to market data parquet/csv/pickle file.")
    parser.add_argument("--dataset", default=None, help="Dataset name. Defaults to parent dir of --data-path.")
    parser.add_argument("--split", default=None, help="Split name. Defaults to stem of --data-path.")
    parser.add_argument("--feature-format", choices=("parquet", "csv"), default="parquet")
    parser.add_argument("--weights-format", choices=("parquet", "csv"), default="parquet")
    parser.add_argument("--directions", default="raw,neg", help="Comma-separated directions: raw,neg.")
    parser.add_argument("--top-selector", type=float, default=20.0)
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--periods-per-year", type=int, default=252 * 78)
    parser.add_argument("--feature-names", default=None, help="Optional comma-separated feature allowlist.")
    parser.add_argument("--max-features", type=int, default=None, help="Optional first-N feature limit for smoke tests.")
    parser.add_argument("--skip-save-features", action="store_true")
    parser.add_argument("--skip-save-weights", action="store_true")
    parser.add_argument("--skip-save-ic-series", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset, split = resolve_dataset_split(args.data_path, args.dataset, args.split)
    paths = build_dataset_paths(dataset, split)
    summary = run_feature_research(
        data_path=args.data_path,
        dataset=dataset,
        split=split,
        feature_format=args.feature_format,
        weights_format=args.weights_format,
        directions=_parse_directions(args.directions),
        top_selector=args.top_selector,
        max_weight=args.max_weight,
        periods_per_year=args.periods_per_year,
        save_features=not args.skip_save_features,
        save_weights=not args.skip_save_weights,
        save_ic_series=not args.skip_save_ic_series,
        feature_names=args.feature_names,
        max_features=args.max_features,
    )
    print(f"Saved summary with {len(summary)} rows to {paths.mom_summary_path}")


if __name__ == "__main__":
    main()
