from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EPSILON = 1e-12


@dataclass(frozen=True)
class DatasetPaths:
    dataset: str
    split: str
    mom_output_dir: Path
    mom_feature_dir: Path
    mom_ic_series_dir: Path
    mom_summary_path: Path
    mom_weights_dir: Path
    feat_ana_dir: Path
    selected_feature_sets_path: Path
    strategy_ver2_output_dir: Path
    strategy_ver2_score_dir: Path
    strategy_ver2_manifest_path: Path
    strategy_ver2_weights_dir: Path
    strategy_ver2_backtest_dir: Path


def load_tabular_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_parquet(path, engine="fastparquet")

    if suffix == ".csv":
        return pd.read_csv(path, index_col=0, parse_dates=True)

    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)

    raise ValueError(f"Unsupported file format: {path}")


def save_tabular_data(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        frame.to_parquet(path)
    elif suffix == ".csv":
        frame.to_csv(path)
    elif suffix in {".pkl", ".pickle"}:
        frame.to_pickle(path)
    else:
        raise ValueError(f"Unsupported output format: {path}")

    return path


def infer_dataset_split(data_path: str | Path) -> tuple[str, str]:
    path = Path(data_path)
    split = path.stem
    dataset = path.parent.name
    if not dataset or dataset == "." or not split:
        raise ValueError(f"Cannot infer dataset/split from data path: {data_path}")
    return dataset, split


def resolve_dataset_split(
    data_path: str | Path | None = None,
    dataset: str | None = None,
    split: str | None = None,
) -> tuple[str, str]:
    inferred_dataset = inferred_split = None
    if data_path is not None:
        inferred_dataset, inferred_split = infer_dataset_split(data_path)

    dataset = dataset or inferred_dataset
    split = split or inferred_split
    if not dataset or not split:
        raise ValueError("dataset and split are required when data_path is not provided.")

    return dataset, split


def build_dataset_paths(
    dataset: str,
    split: str,
    output_root: str | Path = "data/output",
    weights_root: str | Path = "data/weights",
) -> DatasetPaths:
    output_root = Path(output_root)
    weights_root = Path(weights_root)

    mom_output_dir = output_root / "mom_features" / dataset / split
    feat_ana_dir = output_root / "feat_ana" / dataset / split
    strategy_output_dir = output_root / "strategy_ver2" / dataset / split

    return DatasetPaths(
        dataset=dataset,
        split=split,
        mom_output_dir=mom_output_dir,
        mom_feature_dir=mom_output_dir / "features",
        mom_ic_series_dir=mom_output_dir / "ic_series",
        mom_summary_path=mom_output_dir / "feature_summary.csv",
        mom_weights_dir=weights_root / "mom_features" / dataset / split,
        feat_ana_dir=feat_ana_dir,
        selected_feature_sets_path=feat_ana_dir / "selected_feature_sets.csv",
        strategy_ver2_output_dir=strategy_output_dir,
        strategy_ver2_score_dir=strategy_output_dir / "scores",
        strategy_ver2_manifest_path=strategy_output_dir / "manifest.csv",
        strategy_ver2_weights_dir=weights_root / "strategy_ver2" / dataset / split,
        strategy_ver2_backtest_dir=output_root / "backtest" / "strategy_ver2" / dataset / split,
    )


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


def _ensure_datetime_index(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if isinstance(frame.index, pd.DatetimeIndex):
        return frame.sort_index()
    if "datetime" not in frame.columns:
        raise ValueError(f"{name} must have a DatetimeIndex or a 'datetime' column.")
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    return out.set_index("datetime").sort_index()


def extract_close_volume_frames(market_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    market_data = _ensure_datetime_index(market_data, "Market data")

    if isinstance(market_data.columns, pd.MultiIndex):
        fields = market_data.columns.get_level_values(-1)
        if "close" not in fields or "volume" not in fields:
            raise ValueError("MultiIndex market data must include close and volume fields.")
        close = market_data.xs("close", axis=1, level=-1)
        volume = market_data.xs("volume", axis=1, level=-1)
    else:
        parsed = {_parse_tuple_column(col): col for col in market_data.columns}
        pairs = [pair for pair in parsed if pair is not None]
        if not pairs:
            raise ValueError("Expected MultiIndex columns or stringified '(ticker, field)' columns.")

        tickers = sorted({ticker for ticker, _ in pairs})
        close_cols = []
        volume_cols = []
        for ticker in tickers:
            close_key = (ticker, "close")
            volume_key = (ticker, "volume")
            if close_key not in parsed or volume_key not in parsed:
                raise ValueError(f"Missing close/volume fields for ticker: {ticker}")
            close_cols.append(parsed[close_key])
            volume_cols.append(parsed[volume_key])

        close = market_data[close_cols].copy()
        volume = market_data[volume_cols].copy()
        close.columns = tickers
        volume.columns = tickers

    close = close.astype(float)
    volume = volume.astype(float)
    close.columns = pd.Index(close.columns.astype(str))
    volume.columns = pd.Index(volume.columns.astype(str))

    if close.isna().any().any():
        raise ValueError("Close price panel contains NaN values.")
    if volume.isna().any().any():
        raise ValueError("Volume panel contains NaN values.")
    if not close.columns.equals(volume.columns):
        raise ValueError("Close and volume columns are not aligned.")

    return close, volume


def extract_close_prices(market_data: pd.DataFrame) -> pd.DataFrame:
    market_data = _ensure_datetime_index(market_data, "Market data")

    if isinstance(market_data.columns, pd.MultiIndex):
        fields = market_data.columns.get_level_values(-1)
        if "close" not in fields:
            raise ValueError("MultiIndex market data must include a close field.")
        close = market_data.xs("close", axis=1, level=-1)
    else:
        parsed = {_parse_tuple_column(col): col for col in market_data.columns}
        close_pairs = sorted(pair for pair in parsed if pair is not None and pair[1] == "close")
        if close_pairs:
            close_cols = [parsed[pair] for pair in close_pairs]
            close = market_data[close_cols].copy()
            close.columns = [ticker for ticker, _ in close_pairs]
        else:
            close = market_data.copy()

    close = close.sort_index().astype(float)
    close.columns = pd.Index(close.columns.astype(str))
    if close.isna().any().any():
        raise ValueError("Market close prices contain NaN values.")
    return close


def clean_numeric_frame(frame: pd.DataFrame, fill_value: float | None = None) -> pd.DataFrame:
    out = frame.replace([np.inf, -np.inf], np.nan).astype(float)
    if fill_value is not None:
        out = out.fillna(fill_value)
    return out


def cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    clean = clean_numeric_frame(frame)
    demeaned = clean.sub(clean.mean(axis=1), axis=0)
    std = clean.std(axis=1).replace(0.0, np.nan)
    return demeaned.div(std, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def time_series_zscore(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 0:
        raise ValueError("window must be positive.")
    mean = frame.rolling(window=window, min_periods=window).mean()
    std = frame.rolling(window=window, min_periods=window).std().replace(0.0, np.nan)
    return clean_numeric_frame((frame - mean) / std)


def resolve_selection_count(top_selector: float, n_assets: int) -> int:
    if n_assets <= 0:
        raise ValueError("n_assets must be positive.")

    selector = float(top_selector)
    if np.isclose(selector, 1.0) or np.isclose(selector, float(n_assets)):
        return n_assets

    if selector < 1.0:
        return min(max(int(math.ceil(selector * n_assets)), 1), n_assets)

    return min(max(int(round(selector)), 1), n_assets)


def build_top_score_weights(
    scores: pd.DataFrame,
    top_selector: float,
    max_weight: float,
    require_positive: bool = True,
) -> pd.DataFrame:
    if scores.empty:
        raise ValueError("scores must not be empty.")
    if max_weight <= 0.0:
        raise ValueError("max_weight must be positive.")

    values = scores.to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = 0.0
    n_rows, n_assets = values.shape
    selection_count = resolve_selection_count(top_selector, n_assets)
    weights = np.zeros_like(values)

    if selection_count >= n_assets:
        selected_idx = np.tile(np.arange(n_assets), (n_rows, 1))
    else:
        selected_idx = np.argpartition(values, -selection_count, axis=1)[:, -selection_count:]

    row_idx = np.arange(n_rows)[:, None]
    selected_scores = values[row_idx, selected_idx]
    if require_positive:
        allocation_scores = np.where(selected_scores > 0.0, selected_scores, 0.0)
    else:
        allocation_scores = selected_scores - np.nanmin(selected_scores, axis=1, keepdims=True)
        allocation_scores = np.where(allocation_scores > 0.0, allocation_scores, 0.0)

    score_sum = allocation_scores.sum(axis=1, keepdims=True)
    nonzero = score_sum[:, 0] > 0.0
    if nonzero.any():
        raw = allocation_scores[nonzero] / score_sum[nonzero]
        raw = np.minimum(raw, max_weight)
        weights[np.arange(n_rows)[nonzero, None], selected_idx[nonzero]] = raw

    out = pd.DataFrame(weights, index=scores.index, columns=scores.columns, dtype=float)
    validate_weight_frame(out, max_weight=max_weight)
    return out


def validate_weight_frame(weights: pd.DataFrame, max_weight: float | None = None) -> None:
    if (weights < -EPSILON).any().any():
        raise ValueError("Generated negative weights.")
    if (weights.sum(axis=1) > 1.0 + EPSILON).any():
        raise ValueError("Generated weights exceed total portfolio weight of 1.0.")
    if max_weight is not None and (weights > max_weight + EPSILON).any().any():
        raise ValueError("Generated weights exceed max_weight.")


def load_feature_panel(
    path: str | Path,
    direction: str = "raw",
    target_index: pd.Index | None = None,
    target_columns: pd.Index | None = None,
) -> pd.DataFrame:
    if direction not in {"raw", "neg"}:
        raise ValueError("direction must be 'raw' or 'neg'.")

    frame = load_tabular_data(path)
    frame = _ensure_datetime_index(frame, "Feature panel")
    if isinstance(frame.columns, pd.MultiIndex):
        raise ValueError("Feature panel columns must be ticker names, not a MultiIndex.")

    frame.columns = pd.Index(frame.columns.astype(str))
    frame = clean_numeric_frame(frame)
    if direction == "neg":
        frame = -frame

    if target_columns is not None:
        frame = frame.reindex(columns=pd.Index(target_columns.astype(str)), fill_value=np.nan)
    if target_index is not None:
        frame = frame.reindex(target_index)

    return frame


def load_selected_feature_set(
    selected_features_path: str | Path,
    set_name: str,
    feature_limit: int | None = None,
) -> pd.DataFrame:
    path = Path(selected_features_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find selected feature sets file: {path}")

    selected = pd.read_csv(path)
    required = {"set_name", "selection_rank", "variant", "feature", "direction", "feature_path"}
    missing = sorted(required.difference(selected.columns))
    if missing:
        raise ValueError(f"Selected feature sets file is missing columns: {missing}")

    subset = selected[selected["set_name"] == set_name].sort_values("selection_rank")
    if subset.empty:
        available = sorted(selected["set_name"].dropna().unique())
        raise ValueError(f"Unknown set_name '{set_name}'. Available sets: {available}")
    if feature_limit is not None:
        subset = subset.head(feature_limit)

    missing_paths = [path for path in subset["feature_path"] if not Path(path).exists()]
    if missing_paths:
        preview = missing_paths[:5]
        raise FileNotFoundError(f"Missing feature files: {preview}")

    return subset.reset_index(drop=True)


def finite_corr(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 5:
        return np.nan

    x = left[mask]
    y = right[mask]
    x_std = x.std()
    y_std = y.std()
    if x_std == 0.0 or y_std == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])
