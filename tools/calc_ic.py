import pandas as pd
import numpy as np
from pathlib import Path


def load_data(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def extract_field(data, field):
    """
    Extract close or volume from data with MultiIndex columns.
    Expected format:
        columns = MultiIndex: ticker, field
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        if "datetime" in data.columns:
            data["datetime"] = pd.to_datetime(data["datetime"])
            data = data.set_index("datetime")
        else:
            raise ValueError("Data must have DatetimeIndex or datetime column.")

    if isinstance(data.columns, pd.MultiIndex):
        fields = data.columns.get_level_values(-1)
        if field not in fields:
            raise ValueError(f"Cannot find field: {field}")
        out = data.xs(field, axis=1, level=-1)
    else:
        raise ValueError("Expected MultiIndex columns like ticker-field.")

    out = out.sort_index().astype(float)
    return out


def cross_sectional_zscore(x):
    mean = x.mean()
    std = x.std()

    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=x.index)

    z = (x - mean) / std
    return z.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def build_momentum_signal(
    close,
    volume,
    momentum_windows=(3, 6),
    volume_window=78,
    min_relative_volume=0.8,
    volume_cap=2.0,
    use_negative_sign=True,
):
    """
    Build signal similar to your MomentumStrategy.

    use_negative_sign=True means:
        scores = - momentum_score * volume_multiplier

    This matches the current strategy.py logic if you kept:
        scores = -scores
    """

    signal = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)

    max_window = max(max(momentum_windows), volume_window)

    for i in range(max_window, len(close)):
        current_price = close.iloc[i]
        current_volume = volume.iloc[i]

        momentum_parts = []

        for window in momentum_windows:
            past_price = close.iloc[i - window]
            raw_return = current_price / past_price - 1.0
            momentum_parts.append(cross_sectional_zscore(raw_return))

        momentum_score = sum(momentum_parts) / len(momentum_parts)

        avg_volume = volume.iloc[i - volume_window + 1 : i + 1].mean()
        relative_volume = current_volume / avg_volume.replace(0.0, np.nan)
        relative_volume = relative_volume.replace([np.inf, -np.inf], np.nan)

        volume_multiplier = relative_volume.clip(lower=0.0, upper=volume_cap) / volume_cap
        volume_multiplier = volume_multiplier.where(relative_volume >= min_relative_volume, 0.0)
        volume_multiplier = volume_multiplier.fillna(0.0)

        score = momentum_score * volume_multiplier

        if use_negative_sign:
            score = -score

        signal.iloc[i] = score

    signal = signal.replace([np.inf, -np.inf], np.nan)
    return signal


def calculate_rank_ic(signal, close, forward_horizon=1):
    """
    Rank IC:
        Spearman correlation between signal_t and future return.
    """

    future_return = close.shift(-forward_horizon) / close - 1.0

    ic_list = []

    for t in signal.index:
        s = signal.loc[t]
        r = future_return.loc[t]

        valid = s.notna() & r.notna()

        s = s[valid]
        r = r[valid]

        # Need enough cross-sectional assets
        if len(s) < 5:
            continue

        # If signal or return has no variation, skip
        if s.std() == 0 or r.std() == 0:
            continue

        ic = s.corr(r, method="spearman")

        if pd.notna(ic):
            ic_list.append(ic)

    ic_series = pd.Series(ic_list, name="Rank_IC")

    if ic_series.empty:
        return {
            "mean_ic": np.nan,
            "median_ic": np.nan,
            "ic_std": np.nan,
            "icir": np.nan,
            "positive_ic_ratio": np.nan,
            "n_obs": 0,
            "ic_series": ic_series,
        }

    mean_ic = ic_series.mean()
    ic_std = ic_series.std()
    icir = mean_ic / ic_std if ic_std != 0 else np.nan

    return {
        "mean_ic": mean_ic,
        "median_ic": ic_series.median(),
        "ic_std": ic_std,
        "icir": icir,
        "positive_ic_ratio": (ic_series > 0).mean(),
        "n_obs": len(ic_series),
        "ic_series": ic_series,
    }


def run_ic_test(data_path, windows_list, forward_horizons=(1, 3, 6, 12)):
    data = load_data(data_path)

    close = extract_field(data, "close")
    volume = extract_field(data, "volume")

    rows = []

    for windows in windows_list:
        signal = build_momentum_signal(
            close=close,
            volume=volume,
            momentum_windows=windows,
            use_negative_sign=True,
        )

        for horizon in forward_horizons:
            result = calculate_rank_ic(
                signal=signal,
                close=close,
                forward_horizon=horizon,
            )

            rows.append({
                "momentum_windows": str(windows),
                "forward_horizon": horizon,
                "mean_ic": result["mean_ic"],
                "median_ic": result["median_ic"],
                "ic_std": result["ic_std"],
                "icir": result["icir"],
                "positive_ic_ratio": result["positive_ic_ratio"],
                "n_obs": result["n_obs"],
            })

    summary = pd.DataFrame(rows)
    return summary


if __name__ == "__main__":
    data_path = "data/mini1/validation.parquet"

    windows_list = [
        (3, 6),
        (4, 8),
        (5, 10),
    ]

    summary = run_ic_test(
        data_path=data_path,
        windows_list=windows_list,
        forward_horizons=(1, 3, 6, 12, 78),
    )

    print("\n=== Rank IC Summary ===")
    print(summary.to_string(index=False))

    output_path = Path("tools/output/ic_summary.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print(f"\nSaved IC summary to: {output_path}")