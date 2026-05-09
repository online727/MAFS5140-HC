from __future__ import annotations

from collections import deque
import math

import numpy as np
import pandas as pd
import json

EPSILON = 1e-12
DEFAULT_TOP_SELECTOR = 10.0
DEFAULT_MAX_WEIGHT = 0.10


def load_feature_sets_from_json(path: str) -> dict[str, list[tuple[str, str]]]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("JSON content must be a dictionary of feature sets.")
    feature_sets = {}
    for set_name, features in data.items():
        if not isinstance(set_name, str):
            raise ValueError(f"Feature set name must be a string: {set_name}")
        if not isinstance(features, list):
            raise ValueError(f"Features for set '{set_name}' must be a list.")
        parsed_features = []
        for item in features:
            if (not isinstance(item, list) and not isinstance(item, tuple)) or len(item) != 2:
                raise ValueError(f"Each feature entry must be a list or tuple of (feature_name, direction): {item}")
            feature_name, direction = item
            if not isinstance(feature_name, str) or direction not in ("raw", "neg"):
                raise ValueError(f"Invalid feature entry: {item}. Direction must be 'raw' or 'neg'.")
            parsed_features.append((feature_name, direction))
        feature_sets[set_name] = parsed_features
    return feature_sets

BUILTIN_FEATURE_SETS: dict[str, list[tuple[str, str]]] = load_feature_sets_from_json("feature_sets.json")
FEATURE_SETS_WEIGHTS: pd.DataFrame = pd.read_csv("feature_sets_weights.csv", index_col=0)
DEFAULT_WEIGHTS_COL: str = "sharpe_ratio"

def _clean_feature_series(values: pd.Series, index: pd.Index | None = None) -> pd.Series:
    if index is not None:
        values = values.reindex(index)
    return values.astype(float).replace([np.inf, -np.inf], np.nan)


def _clean_weight_series(values: pd.Series, index: pd.Index | None = None) -> pd.Series:
    return _clean_feature_series(values, index=index).fillna(0.0)


def _zero_series(index: pd.Index) -> pd.Series:
    return pd.Series(0.0, index=index, dtype=float)


def _nan_series(index: pd.Index) -> pd.Series:
    return pd.Series(np.nan, index=index, dtype=float)


def _safe_divide(numerator: pd.Series, denominator: pd.Series | float) -> pd.Series:
    return numerator.div(denominator).replace([np.inf, -np.inf], np.nan)


def cross_sectional_zscore(values: pd.Series) -> pd.Series:
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan)
    mean = clean.mean()
    std = clean.std()
    if pd.isna(std) or std == 0.0:
        return _zero_series(values.index)
    return _clean_weight_series((clean - mean) / std, values.index)


def resolve_selection_count(top_selector: float, n_assets: int) -> int:
    if n_assets <= 0:
        raise ValueError("n_assets must be positive.")
    selector = float(top_selector)
    if np.isclose(selector, 1.0) or np.isclose(selector, float(n_assets)):
        return n_assets
    if selector < 1.0:
        return min(max(int(math.ceil(selector * n_assets)), 1), n_assets)
    return min(max(int(round(selector)), 1), n_assets)


def build_top_score_weights(scores: pd.Series, top_selector: float, max_weight: float) -> pd.Series:
    if max_weight <= 0.0:
        raise ValueError("max_weight must be positive.")

    clean = _clean_weight_series(scores, scores.index)
    weights = _zero_series(clean.index)
    if clean.empty:
        return weights

    selection_count = resolve_selection_count(top_selector, len(clean))
    selected = clean.index if selection_count >= len(clean) else clean.nlargest(selection_count).index
    selected_scores = clean.loc[selected]
    positive_scores = selected_scores[selected_scores > 0.0]
    if positive_scores.empty:
        return weights

    raw_weights = positive_scores / positive_scores.sum()
    clipped = raw_weights.clip(upper=max_weight)
    weights.loc[clipped.index] = clipped
    return validate_weights(weights, max_weight=max_weight)


def validate_weights(weights: pd.Series, max_weight: float | None = None) -> pd.Series:
    out = _clean_weight_series(weights.clip(lower=0.0), weights.index)
    if max_weight is not None:
        out = out.clip(upper=max_weight)
    total = float(out.sum())
    if total > 1.0 + EPSILON:
        out = out / total
        if max_weight is not None:
            out = out.clip(upper=max_weight)
    return _clean_weight_series(out, weights.index)


def normalize_named_weights(raw: dict[str, float]) -> dict[str, float]:
    clean = {str(k): max(float(v), 0.0) for k, v in raw.items()}
    total = sum(clean.values())
    if total <= 0.0:
        raise ValueError("At least one feature set weight must be positive.")
    return {k: v / total for k, v in clean.items() if v > 0.0}


class StreamingMomentumFeatureEngine:
    def __init__(self, max_history: int = 240):
        self.max_history = int(max_history)
        self.close_history: deque[pd.Series] = deque(maxlen=self.max_history)
        self.volume_history: deque[pd.Series] = deque(maxlen=self.max_history)
        self.return_history: deque[pd.Series] = deque(maxlen=self.max_history)
        self.close_diff_history: deque[pd.Series] = deque(maxlen=self.max_history)
        self.volume_change_history: deque[pd.Series] = deque(maxlen=self.max_history)
        self.dollar_volume_history: deque[pd.Series] = deque(maxlen=self.max_history)
        self.obv_history: deque[pd.Series] = deque(maxlen=self.max_history)
        self.ema_history: dict[int, deque[pd.Series]] = {}
        self.ema_values: dict[int, pd.Series] = {}
        self.macd_signal_values: dict[tuple[int, int, int], pd.Series] = {}
        self.macd_valid_counts: dict[tuple[int, int, int], int] = {}
        self.last_nonzero_direction: pd.Series | None = None
        self.tickers: pd.Index | None = None
        self.step_count = 0
        self._frame_cache: dict[str, pd.DataFrame] = {}

    def update(self, current_market_data: pd.DataFrame) -> None:
        if "close" not in current_market_data.columns or "volume" not in current_market_data.columns:
            raise ValueError("Input market data must contain both 'close' and 'volume' columns.")

        current = current_market_data.copy()
        current["close"] = current["close"].astype(float)
        current["volume"] = current["volume"].astype(float)

        if self.tickers is None:
            self.tickers = pd.Index(current.index)
            self.last_nonzero_direction = _zero_series(self.tickers)
        else:
            if set(current.index) != set(self.tickers):
                raise ValueError("Ticker universe changed during backtest.")
            current = current.reindex(self.tickers)

        close = _clean_feature_series(current["close"], self.tickers)
        volume = _clean_feature_series(current["volume"], self.tickers)
        prev_close = self.close_history[-1] if self.close_history else None

        if prev_close is None:
            returns = _nan_series(self.tickers)
            close_diff = _nan_series(self.tickers)
            volume_change = _nan_series(self.tickers)
            direction = _zero_series(self.tickers)
            obv = _zero_series(self.tickers)
        else:
            close_diff = close - prev_close
            returns = _safe_divide(close, prev_close.replace(0.0, np.nan)) - 1.0
            prev_volume = self.volume_history[-1]
            volume_change = _safe_divide(volume, prev_volume.replace(0.0, np.nan)) - 1.0
            raw_direction = np.sign(close_diff)
            direction = self.last_nonzero_direction.where(raw_direction == 0.0, raw_direction)
            direction = _clean_weight_series(direction, self.tickers)
            obv = self.obv_history[-1] + direction * volume if self.obv_history else direction * volume
            self.last_nonzero_direction = direction

        self.close_history.append(close)
        self.volume_history.append(volume)
        self.return_history.append(_clean_feature_series(returns, self.tickers))
        self.close_diff_history.append(_clean_feature_series(close_diff, self.tickers))
        self.volume_change_history.append(_clean_feature_series(volume_change, self.tickers))
        self.dollar_volume_history.append(close * volume)
        self.obv_history.append(_clean_weight_series(obv, self.tickers))
        self.step_count += 1
        self._frame_cache = {}

    def _frame(self, key: str, history: deque[pd.Series]) -> pd.DataFrame:
        if key not in self._frame_cache:
            frame = pd.DataFrame(list(history))
            frame.index = range(len(frame))
            self._frame_cache[key] = frame
        return self._frame_cache[key]

    @property
    def close(self) -> pd.Series:
        return self.close_history[-1]

    @property
    def volume(self) -> pd.Series:
        return self.volume_history[-1]

    def _window_frame(self, key: str, history: deque[pd.Series], window: int) -> pd.DataFrame | None:
        if len(history) < window:
            return None
        return self._frame(key, history).iloc[-window:]

    def _rolling_mean(self, key: str, history: deque[pd.Series], window: int) -> pd.Series:
        frame = self._frame(key, history)
        value = frame.rolling(window=window, min_periods=window).mean().iloc[-1]
        return _clean_feature_series(value, self.tickers)

    def _rolling_sum(self, key: str, history: deque[pd.Series], window: int) -> pd.Series:
        frame = self._frame(key, history)
        value = frame.rolling(window=window, min_periods=window).sum().iloc[-1]
        return _clean_feature_series(value, self.tickers)

    def _rolling_std(self, key: str, history: deque[pd.Series], window: int) -> pd.Series:
        frame = self._frame(key, history)
        value = frame.rolling(window=window, min_periods=window).std().iloc[-1]
        return _clean_feature_series(value, self.tickers)

    def _time_series_zscore(self, key: str, history: deque[pd.Series], window: int) -> pd.Series:
        frame = self._frame(key, history)
        mean = frame.rolling(window=window, min_periods=window).mean().iloc[-1]
        std = frame.rolling(window=window, min_periods=window).std().iloc[-1].replace(0.0, np.nan)
        value = (frame.iloc[-1] - mean) / std
        return _clean_feature_series(value, self.tickers)

    def _pct_change_from_history(self, history: deque[pd.Series], window: int) -> pd.Series:
        if len(history) <= window:
            return _nan_series(self.tickers)
        past = list(history)[-window - 1]
        return _clean_feature_series(_safe_divide(history[-1], past.replace(0.0, np.nan)) - 1.0, self.tickers)

    def _diff_from_history(self, history: deque[pd.Series], window: int) -> pd.Series:
        if len(history) <= window:
            return _nan_series(self.tickers)
        past = list(history)[-window - 1]
        return _clean_feature_series(history[-1] - past, self.tickers)

    def _rolling_minmax(self, window: int) -> tuple[pd.Series, pd.Series] | None:
        frame = self._window_frame("close", self.close_history, window)
        if frame is None:
            return None
        return frame.min(), frame.max()

    def _ema(self, span: int) -> pd.Series:
        alpha = 2.0 / (span + 1.0)
        if span not in self.ema_values:
            self.ema_values[span] = self.close.copy()
            self.ema_history[span] = deque(maxlen=self.max_history)
        else:
            self.ema_values[span] = alpha * self.close + (1.0 - alpha) * self.ema_values[span]
        self.ema_history[span].append(_clean_feature_series(self.ema_values[span], self.tickers))
        return self.ema_values[span]

    def update_recursive_indicators(self, required_features: set[str]) -> None:
        spans = set()
        macd_specs = []
        for name in required_features:
            parts = name.split("_")
            if name.startswith("ema_dist_") or name.startswith("ema_slope_"):
                spans.add(int(parts[-1]))
            elif name.startswith("ma_cross_"):
                continue
            elif name.startswith("macd_"):
                fast, slow, signal = map(int, parts[1:4])
                spans.update((fast, slow))
                macd_specs.append((fast, slow, signal))

        for span in sorted(spans):
            self._ema(span)

        for fast, slow, signal in macd_specs:
            key = (fast, slow, signal)
            macd_line = self.ema_values[fast] - self.ema_values[slow]
            if self.step_count < slow:
                continue
            if key not in self.macd_signal_values:
                self.macd_signal_values[key] = macd_line.copy()
                self.macd_valid_counts[key] = 1
            else:
                alpha = 2.0 / (signal + 1.0)
                self.macd_signal_values[key] = alpha * macd_line + (1.0 - alpha) * self.macd_signal_values[key]
                self.macd_valid_counts[key] += 1

    def compute_feature(self, name: str) -> pd.Series:
        parts = name.split("_")

        if name.startswith("rolling_mean_ret_"):
            return self._rolling_mean("returns", self.return_history, int(parts[-1]))
        if name.startswith("rolling_sum_ret_"):
            return self._rolling_sum("returns", self.return_history, int(parts[-1]))
        if name.startswith("ret_zscore_ts_"):
            return self._time_series_zscore("returns", self.return_history, int(parts[-1]))
        if name.startswith("price_zscore_ts_"):
            return self._time_series_zscore("close", self.close_history, int(parts[-1]))

        if name.startswith("ret_"):
            return self._pct_change_from_history(self.close_history, int(parts[-1]))
        if name.startswith("logret_"):
            if len(self.close_history) <= int(parts[-1]):
                return _nan_series(self.tickers)
            past = list(self.close_history)[-int(parts[-1]) - 1]
            value = np.log(self.close.replace(0.0, np.nan)) - np.log(past.replace(0.0, np.nan))
            return _clean_feature_series(value, self.tickers)
        if name.startswith("roc_"):
            window = int(parts[-1])
            diff = self._diff_from_history(self.close_history, window)
            if len(self.close_history) <= window:
                return _nan_series(self.tickers)
            past = list(self.close_history)[-window - 1]
            return _clean_feature_series(diff.div(past.replace(0.0, np.nan)), self.tickers)
        if name.startswith("volume_change_"):
            return self._pct_change_from_history(self.volume_history, int(parts[-1]))

        if name.startswith("sma_dist_"):
            window = int(parts[-1])
            sma = self._rolling_mean("close", self.close_history, window)
            return _clean_feature_series(self.close.div(sma.replace(0.0, np.nan)) - 1.0, self.tickers)
        if name.startswith("ema_dist_"):
            window = int(parts[-1])
            if self.step_count < window or window not in self.ema_values:
                return _nan_series(self.tickers)
            return _clean_feature_series(self.close.div(self.ema_values[window].replace(0.0, np.nan)) - 1.0, self.tickers)
        if name.startswith("sma_slope_"):
            window = int(parts[-1])
            if len(self.close_history) < 2 * window:
                return _nan_series(self.tickers)
            close_frame = self._frame("close", self.close_history)
            current_sma = close_frame.iloc[-window:].mean()
            past_sma = close_frame.iloc[-2 * window:-window].mean()
            return _clean_feature_series(current_sma.div(past_sma.replace(0.0, np.nan)) - 1.0, self.tickers)
        if name.startswith("ema_slope_"):
            window = int(parts[-1])
            if self.step_count < 2 * window or window not in self.ema_history or len(self.ema_history[window]) <= window:
                return _nan_series(self.tickers)
            past_ema = list(self.ema_history[window])[-window - 1]
            return _clean_feature_series(self.ema_values[window].div(past_ema.replace(0.0, np.nan)) - 1.0, self.tickers)
        if name.startswith("ma_cross_"):
            fast, slow = map(int, parts[-2:])
            fast_sma = self._rolling_mean("close", self.close_history, fast)
            slow_sma = self._rolling_mean("close", self.close_history, slow)
            return _clean_feature_series(fast_sma.div(slow_sma.replace(0.0, np.nan)) - 1.0, self.tickers)

        if name.startswith("rsi_"):
            window = int(parts[-1])
            frame = self._window_frame("close_diff", self.close_diff_history, window)
            if frame is None:
                return _nan_series(self.tickers)
            gain = frame.clip(lower=0.0).mean()
            loss = (-frame.clip(upper=0.0)).mean()
            valid = frame.count() >= window
            rs = gain.div(loss.replace(0.0, np.nan))
            value = 100.0 - (100.0 / (1.0 + rs))
            value = value.where(loss != 0.0, 100.0)
            value = value.where(gain != 0.0, 0.0)
            value = value.where(valid)
            return _clean_feature_series((value - 50.0) / 50.0, self.tickers)
        if name.startswith("stoch_close_"):
            window = int(parts[-1])
            minmax = self._rolling_minmax(window)
            if minmax is None:
                return _nan_series(self.tickers)
            roll_min, roll_max = minmax
            return _clean_feature_series((self.close - roll_min).div((roll_max - roll_min).replace(0.0, np.nan)) - 0.5, self.tickers)
        if name.startswith("macd_"):
            fast, slow, signal = map(int, parts[1:4])
            key = (fast, slow, signal)
            if key not in self.macd_signal_values or self.macd_valid_counts.get(key, 0) < signal:
                return _nan_series(self.tickers)
            macd_line = self.ema_values[fast] - self.ema_values[slow]
            hist = macd_line - self.macd_signal_values[key]
            return _clean_feature_series(hist.div(self.close.replace(0.0, np.nan)), self.tickers)

        if name.startswith("relative_volume_"):
            window = int(parts[-1])
            avg_volume = self._rolling_mean("volume", self.volume_history, window)
            return _clean_feature_series(self.volume.div(avg_volume.replace(0.0, np.nan)), self.tickers)
        if name.startswith("volume_zscore_"):
            return self._time_series_zscore("volume", self.volume_history, int(parts[-1]))
        if name.startswith("dollar_volume_zscore_"):
            return self._time_series_zscore("dollar_volume", self.dollar_volume_history, int(parts[-1]))
        if name.startswith("price_volume_corr_"):
            window = int(parts[-1])
            ret_frame = self._window_frame("returns", self.return_history, window)
            vol_frame = self._window_frame("volume_change", self.volume_change_history, window)
            if ret_frame is None or vol_frame is None:
                return _nan_series(self.tickers)
            value = ret_frame.corrwith(vol_frame)
            valid = (ret_frame.notna() & vol_frame.notna()).sum() >= window
            return _clean_feature_series(value.where(valid), self.tickers)
        if name.startswith("obv_slope_"):
            window = int(parts[-1])
            if len(self.obv_history) <= window or len(self.volume_history) < window:
                return _nan_series(self.tickers)
            obv_past = list(self.obv_history)[-window - 1]
            scale = self._rolling_sum("volume", self.volume_history, window)
            return _clean_feature_series((self.obv_history[-1] - obv_past).div(scale.replace(0.0, np.nan)), self.tickers)

        if name.startswith("realized_vol_"):
            return self._rolling_std("returns", self.return_history, int(parts[-1]))
        if name.startswith("downside_vol_"):
            window = int(parts[-1])
            frame = self._window_frame("returns", self.return_history, window)
            if frame is None:
                return _nan_series(self.tickers)
            downside = frame.where(frame < 0.0, 0.0)
            return _clean_feature_series(downside.std(), self.tickers)
        if name.startswith("return_range_"):
            window = int(parts[-1])
            minmax = self._rolling_minmax(window)
            if minmax is None:
                return _nan_series(self.tickers)
            roll_min, roll_max = minmax
            return _clean_feature_series(roll_max.div(roll_min.replace(0.0, np.nan)) - 1.0, self.tickers)
        if name.startswith("vol_adjusted_ret_"):
            window = int(parts[-1])
            ret = self._pct_change_from_history(self.close_history, window)
            vol = self._rolling_std("returns", self.return_history, window)
            return _clean_feature_series(ret.div(vol.replace(0.0, np.nan)), self.tickers)

        raise ValueError(f"Unsupported feature: {name}")


class Strategy:
    def __init__(
        self,
        set_names: list[str] | tuple[str, ...] | None = None,
        set_weights: dict[str, float] | None = None,
        top_selector: float = DEFAULT_TOP_SELECTOR,
        max_weight: float = DEFAULT_MAX_WEIGHT,
        feature_limit: int | None = None,
    ):
        self.feature_sets = BUILTIN_FEATURE_SETS
        self.top_selector = top_selector
        self.max_weight = max_weight
        self.feature_limit = feature_limit

        if set_weights is not None:
            if set_names is not None:
                names = set(str(name) for name in set_names)
                raw_weights = {name: weight for name, weight in set_weights.items() if name in names}
            else:
                raw_weights = dict(set_weights)
        else:
            if set_names is None:
                raw_weights = {"diversified_best": 1.0}
            else:
                raw_weights = {str(name): FEATURE_SETS_WEIGHTS.loc[name, DEFAULT_WEIGHTS_COL] for name in set_names}

        missing = sorted(set(raw_weights).difference(self.feature_sets))
        if missing:
            raise ValueError(f"Unknown feature set names: {missing}. Available: {sorted(self.feature_sets)}")
        self.set_weights = normalize_named_weights(raw_weights)
        print(f"Using feature sets with weights: {self.set_weights}")

        required_features = set()
        for set_name in self.set_weights:
            selected = self._selected_features(set_name)
            required_features.update(feature for feature, _ in selected)

        self.required_features = required_features
        self.engine = StreamingMomentumFeatureEngine()
        self.last_feature_values: dict[str, pd.Series] = {}
        self.last_set_scores: dict[str, pd.Series] = {}
        self.last_set_weights: dict[str, pd.Series] = {}
        self.last_combined_weights: pd.Series | None = None

    def _selected_features(self, set_name: str) -> list[tuple[str, str]]:
        selected = self.feature_sets[set_name]
        if self.feature_limit is not None:
            return selected[: self.feature_limit]
        return selected

    def _compute_feature_cache(self) -> dict[str, pd.Series]:
        cache = {}
        for name in self.required_features:
            cache[name] = self.engine.compute_feature(name)
        return cache

    def _build_set_score(self, set_name: str, feature_cache: dict[str, pd.Series]) -> pd.Series:
        components = []
        for feature_name, direction in self._selected_features(set_name):
            values = feature_cache[feature_name]
            if direction == "neg":
                values = -values
            elif direction != "raw":
                raise ValueError(f"Unsupported direction: {direction}")
            components.append(cross_sectional_zscore(values))

        if not components:
            return _zero_series(self.engine.tickers)

        score = sum(components) / len(components)
        return _clean_weight_series(score, self.engine.tickers)

    def step(self, current_market_data: pd.DataFrame) -> pd.Series:
        current_index = pd.Index(current_market_data.index)
        self.engine.update(current_market_data)
        self.engine.update_recursive_indicators(self.required_features)

        feature_cache = self._compute_feature_cache()
        self.last_feature_values = feature_cache
        self.last_set_scores = {}
        self.last_set_weights = {}

        combined = _zero_series(self.engine.tickers)
        for set_name, set_weight in self.set_weights.items():
            score = self._build_set_score(set_name, feature_cache)
            weights = build_top_score_weights(
                scores=score,
                top_selector=self.top_selector,
                max_weight=self.max_weight,
            )
            self.last_set_scores[set_name] = score
            self.last_set_weights[set_name] = weights
            combined = combined + set_weight * weights

        combined = validate_weights(combined, max_weight=self.max_weight)
        self.last_combined_weights = combined
        return combined.reindex(current_index).fillna(0.0)
