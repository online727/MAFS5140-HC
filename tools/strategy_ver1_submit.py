from collections import deque

import numpy as np
import pandas as pd


class MomentumStrategy:
    def __init__(self):
        self.momentum_windows = (6, 12)
        self.volume_window = 78
        self.min_relative_volume = 0.8
        self.volume_cap = 2.0
        self.top_selector = 20.0
        self.max_weight = 0.05

        self.max_price_history = max(self.momentum_windows) + 1
        self.price_history = deque(maxlen=self.max_price_history)
        self.volume_history = deque(maxlen=self.volume_window)
        self.tickers = None

    def _cross_sectional_zscore(self, values: pd.Series) -> pd.Series:
        mean = values.mean()
        std = values.std()
        if pd.isna(std) or std == 0.0:
            return pd.Series(0.0, index=values.index)
        zscore = (values - mean) / std
        return zscore.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def _selection_count(self, n_assets: int) -> int:
        if np.isclose(self.top_selector, 1.0) or np.isclose(self.top_selector, float(n_assets)):
            return n_assets
        if self.top_selector < 1.0:
            return min(max(int(np.ceil(self.top_selector * n_assets)), 1), n_assets)
        return min(max(int(round(self.top_selector)), 1), n_assets)

    def _build_scores(self) -> pd.Series:
        price_frame = pd.DataFrame(self.price_history)
        volume_frame = pd.DataFrame(self.volume_history)

        momentum_parts = []
        latest_prices = price_frame.iloc[-1]
        for window in self.momentum_windows:
            start_idx = max(0, len(price_frame) - 1 - window)
            past_prices = price_frame.iloc[start_idx]
            raw_return = (latest_prices / past_prices) - 1.0
            momentum_parts.append(self._cross_sectional_zscore(raw_return))

        momentum_score = sum(momentum_parts) / len(momentum_parts)

        effective_volume_window = min(len(volume_frame), self.volume_window)
        average_volume = volume_frame.iloc[-effective_volume_window:].mean()
        relative_volume = volume_frame.iloc[-1] / average_volume.replace(0.0, np.nan)
        relative_volume = relative_volume.replace([np.inf, -np.inf], np.nan)
        volume_multiplier = relative_volume.clip(lower=0.0, upper=self.volume_cap) / self.volume_cap
        volume_multiplier = volume_multiplier.where(relative_volume >= self.min_relative_volume, 0.0)
        volume_multiplier = volume_multiplier.fillna(0.0)

        scores = momentum_score * volume_multiplier
        return - scores.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def _build_weights(self, scores: pd.Series) -> pd.Series:
        weights = pd.Series(0.0, index=scores.index)
        if sum(scores > 0.0) == 0:
            return weights
        selection_count = self._selection_count(len(scores))

        if selection_count >= len(scores):
            selected = scores.index
        else:
            selected = scores.nlargest(selection_count).index

        selected_scores = scores.loc[selected]
        positive_scores = selected_scores[selected_scores > 0.0]

        if positive_scores.empty:
            raw_weights = pd.Series(1.0 / len(selected), index=selected)
        else:
            raw_weights = positive_scores / positive_scores.sum()

        clipped_weights = raw_weights.clip(upper=self.max_weight)
        weights.loc[clipped_weights.index] = clipped_weights
        return weights

    def step(self, current_market_data: pd.DataFrame) -> pd.Series:
        if "close" not in current_market_data.columns or "volume" not in current_market_data.columns:
            raise ValueError("Input market data must contain both 'close' and 'volume' columns.")

        current_market_data = current_market_data.copy()
        current_market_data["close"] = current_market_data["close"].astype(float)
        current_market_data["volume"] = current_market_data["volume"].astype(float)

        current_index = current_market_data.index
        if self.tickers is None:
            self.tickers = current_index.copy()
        else:
            if set(current_index) != set(self.tickers):
                raise ValueError("Ticker universe changed during backtest.")
            current_market_data = current_market_data.reindex(self.tickers)
        
        if len(self.price_history) == 0:
            for _ in range(self.max_price_history - 1):
                self.price_history.append(current_market_data["close"])

        self.price_history.append(current_market_data["close"])
        self.volume_history.append(current_market_data["volume"])

        required_prices = min(self.momentum_windows)
        if len(self.price_history) < required_prices:
            return pd.Series(0.0, index=current_index)

        scores = self._build_scores()
        weights = self._build_weights(scores)

        if (weights < 0.0).any():
            raise ValueError("Generated negative weights, which is not allowed.")
        if weights.sum() > 1.000001:
            raise ValueError("Generated weights exceed the total portfolio limit of 1.0.")

        return weights.reindex(current_index).fillna(0.0)
