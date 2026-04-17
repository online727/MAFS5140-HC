import pandas as pd
import numpy as np
import scipy.sparse as sp
from qpsolvers import solve_qp
from collections import deque


class MovingAverageStrategy:
    def __init__(self):
        self.price_history = []
        self.lookback_period = 78

    def step(self, current_market_data: pd.DataFrame) -> pd.Series:
        if "close" not in current_market_data.columns:
            raise ValueError("Input market data must contain a 'close' column.")

        current_prices = current_market_data["close"]
        self.price_history.append(current_prices)

        if len(self.price_history) > self.lookback_period:
            self.price_history.pop(0)

        if len(self.price_history) < self.lookback_period:
            return pd.Series(0.0, index=current_prices.index)

        history_df = pd.DataFrame(self.price_history)
        moving_average = history_df.mean()

        bullish_assets = current_prices[current_prices > moving_average].index

        weights = pd.Series(0.0, index=current_prices.index)
        num_bullish = len(bullish_assets)
        if num_bullish > 0:
            weights[bullish_assets] = 1.0 / num_bullish

        return weights


class EqualWeightStrategy:
    def __init__(self):
        pass

    def step(self, current_market_data: pd.DataFrame) -> pd.Series:
        n_assets = current_market_data.shape[0]
        if n_assets == 0:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / n_assets, index=current_market_data.index)


class MomentumStrategy:
    """
    Momentum strategy used as a signal filter, not the final allocator.
    It builds cross-sectional momentum scores with a volume confirmation filter.
    """

    def __init__(self):
        self.momentum_windows = (3, 6)
        self.volume_window = 78
        self.min_relative_volume = 0.8
        self.volume_cap = 2.0
        self.top_selector = 20
        self.max_weight = 0.10

        self.max_price_history = max(self.momentum_windows) + 1
        self.price_history = deque(maxlen=self.max_price_history)
        self.volume_history = deque(maxlen=self.volume_window)
        self.tickers = None

    def _cross_sectional_zscore(self, values: pd.Series) -> pd.Series:
        mean = values.mean()
        std = values.std()
        if pd.isna(std) or std == 0.0:
            return pd.Series(0.0, index=values.index)

        z = (values - mean) / std
        return z.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def _selection_count(self, n_assets: int) -> int:
        if n_assets <= 0:
            return 0
        if np.isclose(self.top_selector, 1.0) or np.isclose(self.top_selector, float(n_assets)):
            return n_assets
        if self.top_selector < 1.0:
            return min(max(int(np.ceil(self.top_selector * n_assets)), 1), n_assets)
        return min(max(int(round(self.top_selector)), 1), n_assets)

    def _update_history(self, current_market_data: pd.DataFrame) -> pd.DataFrame:
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

        return current_market_data

    def _build_scores(self) -> pd.Series:
        price_frame = pd.DataFrame(self.price_history)
        volume_frame = pd.DataFrame(self.volume_history)

        latest_prices = price_frame.iloc[-1]
        momentum_parts = []

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
        scores = -scores
        return scores.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def get_selected_universe(self, current_market_data: pd.DataFrame):
        if "close" not in current_market_data.columns or "volume" not in current_market_data.columns:
            raise ValueError("Input market data must contain both 'close' and 'volume' columns.")

        current_market_data = self._update_history(current_market_data)
        current_index = current_market_data.index

        required_prices = min(self.momentum_windows)
        if len(self.price_history) < required_prices:
            return [], pd.Series(0.0, index=current_index)

        scores = self._build_scores()
        positive_scores = scores[scores > 0.0]

        if positive_scores.empty:
            return [], scores.reindex(current_index).fillna(0.0)

        selection_count = self._selection_count(len(scores))
        selected = positive_scores.nlargest(selection_count).index.tolist()

        return selected, scores.reindex(current_index).fillna(0.0)

    def step(self, current_market_data: pd.DataFrame) -> pd.Series:
        """
        Kept for compatibility/debugging.
        The final Strategy class does NOT use this as final portfolio weights.
        """
        selected, scores = self.get_selected_universe(current_market_data)
        weights = pd.Series(0.0, index=current_market_data.index)

        if len(selected) == 0:
            return weights

        selected_scores = scores.loc[selected]
        selected_scores = selected_scores[selected_scores > 0.0]

        if selected_scores.empty:
            return weights

        raw_weights = selected_scores / selected_scores.sum()
        clipped_weights = raw_weights.clip(upper=self.max_weight)
        weights.loc[clipped_weights.index] = clipped_weights

        if weights.sum() > 1.0:
            weights = weights / weights.sum()

        return weights.fillna(0.0)


class MeanVarianceStrategy:
    """
    Shrinkage mean-variance optimizer used after momentum filtering.
    """

    def __init__(self):
        self.lookback_period = 78 * 5
        self.min_period = 78 * 5
        self.price_history = deque(maxlen=self.lookback_period + 1)

        self.gamma = 10.0
        self.mean_shrinkage = 0.2
        self.cov_shrinkage = 0.2
        self.max_weight_per_asset = 0.10
        self.ridge = 1e-6

    def _estimate_sigma(self, history_df: pd.DataFrame):
        returns = history_df.pct_change().dropna()
        if returns.empty:
            return None

        sigma_hat = returns.cov()
        sigma_hat = sigma_hat.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        n = sigma_hat.shape[0]
        if n == 0:
            return None

        sigma_target = pd.DataFrame(
            np.diag(np.diag(sigma_hat.values)),
            index=sigma_hat.index,
            columns=sigma_hat.columns
        )

        sigma_shrunk = (1.0 - self.cov_shrinkage) * sigma_hat + self.cov_shrinkage * sigma_target
        sigma_shrunk = sigma_shrunk + self.ridge * np.eye(n)

        return sigma_shrunk.values

    def _estimate_sample_mu(self, history_df: pd.DataFrame):
        returns = history_df.pct_change().dropna()
        if returns.empty:
            return None

        mu_hat = returns.mean()
        mu_hat = mu_hat.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        mu_target = pd.Series(0.0, index=mu_hat.index)
        mu_shrunk = (1.0 - self.mean_shrinkage) * mu_hat + self.mean_shrinkage * mu_target

        return mu_shrunk.values

    def _solve_mean_variance(self, mu: np.ndarray, sigma: np.ndarray, tickers) -> pd.Series:
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        n = len(mu)

        if n == 0:
            return pd.Series(dtype=float)

        P = sp.csc_matrix(self.gamma * sigma)
        q = -mu

        G = sp.vstack([
            -sp.eye(n, format="csc"),         # w >= 0
            sp.eye(n, format="csc"),          # w <= max_weight
            sp.csc_matrix(np.ones((1, n)))    # sum(w) <= 1
        ], format="csc")

        h = np.concatenate([
            np.zeros(n),
            np.full(n, self.max_weight_per_asset),
            np.array([1.0])
        ])

        w = solve_qp(P, q, G, h, solver="osqp")

        if w is None:
            return pd.Series(0.0, index=tickers)

        w = np.clip(w, 0.0, self.max_weight_per_asset)

        if w.sum() > 1.0:
            w = w / w.sum()

        return pd.Series(w, index=tickers)

    def step(self, current_market_data: pd.DataFrame, selected_tickers=None, alpha_scores=None) -> pd.Series:
        if "close" not in current_market_data.columns:
            raise ValueError("Input market data must contain a 'close' column.")

        current_prices = current_market_data["close"].astype(float)
        self.price_history.append(current_prices)

        if len(self.price_history) < self.min_period + 1:
            return pd.Series(0.0, index=current_prices.index)

        history_df = pd.DataFrame(self.price_history)

        if selected_tickers is None or len(selected_tickers) == 0:
            return pd.Series(0.0, index=current_prices.index)

        selected_tickers = [ticker for ticker in selected_tickers if ticker in history_df.columns]
        if len(selected_tickers) == 0:
            return pd.Series(0.0, index=current_prices.index)

        history_df_sub = history_df[selected_tickers]

        if alpha_scores is not None:
            alpha_sub = alpha_scores.reindex(selected_tickers).fillna(0.0)
            alpha_sub = alpha_sub.replace([np.inf, -np.inf], 0.0).fillna(0.0)
            alpha_sub = alpha_sub[alpha_sub > 0.0]

            if alpha_sub.empty:
                return pd.Series(0.0, index=current_prices.index)

            selected_tickers = list(alpha_sub.index)
            history_df_sub = history_df[selected_tickers]

            mu = alpha_sub.values.astype(float)
            mu_norm = np.linalg.norm(mu)
            if mu_norm > 0:
                mu = mu / mu_norm
        else:
            mu = self._estimate_sample_mu(history_df_sub)
            if mu is None:
                return pd.Series(0.0, index=current_prices.index)

        sigma = self._estimate_sigma(history_df_sub)
        if sigma is None:
            return pd.Series(0.0, index=current_prices.index)

        sub_weights = self._solve_mean_variance(mu, sigma, selected_tickers)

        weights = pd.Series(0.0, index=current_prices.index)
        weights.loc[sub_weights.index] = sub_weights

        weights = weights.reindex(current_prices.index).fillna(0.0)
        weights = weights.clip(lower=0.0, upper=self.max_weight_per_asset)

        if weights.sum() > 1.0:
            weights = weights / weights.sum()

        return weights


class Strategy:
    def __init__(self):
        self.mom_strategy = MomentumStrategy()
        self.mean_variance_strategy = MeanVarianceStrategy()

    def step(self, current_market_data: pd.DataFrame) -> pd.Series:
        selected_tickers, scores = self.mom_strategy.get_selected_universe(current_market_data)

        if len(selected_tickers) == 0:
            return pd.Series(0.0, index=current_market_data.index)

        weights = self.mean_variance_strategy.step(
            current_market_data=current_market_data,
            selected_tickers=selected_tickers,
            alpha_scores=scores
        )

        weights = weights.reindex(current_market_data.index).fillna(0.0)

        if (weights < 0.0).any():
            raise ValueError("Generated negative weights, which is not allowed.")
        if weights.sum() > 1.000001:
            raise ValueError("Generated weights exceed the total portfolio limit of 1.0.")

        return weights