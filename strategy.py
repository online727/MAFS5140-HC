import pandas as pd
import numpy as np
import scipy.sparse as sp
from qpsolvers import solve_qp
from collections import deque

import pandas as pd

"""
STUDENT INSTRUCTIONS:
1. This Strategy class is where you will implement your own trading strategy.
2. The current implementation is just a SIMPLE EXAMPLE (Moving Average Trend Following) 
   provided for your reference. Please modify this class to build your own strategy.
3. You may create new Python scripts and import them into this file if you 
   want to organize your code. 
4. IMPORTANT: Do NOT modify any other existing scripts in the backtest 
   framework. Changing core engine files may break the backtester and cause 
   evaluation errors.
"""

class MovingAverageStrategy:
    def __init__(self):
        """
        Initialize any state variables here.
        This function is called exactly once at the very beginning of the backtest.
        
        GUIDANCE:
        - You can create state variables using 'self.' to store data across steps.
        - For example, you might want to store historical market data, indicators, 
          or previous portfolio allocations.
        - PERFORMANCE WARNING: Storing too much historical data in memory (e.g., 
          growing a list infinitely) can significantly slow down the backtest or 
          cause memory crashes. Always try to keep only the data you need. 
        """
        # EXAMPLE STATE VARIABLES (Modify or remove these for your strategy):
        # We will use a list to store the historical price Series
        self.price_history = []
        self.lookback_period = 78

    def step(self, current_market_data: pd.DataFrame) -> pd.Series:
        """
        Core strategy logic. 
        This function is called at every timestamp by the BacktestEngine.
        
        INPUT:
        current_market_data (pd.DataFrame): Market snapshot at the current timestamp.
                                            Index = Tickers, Columns = fields
                                            ('close', 'volume').
                                    
        OUTPUT:
        pd.Series: Target weights for the portfolio.
                   Index = Tickers, Values = Weights (0.0 to 1.0).
                   The sum of weights must be <= 1.0.
                   
        GUIDANCE:
        - The code below is just a reference/example implementation. 
        - Please completely modify this function to reflect your own trading logic.
        """
        
        # --- START OF EXAMPLE STRATEGY LOGIC ---
        
        if "close" not in current_market_data.columns:
            raise ValueError("Input market data must contain a 'close' column.")

        current_prices = current_market_data["close"]

        # 1. Update internal state with the new data
        self.price_history.append(current_prices)
        
        # Keep only the required lookback period to save memory (Best Practice!)
        if len(self.price_history) > self.lookback_period:
            self.price_history.pop(0)
            
        # 2. Strategy Logic
        # If we don't have enough data yet, stay 100% in cash (return all zeros)
        if len(self.price_history) < self.lookback_period:
            return pd.Series(0.0, index=current_prices.index)
            
        # Convert our history list into a DataFrame to easily calculate the mean
        history_df = pd.DataFrame(self.price_history)
        moving_average = history_df.mean()
        
        # Identify assets where the current price is ABOVE its moving average (Trend Following)
        bullish_assets = current_prices[current_prices > moving_average].index
        
        # 3. Portfolio Allocation
        # Initialize all weights to 0.0
        weights = pd.Series(0.0, index=current_prices.index)
        
        # Allocate equally among bullish assets
        num_bullish = len(bullish_assets)
        if num_bullish > 0:
            weight_per_asset = 1.0 / num_bullish
            weights[bullish_assets] = weight_per_asset
            
        # Return the weights. 
        # The engine will verify that weights >= 0 and weights.sum() <= 1.0
        return weights
        
        # --- END OF EXAMPLE STRATEGY LOGIC ---

class EqualWeightStrategy:
    def __init__(self):
        """
        Initialize any state variables here.
        This function is called exactly once at the very beginning of the backtest.
        
        GUIDANCE:
        - You can create state variables using 'self.' to store data across steps.
        - For example, you might want to store historical market data, indicators, 
          or previous portfolio allocations.
        - PERFORMANCE WARNING: Storing too much historical data in memory (e.g., 
          growing a list infinitely) can significantly slow down the backtest or 
          cause memory crashes. Always try to keep only the data you need. 
        """
        self.price_history = []
        self.lookback_period = 78

    def step(self, current_market_data: pd.DataFrame) -> pd.Series:
        """
        Core strategy logic. 
        This function is called at every timestamp by the BacktestEngine.
        
        INPUT:
        current_market_data (pd.DataFrame): Market snapshot at the current timestamp.
                                            Index = Tickers, Columns = fields
                                            ('close', 'volume').
                                    
        OUTPUT:
        pd.Series: Target weights for the portfolio.
                   Index = Tickers, Values = Weights (0.0 to 1.0).
                   The sum of weights must be <= 1.0.
                   
        GUIDANCE:
        - The code below is just a reference/example implementation. 
        - Please completely modify this function to reflect your own trading logic.
        """
        weights_per_asset = 1 / current_market_data.shape[0]
        weights = pd.Series(weights_per_asset, index=current_market_data.index)
        return weights

class MeanVarianceStrategy:
    def __init__(self):
        """
        Shrinkage Mean-Variance Optimization Strategy

        Objective:
            maximize    mu_shrunk^T w - (gamma / 2) * w^T Sigma_shrunk w

        Constraints:
            w_i >= 0
            sum(w) <= 1

        Notes:
        - long-only
        - no leverage
        - cash is allowed implicitly when sum(w) < 1
        """
        # ===== Tunable parameters =====
        # Number of return observations used for estimation
        self.lookback_period = 78 * 5   # 5 trading days if 78 bars/day
        self.min_period = 78 * 5

        self.price_history = deque(maxlen=self.lookback_period + 1)

        # Risk aversion coefficient
        self.gamma = 10.0

        # Mean shrinkage intensity: in [0, 1]
        # 0   -> pure sample mean
        # 1   -> fully shrink mean to 0
        self.mean_shrinkage = 0.2

        # Covariance shrinkage intensity: in [0, 1]
        # 0   -> pure sample covariance
        # 1   -> fully shrink to diagonal covariance
        self.cov_shrinkage = 0.2

        # Upper bound per asset
        self.max_weight_per_asset = 0.10

        # Tiny ridge term for numerical stability
        self.ridge = 1e-6

    def _estimate_inputs(self, history_df: pd.DataFrame):
        """
        Estimate shrunk expected returns and shrunk covariance matrix.
        """
        returns = history_df.pct_change().dropna()

        if returns.empty:
            return None, None

        # Sample estimates
        mu_hat = returns.mean()
        sigma_hat = returns.cov()

        # Clean numerical issues
        mu_hat = mu_hat.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        sigma_hat = sigma_hat.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        n = len(mu_hat)
        if n == 0:
            return None, None

        # ===== Mean shrinkage =====
        # Shrink sample mean toward zero
        mu_target = pd.Series(0.0, index=mu_hat.index)
        mu_shrunk = (1.0 - self.mean_shrinkage) * mu_hat + self.mean_shrinkage * mu_target

        # ===== Covariance shrinkage =====
        # Shrink sample covariance toward diagonal covariance
        sigma_target = pd.DataFrame(
            np.diag(np.diag(sigma_hat.values)),
            index=sigma_hat.index,
            columns=sigma_hat.columns
        )
        sigma_shrunk = (1.0 - self.cov_shrinkage) * sigma_hat + self.cov_shrinkage * sigma_target

        # Add ridge for numerical stability
        sigma_shrunk = sigma_shrunk + self.ridge * np.eye(n)

        return mu_shrunk.values, sigma_shrunk.values

    def _solve_mean_variance(self, mu: np.ndarray, sigma: np.ndarray, tickers) -> pd.Series:
        """
        Solve constrained mean-variance optimization:
            maximize    mu^T w - (gamma / 2) * w^T Sigma w
            subject to  sum(w) <= 1
                        w_i >= 0
                        w_i <= max_weight_per_asset
        """
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        n = len(mu)

        P = sp.csc_matrix(self.gamma * sigma)
        q = -mu

        # G w <= h
        G = sp.vstack([
            -sp.eye(n, format="csc"),                      # w >= 0  -> -w <= 0
            sp.eye(n, format="csc"),                       # w <= max_weight
            sp.csc_matrix(np.ones((1, n)))                # sum(w) <= 1
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

    def step(self, current_market_data: pd.DataFrame) -> pd.Series:
        """
        Core strategy logic. 
        This function is called at every timestamp by the BacktestEngine.
        
        INPUT:
        current_market_data (pd.DataFrame): Market snapshot at the current timestamp.
                                            Index = Tickers, Columns = fields
                                            ('close', 'volume').
                                    
        OUTPUT:
        pd.Series: Target weights for the portfolio.
                   Index = Tickers, Values = Weights (0.0 to 1.0).
                   The sum of weights must be <= 1.0.
        """
        if "close" not in current_market_data.columns:
            raise ValueError("Input market data must contain a 'close' column.")

        current_prices = current_market_data["close"].astype(float)

        # Store current prices
        self.price_history.append(current_prices)

        # Need enough data to compute returns
        if len(self.price_history) < self.min_period + 1:
            return pd.Series(0.0, index=current_prices.index)

        # Build price history DataFrame
        history_df = pd.DataFrame(self.price_history)

        # Estimate shrunk mu and Sigma
        mu, sigma = self._estimate_inputs(history_df)
        if mu is None or sigma is None:
            return pd.Series(0.0, index=current_prices.index)

        # Solve optimization
        weights = self._solve_mean_variance(mu, sigma, current_prices.index)

        # Strict alignment with engine requirements
        weights = weights.reindex(current_prices.index).fillna(0.0)
        weights = weights.clip(lower=0.0, upper=self.max_weight_per_asset)

        if weights.sum() > 1.0:
            weights = weights / weights.sum()

        return weights

class MomentumStrategy:
    def __init__(self):
        self.momentum_windows = (6, 12)
        self.volume_window = 78
        self.min_relative_volume = 0.8
        self.volume_cap = 2.0
        self.top_selector = 10
        self.max_weight = 0.1

        self.max_price_history = max(self.momentum_windows) + 1
        self.price_history = deque(maxlen=self.max_price_history)
        self.volume_history = deque(maxlen=self.volume_window)
        self.tickers = None

        print(f"MomentumStrategy trade top {self._selection_count(438)}/{438} assets based on momentum and volume signals.")

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

class Strategy:
    def __init__(self):
        self.mom_strategy = MomentumStrategy()
        self.mean_variance_strategy = MeanVarianceStrategy()
        self.counter = 0
        self.initial_mul_mom = 1
        self.final_mul_mom = 0.2

    def step(self, current_market_data: pd.DataFrame) -> pd.Series:
        self.counter += 1
        weights_mom = self.mom_strategy.step(current_market_data)
        weights_mv = self.mean_variance_strategy.step(current_market_data)
        mul_mom = self.initial_mul_mom - (self.initial_mul_mom - self.final_mul_mom) * min(1.0, self.counter / 78*5)  # Linearly decay over 10 trading days
        return mul_mom * weights_mom + (1 - mul_mom) * weights_mv