import pandas as pd
import numpy as np
import scipy.sparse as sp
from qpsolvers import solve_qp
from collections import deque


class Strategy:
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
