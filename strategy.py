import pandas as pd
import numpy as np
from scipy.optimize import minimize

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

class Strategy:
    def __init__(self):
        """
        Standard Mean-Variance Optimization Strategy
        Objective:
            maximize    mu^T w - (gamma / 2) * w^T Sigma w

        Constraints:
            w_i >= 0
            sum(w) <= 1

        Notes:
        - long-only
        - no leverage
        - cash is allowed implicitly when sum(w) < 1
        """
        self.price_history = []

        # Rolling window length for estimating mu and Sigma
        self.lookback_period = 78

        # Risk aversion coefficient gamma
        self.gamma = 10.0

        # Numerical stabilization for covariance matrix
        self.ridge = 1e-5

        # Optional upper bound per asset to avoid extreme concentration
        # Set to 1.0 if you do not want a cap
        self.max_weight_per_asset = 1.0

    def _estimate_inputs(self, history_df: pd.DataFrame):
        """
        Estimate expected returns and covariance matrix from price history.
        """
        returns = history_df.pct_change().dropna()

        if returns.empty:
            return None, None

        mu = returns.mean().values
        sigma = returns.cov().values

        # Ridge regularization for numerical stability
        n = sigma.shape[0]
        sigma = sigma + self.ridge * np.eye(n)

        return mu, sigma

    def _solve_mean_variance(self, mu: np.ndarray, sigma: np.ndarray, tickers) -> pd.Series:
        """
        Solve the constrained mean-variance optimization:
            maximize    mu^T w - (gamma / 2) * w^T Sigma w
            subject to  sum(w) <= 1
                        w_i >= 0
        """
        n = len(mu)

        # Objective for scipy minimize: convert maximization to minimization
        def objective(w):
            portfolio_return = np.dot(mu, w)
            portfolio_variance = np.dot(w, sigma @ w)
            utility = portfolio_return - 0.5 * self.gamma * portfolio_variance
            return -utility

        # Sum of weights <= 1
        constraints = [
            {"type": "ineq", "fun": lambda w: 1.0 - np.sum(w)}
        ]

        # Long-only bounds
        bounds = [(0.0, self.max_weight_per_asset) for _ in range(n)]

        # Initial guess: equally weighted but scaled to satisfy sum <= 1
        x0 = np.ones(n) / n
        if x0.sum() > 1.0:
            x0 = x0 / x0.sum()

        result = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-9, "disp": False}
        )

        if not result.success:
            # Fallback: stay in cash
            return pd.Series(0.0, index=tickers)

        weights = pd.Series(result.x, index=tickers)

        # Final cleanup for tiny numerical issues
        weights = weights.clip(lower=0.0)
        if weights.sum() > 1.0:
            weights = weights / weights.sum()

        return weights

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

        # Save current prices
        self.price_history.append(current_prices)

        # Keep only needed history to control memory usage
        if len(self.price_history) > self.lookback_period + 1:
            self.price_history.pop(0)

        # Need enough price points to compute returns
        if len(self.price_history) < self.lookback_period + 1:
            return pd.Series(0.0, index=current_prices.index)

        # Build historical price DataFrame
        history_df = pd.DataFrame(self.price_history)

        # Estimate mu and Sigma
        mu, sigma = self._estimate_inputs(history_df)

        if mu is None or sigma is None:
            return pd.Series(0.0, index=current_prices.index)

        # Solve mean-variance optimization
        weights = self._solve_mean_variance(mu, sigma, current_prices.index)

        # Final strict alignment with engine requirements
        weights = weights.reindex(current_prices.index).fillna(0.0)
        weights = weights.clip(lower=0.0)

        if weights.sum() > 1.0:
            weights = weights / weights.sum()

        return weights