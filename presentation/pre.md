Hello everyone, today I will present our trading strategy for this course project.

# 1. Strategy Overview

Our strategy combines two signals: a mean-variance allocation signal and a momentum signal.

The mean-variance part focuses on portfolio allocation. It estimates expected return and risk, then decides how to distribute capital across assets. The momentum part focuses on short-term cross-sectional opportunities. It uses recent price movement and volume information to rank assets. And the final strategy blends these two parts.

# 2. Mean-Variance Strategy Logic

The mean-variance component is a shrinkage-based Markowitz-style model. It estimates recent asset returns and their covariance matrix, then solves an optimization problem that balances expected return and portfolio variance.

The model can be written as this one:

```text
maximize    mu_shrunk^T w - (gamma / 2) * w^T Sigma_shrunk w

subject to  w_i <= position cap
```

The required inputs are:

- `mu_shrunk`: the shrinkage expected return vector.
- `Sigma_shrunk`: the shrinkage covariance matrix.
- `gamma`: the risk-aversion coefficient.
- `position cap`: the maximum weight allowed for a single asset.

The expected return estimate is shrunk toward zero. This makes the return forecast more conservative and reduces the effect of noisy short-term returns.

The covariance matrix is shrunk toward a diagonal matrix. This keeps part of the cross-asset covariance information, but reduces the impact of unstable correlation estimates.

After estimating these inputs, the optimizer produces a target weight vector. Assets with stronger risk-adjusted expected return can receive higher weights, capped by the single-asset max weight.

# 3. Momentum Strategy Logic

The momentum component is a cross-sectional ranking signal based on price movement and volume confirmation.

First, the strategy computes short-term price returns over multiple horizons. These returns are standardized across the asset universe using z-scores. This converts raw returns into relative strength scores, so the strategy compares each asset against other assets at the same timestamp.

Second, the strategy computes relative volume by comparing the latest volume with recent average volume. This is used as a confirmation signal. Price movement with stronger relative volume receives more support, while weak-volume movement is down-weighted.

The current feature list is showed below:

- Short-horizon price return.
- Multi-horizon momentum score.
- Cross-sectional momentum z-score.
- Relative volume.
- Volume-adjusted momentum score.
- Cross-sectional ranking score.

After combining the price and volume information, each asset receives a final score. The strategy selects the strongest candidates and assigns weights according to their scores, with a maximum weight limit for each asset.

# 4. Signal Generation and Portfolio Construction

Both components output target portfolio weights.

The mean-variance output is an optimized allocation vector: `w_mv`

The momentum output is a ranking-based allocation vector: `w_mom`

The final strategy combines them through a weighted average:

```text
w_final = alpha * w_mom + (1 - alpha) * w_mv
```

Here, `alpha` is a multiplier that controls the blend between the two strategies.

# 5. Time-Varying Blend

The multiplier `alpha` changes over time dynamically, so that to adjust the strategy's focus between momentum and mean-variance.

<!-- At the beginning, the mean-variance model has limited historical data, so the strategy gives more weight to the momentum signal. Momentum can react earlier because it depends on shorter-term price and volume features.

As the backtest moves forward, the mean-variance model has more data to estimate return and risk. The strategy then gradually shifts more weight toward mean-variance optimization.

So the combined model starts as a faster signal-driven strategy and gradually becomes more allocation-driven. Momentum provides asset selection and responsiveness. Mean-variance optimization provides risk-aware weighting and diversification. -->

# 6. Future Improvements

There are two main directions for improvement.

First, I can improve the momentum features. The current version mainly uses price movement and relative volume. Future versions can add features such as volatility, turnover, etc. I can also use machine learning models to combine these features and output asset scores.

Second, I can improve the way the two strategies are combined. The current method directly blends two complete weight vectors, which is simple but rough. A better structure would be to use the momentum model first to select a fixed number of candidate stocks, and then run mean-variance optimization only within this selected universe. In that design, momentum decides what to trade, and mean-variance optimization decides how much weight each selected asset should receive.

Ok, this is the overview of our trading strategy. Thank you for listening.