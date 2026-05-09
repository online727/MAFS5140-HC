# Report 2: Multi-Feature Momentum Strategy

## 1. Introduction

This report describes the second version of my trading strategy. The main idea is to replace a single hand-picked signal with a diversified multi-feature framework. Instead of relying on one momentum definition, the strategy evaluates a large pool of price-volume features, groups strong candidates into several feature sets, and combines the best-performing sets into one final long-only portfolio.

The motivation is that intraday momentum signals are often unstable. A feature that works well in one period may lose effectiveness when market conditions change. A multi-feature design reduces this dependence on any single signal by combining several related but different views of short-term price movement, volume behavior, volatility, trend, and reversal. The strategy therefore focuses on two goals:

1. find features that have both backtest performance and predictive cross-sectional information; and
2. combine selected features in a simple, robust way that satisfies the project constraints of no short selling and no leverage.

The research workflow has four stages. First, I computed many feature variants from historical close and volume data. Second, each variant was evaluated independently using both PnL metrics and information coefficient metrics. Third, the best variants were grouped into several feature sets according to different selection criteria, such as Sharpe ratio, RankICIR, composite score, drawdown control, and family diversification. Finally, the submitted strategy uses the top four feature sets ranked by Sharpe ratio and combines their generated portfolio weights using performance-based feature-set weights.

The implementation is split into research scripts and the final event-driven strategy. Feature computation is implemented in `tools/mom_features.py`. Feature analysis and feature-set construction are implemented in `tools/feat_ana.py`. Offline feature-set portfolio generation is implemented in `tools/strategy_ver2.py` and `tools/strategy_ver2_submit.py`. The final submitted strategy is `strategy.py`, which reproduces the selected feature calculations in a streaming event-driven setting.

## 2. Methodology

### 2.1 Data and Trading Setting

The strategy uses the market data provided by the project framework. At each timestamp, the strategy receives a cross-section of assets with at least two fields: close price and volume. The strategy returns target portfolio weights indexed by ticker.

The project backtesting engine enforces three major portfolio constraints:

- all individual weights must be non-negative;
- the total portfolio weight must be no larger than 1.0; and
- unallocated weight is treated as cash with zero return.

Therefore, the strategy is designed as a long-only cross-sectional selection strategy. At each timestamp, it ranks assets by a combined score, selects the top assets with positive scores, and allocates capital among them. If there are no positive scores, the strategy holds cash.

### 2.2 Feature Universe

The feature pool is generated from close and volume panels. For each raw feature, two directions are considered:

- `raw`: use the feature in its original direction;
- `neg`: multiply the feature by -1, allowing the same formula to represent a reversal signal.

This is important because some short-horizon signals can work in the opposite direction from their economic name. For example, a recent return feature can represent either momentum or short-term reversal depending on whether `raw` or `neg` performs better.

In total, I define 132 raw feature formulas. Because every raw feature is evaluated in both `raw` and `neg` directions, the feature research step evaluates 264 feature variants. The feature universe contains five families:

| Family | Number of raw features | Feature types | Parameters used |
|---|---:|---|---|
| Return | 29 | `ret`, `logret`, `rolling_mean_ret`, `rolling_sum_ret`, `ret_zscore_ts` | `ret/logret`: windows 1, 3, 6, 12, 24, 39, 78; rolling and z-score features: windows 6, 12, 24, 39, 78 |
| Trend | 32 | `price_zscore_ts`, `sma_slope`, `ema_slope`, `sma_dist`, `ema_dist`, `ma_cross` | z-score and slope windows 6, 12, 24, 39, 78; distance windows 6, 12, 24, 39, 78, 156; MA-cross pairs (3,12), (6,24), (12,39), (12,78), (24,78) |
| Oscillator | 19 | `roc`, `rsi`, `stoch_close`, `macd` | `roc`: windows 1, 3, 6, 12, 24, 39, 78; `rsi`: windows 6, 12, 24, 39, 78; stochastic windows 12, 24, 39, 78; MACD triples (6,24,9), (12,26,9), (12,39,9) |
| Volume | 33 | `volume_change`, `relative_volume`, `volume_zscore`, `dollar_volume_zscore`, `price_volume_corr`, `obv_slope` | `volume_change`: windows 1, 3, 6, 12, 24, 39, 78; relative/z-score/dollar-volume z-score windows 6, 12, 24, 39, 78, 156; correlation and OBV slope windows 12, 24, 39, 78 |
| Volatility | 19 | `realized_vol`, `downside_vol`, `vol_adjusted_ret`, `return_range` | volatility and volatility-adjusted return windows 6, 12, 24, 39, 78; return-range windows 12, 24, 39, 78 |

The detailed factor-type table is:

| Family | Factor type | Parameters | Number of raw features |
|---|---|---|---:|
| Return | `ret_w` | $w \in \{1,3,6,12,24,39,78\}$ | 7 |
| Return | `logret_w` | $w \in \{1,3,6,12,24,39,78\}$ | 7 |
| Return | `rolling_mean_ret_w` | $w \in \{6,12,24,39,78\}$ | 5 |
| Return | `rolling_sum_ret_w` | $w \in \{6,12,24,39,78\}$ | 5 |
| Return | `ret_zscore_ts_w` | $w \in \{6,12,24,39,78\}$ | 5 |
| Trend | `price_zscore_ts_w` | $w \in \{6,12,24,39,78\}$ | 5 |
| Trend | `sma_slope_w` | $w \in \{6,12,24,39,78\}$ | 5 |
| Trend | `ema_slope_w` | $w \in \{6,12,24,39,78\}$ | 5 |
| Trend | `sma_dist_w` | $w \in \{6,12,24,39,78,156\}$ | 6 |
| Trend | `ema_dist_w` | $w \in \{6,12,24,39,78,156\}$ | 6 |
| Trend | `ma_cross_f_s` | $(f,s) \in \{(3,12),(6,24),(12,39),(12,78),(24,78)\}$ | 5 |
| Oscillator | `roc_w` | $w \in \{1,3,6,12,24,39,78\}$ | 7 |
| Oscillator | `rsi_w` | $w \in \{6,12,24,39,78\}$ | 5 |
| Oscillator | `stoch_close_w` | $w \in \{12,24,39,78\}$ | 4 |
| Oscillator | `macd_f_s_g` | $(f,s,g) \in \{(6,24,9),(12,26,9),(12,39,9)\}$ | 3 |
| Volume | `volume_change_w` | $w \in \{1,3,6,12,24,39,78\}$ | 7 |
| Volume | `relative_volume_w` | $w \in \{6,12,24,39,78,156\}$ | 6 |
| Volume | `volume_zscore_w` | $w \in \{6,12,24,39,78,156\}$ | 6 |
| Volume | `dollar_volume_zscore_w` | $w \in \{6,12,24,39,78,156\}$ | 6 |
| Volume | `price_volume_corr_w` | $w \in \{12,24,39,78\}$ | 4 |
| Volume | `obv_slope_w` | $w \in \{12,24,39,78\}$ | 4 |
| Volatility | `realized_vol_w` | $w \in \{6,12,24,39,78\}$ | 5 |
| Volatility | `downside_vol_w` | $w \in \{6,12,24,39,78\}$ | 5 |
| Volatility | `vol_adjusted_ret_w` | $w \in \{6,12,24,39,78\}$ | 5 |
| Volatility | `return_range_w` | $w \in \{12,24,39,78\}$ | 4 |

The detailed formulas for all feature types are listed in Appendix A.

### 2.3 Single-Feature Evaluation

Each feature variant is evaluated independently before being used in any combined strategy. The evaluation in `tools/mom_features.py` has two parts.

First, the feature is converted into a simple single-feature portfolio. For every timestamp, the feature values are cross-sectionally standardized:

$$
z_{i,t} = \frac{x_{i,t} - mean_t(x_t)}{std_t(x_t)}
$$

where `i` indexes assets and `t` indexes timestamps. For the `neg` direction, the z-score is multiplied by -1. The strategy then selects the highest-scoring assets and allocates only to selected assets with positive scores. Weights are proportional to positive scores, subject to a maximum single-asset weight cap.

Second, the feature is evaluated using both realized portfolio performance and predictive information metrics. The PnL metrics include cumulative return, annualized return, annualized volatility, Sharpe ratio, and maximum drawdown. The predictive metrics are computed using forward returns over 5-minute, 15-minute, 30-minute, and 60-minute horizons. For each horizon, the analysis computes:

- Pearson IC between feature values and future returns;
- ICIR, defined as mean IC divided by IC standard deviation;
- RankIC between feature ranks and future return ranks; and
- RankICIR, defined as mean RankIC divided by RankIC standard deviation.

This two-part evaluation avoids selecting features only because they happened to produce good portfolio PnL. A good candidate should ideally show both positive backtest performance and stable cross-sectional predictive power.

### 2.4 Feature Scoring and Feature-Set Construction

After all individual variants are evaluated, `tools/feat_ana.py` creates derived scores and feature sets. In the final feature selection run, I use the `validation` split of the `final` dataset. For each constructed feature set, the script selects 15 feature variants, so the parameter is `top_n = 15`.

The analysis uses the following metrics:

- `sharpe_ratio`: risk-adjusted single-feature backtest performance;
- `cumulative_return`: total return of the single-feature portfolio;
- `max_drawdown`: downside risk of the single-feature portfolio;
- `avg_icir`: average Pearson ICIR across the four forward-return horizons;
- `avg_rankicir`: average RankICIR across the four horizons;
- `positive_rankicir_horizons`: number of horizons with positive RankICIR;
- `risk_adjusted_score`: a percentile score combining Sharpe and drawdown; and
- `composite_score`: a weighted percentile score combining PnL, IC, RankIC, and drawdown.

For a feature variant $j$, the single-feature portfolio return at timestamp $t$ is $r_{j,t}$. The main PnL metrics are:

$$
CR_j = \prod_t (1 + r_{j,t}) - 1
$$

$$
AR_j = (1 + CR_j)^{P/T} - 1
$$

$$
AV_j = \sqrt{P} \cdot \sigma(r_{j,t})
$$

$$
Sharpe_j = \frac{AR_j}{AV_j}
$$

$$
MDD_j = \min_t \frac{Equity_{j,t} - \max_{\tau \le t} Equity_{j,\tau}}{\max_{\tau \le t} Equity_{j,\tau}}
$$

where $P$ is the annualization frequency and $T$ is the number of return observations.

For predictive metrics, let $x_{i,t}^{(j)}$ be the value of feature variant $j$ for asset $i$ at timestamp $t$, and let $R_{i,t+h}$ be the forward return over horizon $h$. The cross-sectional IC at time $t$ is:

$$
IC_{j,t,h} = Corr_i(x_{i,t}^{(j)}, R_{i,t+h})
$$

The rank IC replaces both variables with cross-sectional ranks:

$$
RankIC_{j,t,h} = Corr_i(rank(x_{i,t}^{(j)}), rank(R_{i,t+h}))
$$

For each horizon $h \in \{5min, 15min, 30min, 60min\}$:

$$
ICIR_{j,h} = \frac{mean_t(IC_{j,t,h})}{std_t(IC_{j,t,h})}
$$

$$
RankICIR_{j,h} = \frac{mean_t(RankIC_{j,t,h})}{std_t(RankIC_{j,t,h})}
$$

The multi-horizon averages are:

$$
AvgICIR_j = \frac{1}{4}\sum_h ICIR_{j,h}
$$

$$
AvgRankICIR_j = \frac{1}{4}\sum_h RankICIR_{j,h}
$$

The score construction uses percentile ranks. For a metric $m_j$, define $pct(m_j)$ as its percentile rank among all feature variants, with a higher value representing a better variant. Drawdown is scored using the percentile rank of `max_drawdown`, where values closer to zero are better. The risk-adjusted score is:

$$
RiskAdjustedScore_j = 0.70 \cdot pct(Sharpe_j) + 0.30 \cdot pct(MDD_j)
$$

The composite score uses the following weights:

| Component | Weight |
|---|---:|
| Sharpe percentile score | 0.35 |
| Cumulative return percentile score | 0.20 |
| Average RankICIR percentile score | 0.25 |
| Average ICIR percentile score | 0.10 |
| Drawdown percentile score | 0.10 |

Equivalently:

$$
\begin{align*}
    \boldsymbol{w} &= (0.35, 0.20, 0.25, 0.10, 0.10)^\top \\
    \boldsymbol{m}_j &= (pct(Sharpe_j), pct(CR_j), pct(AvgRankICIR_j), pct(AvgICIR_j), pct(MDD_j))^\top \\
    \text{CompositeScore}_j &= \boldsymbol{w}^\top \boldsymbol{m}_j
\end{align*}
$$

Based on these metrics, the analysis constructs several candidate feature sets:

| Feature set | Selection idea |
|---|---|
| `best_by_sharpe` | variants with the highest single-feature Sharpe ratios |
| `best_by_rankicir` | variants with the strongest average RankICIR |
| `best_by_icir` | variants with the strongest average Pearson ICIR |
| `robust_multi_horizon` | variants with positive short-horizon and multi-horizon RankICIR |
| `low_drawdown` | variants with strong risk-adjusted scores using Sharpe and drawdown |
| `family_balanced` | high composite-score variants while keeping feature-family diversification |
| `composite_best` | variants with the highest overall composite scores |
| `diversified_best` | composite-ranked variants after duplicate and correlation pruning |

The redundancy control step is used to avoid selecting many near-identical variants. It compares both feature-value correlation and generated-weight correlation among top candidates. Highly correlated candidates are pruned from the diversified set.

### 2.5 Final Feature Sets and Set Weights

The final selected feature definitions are saved in `feature_sets.json`. Each entry contains the feature name and the selected direction. The feature definitions are chosen from the final validation-set feature analysis, where each feature set contains 15 selected feature variants.

The final feature-set performance table is saved in `feature_sets_weights.csv`. The weights are constructed using both a train part and a validation part, rather than relying only on one period. For each feature set $k$ and performance metric $m$, I compute:

- $m_{k,train}$ from the final train data part, from 2025-01-01 to `train_end_date`;
- $m_{k,valid}$ from the final validation data part; and
- a combined metric using a 70/30 weighted average:

$$
m_{k,final} = 0.70 \cdot m_{k,train} + 0.30 \cdot m_{k,valid}
$$

The final table stores these combined feature-set metrics. The submitted strategy then sorts feature sets by the combined `sharpe_ratio` metric and selects the top four feature sets.

The final strategy uses the top four feature sets sorted by the `sharpe_ratio` column in `feature_sets_weights.csv`:

| Rank | Feature set | Cumulative return | Annualized return | Annualized volatility | Sharpe ratio | Max drawdown |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `low_drawdown` | 0.1308 | 0.1629 | 0.1108 | 0.1955 | 0.0923 |
| 2 | `best_by_sharpe` | 0.1372 | 0.1345 | 0.1175 | 0.1384 | 0.1172 |
| 3 | `family_balanced` | 0.1299 | 0.1310 | 0.1243 | 0.1270 | 0.1310 |
| 4 | `composite_best` | 0.1348 | 0.1261 | 0.1276 | 0.1210 | 0.1164 |

These four sets are combined using their Sharpe ratios as raw set weights. After normalization, the approximate weights are:

| Feature set | Normalized set weight |
|---|---:|
| `low_drawdown` | 0.336 |
| `best_by_sharpe` | 0.238 |
| `family_balanced` | 0.218 |
| `composite_best` | 0.208 |

This weighting scheme gives more influence to feature sets that performed better on a risk-adjusted basis, while still preserving diversification across several selection philosophies.

### 2.6 Portfolio Construction

For each selected feature set, the strategy computes a feature-set score as follows:

1. compute each required feature from the streaming close and volume history;
2. apply the selected direction, either `raw` or `neg`;
3. cross-sectionally z-score each feature at the current timestamp;
4. average the z-scored features equally within the feature set; and
5. convert the resulting feature-set score into a long-only top-asset portfolio.

The top-asset conversion works as follows. The strategy selects the highest-scoring assets according to `top_selector`. In the submitted implementation, `top_selector = 10`, so at most ten assets are selected at each timestamp. Among the selected assets, only positive scores receive capital. Raw weights are proportional to the positive scores:

$$
w_{i,t} = \frac{s_{i,t}}{\sum_j s_{j,t}}, \quad s_{i,t} > 0
$$

The strategy then caps each single-asset weight at `max_weight = 0.10`. This cap improves diversification and ensures no asset dominates the portfolio. If the sum of weights is less than 1 after capping or because only a few assets have positive scores, the remaining capital stays in cash.

The final combined portfolio is the weighted average of the four feature-set portfolios:

$$
W_t = \sum_k \alpha_k W_{k,t}
$$

where `alpha_k` is the normalized feature-set weight and `W_k,t` is the portfolio generated by feature set `k`. The combined portfolio is cleaned again to enforce non-negative weights, the single-asset weight cap, and the no-leverage constraint.

### 2.7 Event-Driven Implementation

The final submitted file is `strategy.py`. Unlike the offline research scripts, this file cannot access the full historical panel at once. It must process one market snapshot at a time. Therefore, it implements a `StreamingMomentumFeatureEngine` that stores rolling histories of close, volume, returns, volume changes, dollar volume, OBV proxy values, and EMA states.

The streaming implementation mirrors the offline feature definitions as closely as possible. Rolling statistics such as moving averages, volatility, z-scores, and price-volume correlations are computed from the stored history. Recursive indicators such as EMA and MACD are updated incrementally. The maximum stored history is 240 bars, which is sufficient for the selected features because the largest lookback used in the final sets is within this range.

At each call to `step(current_market_data)`, the strategy:

1. updates the streaming feature engine with the new close and volume data;
2. updates recursive EMA and MACD indicators needed by selected features;
3. computes all required feature values;
4. builds one score and one portfolio for each selected feature set;
5. combines feature-set portfolios using normalized Sharpe-based set weights; and
6. returns a clean long-only target-weight series aligned with the input ticker index.

This design keeps the final submitted strategy compatible with the project framework while preserving the same research logic used in the offline analysis.

## 3. Results

This section should contain the final empirical results from the backtest. I will fill in the exact tables, figures, and discussion after running the final evaluation.

### 3.1 Overall Performance

Fill in the final performance table from `main.py` or the evaluator output.

| Metric | Value |
|---|---:|
| Cumulative Return | TODO |
| Annualized Return | TODO |
| Annualized Volatility | TODO |
| Sharpe Ratio | TODO |
| Max Drawdown | TODO |

Suggested discussion points:

- compare the final strategy's Sharpe ratio with the previous strategy version;
- discuss whether return improvement came with higher or lower drawdown;
- comment on whether the strategy uses cash often or remains mostly invested;
- explain whether the final risk profile is acceptable under the no-short and no-leverage constraints.

### 3.2 Equity Curve and Drawdown

Insert the following figures:

- final strategy cumulative wealth or equity curve;
- final strategy drawdown curve;
- optional comparison against Strategy Version 1 or a simple equal-weight benchmark.

Suggested interpretation:

- identify periods where the strategy gained most of its return;
- identify drawdown periods and whether they correspond to market reversals;
- discuss whether the equity curve is smooth or concentrated in a small number of intervals.

### 3.3 Feature-Set Contribution

Insert a table comparing the four selected feature sets:

| Feature set | Weight in final strategy | Standalone cumulative return | Standalone Sharpe | Standalone max drawdown | Comment |
|---|---:|---:|---:|---:|---|
| `low_drawdown` | 0.336 | TODO | TODO | TODO | TODO |
| `best_by_sharpe` | 0.238 | TODO | TODO | TODO | TODO |
| `family_balanced` | 0.218 | TODO | TODO | TODO | TODO |
| `composite_best` | 0.208 | TODO | TODO | TODO | TODO |

Suggested discussion points:

- `low_drawdown` receives the largest weight because it has the highest Sharpe ratio in the final feature-set table;
- `best_by_sharpe` adds direct risk-adjusted return strength;
- `family_balanced` reduces dependence on a single feature family;
- `composite_best` adds features selected by a broader score combining return, IC, RankIC, and drawdown.

### 3.4 Feature Analysis Summary

Insert feature analysis figures generated by `tools/feat_ana.py`, such as:

- Sharpe vs average RankICIR scatter plot;
- return vs drawdown scatter plot;
- feature-family Sharpe boxplot;
- feature-family RankICIR boxplot;
- top-candidate RankICIR heatmap by horizon;
- composite-score component bar chart.

Suggested discussion points:

- which feature families performed best overall;
- whether high-Sharpe features also had positive RankICIR;
- whether the selected features are concentrated in one family or diversified;
- whether short-term reversal features, trend features, volume features, or volatility features were most useful.

### 3.5 Robustness Checks

Fill in any additional validation results, if available.

Recommended checks:

- train vs validation performance comparison;
- performance by data split or period;
- sensitivity to `top_selector`, for example top 5, top 10, and top 20 assets;
- sensitivity to `max_weight`, for example 5% and 10%;
- comparison between using all selected feature sets and using only the top feature set.

Suggested table:

| Experiment | Cumulative return | Sharpe ratio | Max drawdown | Comment |
|---|---:|---:|---:|---|
| Final configuration | TODO | TODO | TODO | TODO |
| Top 5 assets | TODO | TODO | TODO | TODO |
| Top 20 assets | TODO | TODO | TODO | TODO |
| 5% max weight | TODO | TODO | TODO | TODO |
| Single best feature set only | TODO | TODO | TODO | TODO |

## 4. Conclusion

The second version of the strategy uses a systematic multi-feature framework instead of a single momentum signal. The research process begins with a broad feature universe, evaluates each feature in both raw and reversed directions, ranks variants using both PnL and IC-based metrics, and constructs several feature sets with different selection objectives. The final submitted strategy combines the top four Sharpe-ranked feature sets using normalized Sharpe-based weights.

The main advantage of this approach is diversification. The final portfolio is not driven by one indicator or one feature family. It combines short-horizon reversal, volume behavior, trend, and volatility-related information. The use of cross-sectional z-scores also makes features comparable across assets and across feature definitions. The long-only top-score allocation is simple and compatible with the project constraints.

There are still several limitations. First, the feature selection process may overfit the available validation period, especially because many variants and directions are tested. Second, transaction costs are not explicitly optimized in the current objective, so high-turnover features may look better in backtests than they would after costs. Third, the final set weights are based on historical performance metrics and may not be stable in future regimes.

Future improvements could include adding turnover penalties, using walk-forward validation, shrinking feature-set weights toward equal weights, adding explicit correlation control at the final portfolio level, and testing whether some feature families should be dynamically weighted according to recent market conditions.

## Appendix A. Feature Formulas

Let $C_{i,t}$ be the close price of asset $i$ at timestamp $t$, and let $V_{i,t}$ be its volume. Define one-period return as:

$$
r_{i,t} = \frac{C_{i,t}}{C_{i,t-1}} - 1
$$

For a rolling window $w$, define:

$$
SMA_w(C_{i,t}) = \frac{1}{w}\sum_{l=0}^{w-1} C_{i,t-l}
$$

$$
TSZ_w(X_{i,t}) = \frac{X_{i,t} - mean(X_{i,t-w+1:t})}{std(X_{i,t-w+1:t})}
$$

The EMA with span $w$ is computed recursively with $\alpha = 2/(w+1)$:

$$
EMA_w(C_{i,t}) = \alpha C_{i,t} + (1-\alpha)EMA_w(C_{i,t-1})
$$

The raw feature formulas are:

| Factor type | Formula |
|---|---|
| `ret_w` | $\frac{C_{i,t}}{C_{i,t-w}} - 1$ |
| `logret_w` | $\log(C_{i,t}) - \log(C_{i,t-w})$ |
| `rolling_mean_ret_w` | $\frac{1}{w}\sum_{l=0}^{w-1} r_{i,t-l}$ |
| `rolling_sum_ret_w` | $\sum_{l=0}^{w-1} r_{i,t-l}$ |
| `ret_zscore_ts_w` | $TSZ_w(r_{i,t})$ |
| `price_zscore_ts_w` | $TSZ_w(C_{i,t})$ |
| `sma_dist_w` | $\frac{C_{i,t}}{SMA_w(C_{i,t})} - 1$ |
| `ema_dist_w` | $\frac{C_{i,t}}{EMA_w(C_{i,t})} - 1$ |
| `sma_slope_w` | $\frac{SMA_w(C_{i,t})}{SMA_w(C_{i,t-w})} - 1$ |
| `ema_slope_w` | $\frac{EMA_w(C_{i,t})}{EMA_w(C_{i,t-w})} - 1$ |
| `ma_cross_f_s` | $\frac{SMA_f(C_{i,t})}{SMA_s(C_{i,t})} - 1$ |
| `roc_w` | $\frac{C_{i,t}-C_{i,t-w}}{C_{i,t-w}}$ |
| `rsi_w` | $\frac{RSI_w(C_{i,t}) - 50}{50}$, where $RSI_w = 100 - \frac{100}{1 + AvgGain_w/AvgLoss_w}$ |
| `stoch_close_w` | $\frac{C_{i,t} - \min(C_{i,t-w+1:t})}{\max(C_{i,t-w+1:t})-\min(C_{i,t-w+1:t})} - 0.5$ |
| `macd_f_s_g` | $\frac{(EMA_f(C_{i,t}) - EMA_s(C_{i,t})) - EMA_g(EMA_f(C_{i,t}) - EMA_s(C_{i,t}))}{C_{i,t}}$ |
| `volume_change_w` | $\frac{V_{i,t}}{V_{i,t-w}} - 1$ |
| `relative_volume_w` | $\frac{V_{i,t}}{SMA_w(V_{i,t})}$ |
| `volume_zscore_w` | $TSZ_w(V_{i,t})$ |
| `dollar_volume_zscore_w` | $TSZ_w(C_{i,t}V_{i,t})$ |
| `price_volume_corr_w` | $Corr(r_{i,t-w+1:t}, \Delta V_{i,t-w+1:t}/V_{i,t-w:t-1})$ |
| `obv_slope_w` | $\frac{OBV_{i,t} - OBV_{i,t-w}}{\sum_{l=0}^{w-1} V_{i,t-l}}$, where $OBV_{i,t}=OBV_{i,t-1}+sign(C_{i,t}-C_{i,t-1})V_{i,t}$ |
| `realized_vol_w` | $std(r_{i,t-w+1:t})$ |
| `downside_vol_w` | $std(\min(r_{i,t-w+1:t},0))$ |
| `return_range_w` | $\frac{\max(C_{i,t-w+1:t})}{\min(C_{i,t-w+1:t})} - 1$ |
| `vol_adjusted_ret_w` | $\frac{C_{i,t}/C_{i,t-w}-1}{std(r_{i,t-w+1:t})}$ |

For each raw feature $x_{i,t}$, two directional variants are evaluated:

$$
x^{raw}_{i,t} = x_{i,t}
$$

$$
x^{neg}_{i,t} = -x_{i,t}
$$
