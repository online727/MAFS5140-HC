## Changes -- 2026-04-14
* Revised `strategy,py`, here are three classes now:
  * `EqualWeightStrategy`
  * `MovingAverageStrategy`
  * `MeanVarianceStrategy`
    * you may need to install `qpsolvers` to run this strategy, you can install it with `pip install qpsolvers numpy scipy cvxopt osqp quadprog`
  * You can modify the code in `main.py` to change the strategy you want to test
* Vectorized backtesting engine
  * You need to generate the weights for backtesing - a dataframe with tiemstamps as index and stock tickers as columns
    * You can run `python zhh/generate_equal_weights.py` to generate the weights for equal weight strategy
  * Then you can run `tools/backtest.py` to generate the backtesting results
  ```bash
  python tools/backtest.py \
    --data-path "/your/data/path" \
    --signal-path "/your/signal/path" \
    --plot-path "data/output/equity_curve.png" \
    --returns-path "data/output/portfolio_returns.csv"
  ```
  * And then you can see the backtesting results in `data/output`
* An momentum based strategy `tools/strategy_ver1.py` with the following parameters:
  * `momentum_windows`: the lookback windows for momentum calculation, default is `6,12,24,78`
  * `volume_window`: the lookback window for volume calculation, default is `78`
  * `min_relative_volume`: the minimum relative volume to be considered for trading, default is `0.8`
  * `volume_cap`: the maximum volume cap to be considered for trading, default is `2.0`
  * `top_selector`: the top `N` or `q%` of stocks to be selected based on momentum, default is `0.1`
    * If `0 < top_selector < 1`, it will be treated as a percentage
    * If `top_selector > 1`, it will be treated as a number of stocks
    * If `top_selector == 1 or top_selector == asset_num`, it will select all stocks
  * `max_weight`: the maximum weight for each stock in the portfolio, default is `0.05`
  * You can run the strategy with the following command:
  ```bash
  python tools/strategy_ver1.py \
    --data-path "/your/data/path" \
    --output-path "/your/signal/path" \
    --momentum-windows "6,12,24,78" \
    --volume-window "78" \ 
    --min-relative-volume "0.8" \
    --volume-cap "2.0" \
    --top-selector "0.1" \
    --max-weight "0.05"
  ```
* A full process of generating signals and backtesting is provided in `tools/run.sh`, you can modify the parameters in this script to test different settings
  * use `strategy_ver1` or `equal_weight` strategy by modifying the command in `tools/run.sh`
  * use `bash tools/run.sh` to run the full process