## Changes -- 2026-04-14
* Revised `strategy,py`, here are three classes now:
  * `EqualWeightStrategy`
  * `MovingAverageStrategy`
  * `MeanVarianceStrategy`
  * You can modify the code in `main.py` to change the strategy you want to test.
* Vectorized backtesting engine
  * You need to generate the weights for backtesing - a dataframe with tiemstamps as index and stock tickers as columns
    * You can run `python zhh/generate_equal_weightspy` to generate the weights for equal weight strategy
  * Then you can run `bash tools/run.sh --mini-number=2 --train-valid=test` to generate the backtesting results
    * `mini-number`: id of our mini project
    * `train-valid`: the data you want to use, i.e. the name of the parquet file in `data/mini{mini-number}`
  * And then you can see the backtesting results in `data/output`
* For `tools/backtest.py`, you can run it with:
```bash
python tools/backtest.py \
  --data-path "/your/data/path" \
  --signal-path "/your/signal/path" \
  --plot-path "data/output/equity_curve.png" \
  --returns-path "data/output/portfolio_returns.csv"
```