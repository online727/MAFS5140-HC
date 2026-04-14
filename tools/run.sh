#!/usr/bin/env bash

# Default values
mini_number="2"
train_valid="train"

# for strategy_ver1.py
data_path="data/mini${mini_number}/${train_valid}.parquet"
signal_path="data/weights/mini${mini_number}_strategy_ver1_${train_valid}.parquet"
momentum_windows="6,12"
volume_window="78"
min_relative_volume="0.8"
volume_cap="2.0"
top_selector="20"
max_weight="0.05"

python tools/strategy_ver1.py \
  --data-path "$data_path" \
  --momentum-windows "$momentum_windows" \
  --volume-window "$volume_window" \
  --min-relative-volume "$min_relative_volume" \
  --volume-cap "$volume_cap" \
  --top-selector "$top_selector" \
  --max-weight "$max_weight" \
  --output-path "$signal_path"

plot_path="data/output/ver1/equity_curve_mini${mini_number}_${train_valid}.png"
returns_path="data/output/ver1/portfolio_returns_mini${mini_number}_${train_valid}.csv"

python tools/backtest.py \
  --data-path "$data_path" \
  --signal-path "$signal_path" \
  --plot-path "$plot_path" \
  --returns-path "$returns_path"

# # for equal weight strategy
# # if you have not generated the weights for equal weight strategy, you can run the following command to generate the weights
# # python zhh/generate_equal_weights.py
# # then you can run the backtest for equal weight strategy with the following command

# data_path="data/mini${mini_number}/${train_valid}.parquet"
# signal_path="data/weights/mini${mini_number}_equal_weights_${train_valid}.parquet"
# plot_path="data/output/equal_weight/equity_curve_mini${mini_number}_${train_valid}.png"
# returns_path="data/output/equal_weight/portfolio_returns_mini${mini_number}_${train_valid}.csv"

# python tools/backtest.py \
#   --data-path "$data_path" \
#   --signal-path "$signal_path" \
#   --plot-path "$plot_path" \
#   --returns-path "$returns_path"