#!/usr/bin/env bash

set -euo pipefail

# Default values
mini_number="1"
train_valid="train"

usage() {
  echo "Usage: bash run.sh [--mini-number=1|2] [--train-valid=train|validation]"
}

# Parse arguments
for arg in "$@"; do
  case "$arg" in
    --mini-number=*)
      mini_number="${arg#*=}"
      ;;
    --train-valid=*)
      train_valid="${arg#*=}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg"
      usage
      exit 1
      ;;
  esac
done

python tools/backtest.py \
  --data-path "data/mini${mini_number}/${train_valid}.parquet" \
  --signal-path "data/weights/mini${mini_number}_equal_weights_${train_valid}.parquet" \
  --plot-path "data/output/equity_curve_mini${mini_number}_${train_valid}.png" \
  --returns-path "data/output/portfolio_returns_mini${mini_number}_${train_valid}.csv"
