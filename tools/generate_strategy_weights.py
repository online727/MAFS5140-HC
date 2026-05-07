import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from strategy import Strategy
from utils import load_tabular_data


def generate_weights(data_path, output_path):
    data = load_tabular_data(data_path)

    if not isinstance(data.index, pd.DatetimeIndex):
        if "datetime" in data.columns:
            data["datetime"] = pd.to_datetime(data["datetime"])
            data = data.set_index("datetime")
        else:
            raise ValueError("Data must have DatetimeIndex or datetime column.")

    strategy = Strategy()

    weights_list = []

    for timestamp, row in data.iterrows():
        if isinstance(data.columns, pd.MultiIndex):
            current_market_data = row.unstack()
        else:
            raise ValueError("Expected MultiIndex columns like ticker-field.")

        weights = strategy.step(current_market_data)
        weights.name = timestamp
        weights_list.append(weights)

    weights_df = pd.DataFrame(weights_list)
    weights_df.index.name = "datetime"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    weights_df.to_parquet(output_path)
    print(f"Saved weights to {output_path}")
    print(weights_df.head())
    print(weights_df.tail())


if __name__ == "__main__":
    generate_weights(
        data_path="data/mini1/validation.parquet",
        output_path="tools/output/strategy_weights.parquet",
    )