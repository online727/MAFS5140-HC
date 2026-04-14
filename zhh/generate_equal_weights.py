import pandas as pd
import numpy as np
from pathlib import Path

BASE_PATH = Path(__file__)
while "MAFS5140-HC" != BASE_PATH.name:
    BASE_PATH = BASE_PATH.parent

def generate_equal_weights(data_path: str | Path, output_path: str | Path):
    data = pd.read_parquet(data_path)
    close_prices = data.xs("close", axis=1, level=1)
    equal_weights = pd.DataFrame(
        np.full_like(close_prices, 1 / close_prices.shape[1]), 
        index=close_prices.index, 
        columns=close_prices.columns
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    equal_weights.to_parquet(output_path)

def main(mini_number: str, train_name: str, valid_name: str):
    data_path = BASE_PATH / "data" / f"mini{mini_number}" / f"{train_name}.parquet"
    output_path = BASE_PATH / "data" / "weights" / f"mini{mini_number}_equal_weights_{train_name}.parquet"
    generate_equal_weights(data_path, output_path)
    print(f"Generated equal weights for mini{mini_number} {train_name} and saved to {output_path}")

    data_path = BASE_PATH / "data" / f"mini{mini_number}" / f"{valid_name}.parquet"
    output_path = BASE_PATH / "data" / "weights" / f"mini{mini_number}_equal_weights_{valid_name}.parquet"
    generate_equal_weights(data_path, output_path)
    print(f"Generated equal weights for mini{mini_number} {valid_name} and saved to {output_path}")

if __name__ == "__main__":
    config = {
        "1": {"train": "train", "valid": "validation"},
        "2": {"train": "train", "valid": "test"}
    }
    for mini_number, names in config.items():
        main(mini_number, names["train"], names["valid"])