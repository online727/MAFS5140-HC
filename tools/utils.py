import pandas as pd
from pathlib import Path

def load_tabular_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_parquet(path, engine="fastparquet")

    if suffix == ".csv":
        return pd.read_csv(path, index_col=0, parse_dates=True)

    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)

    raise ValueError(f"Unsupported file format: {path}")