"""Download California Housing dataset and save it as a CSV file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from sklearn.datasets import fetch_california_housing


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "california_housing.csv"


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = cast(Any, fetch_california_housing(as_frame=True, return_X_y=False))
    frame = dataset.frame
    frame.to_csv(RAW_DATA_PATH, index=False)

    print(f"Saved dataset to {RAW_DATA_PATH}")
    print(f"Rows: {frame.shape[0]}")
    print(f"Columns: {frame.shape[1]}")


if __name__ == "__main__":
    main()
