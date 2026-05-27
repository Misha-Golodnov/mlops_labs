"""Создать очищенную версию датасета California Housing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "california_housing.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "california_housing_clean.csv"


def main() -> None:
    if not RAW_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Исходный датасет не найден: {RAW_DATA_PATH}. "
            "Сначала запустите scripts/download_data.py."
        )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(RAW_DATA_PATH)
    frame = frame.drop_duplicates().reset_index(drop=True)
    frame.columns = [column.lower() for column in frame.columns]

    frame.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Очищенный датасет сохранен: {PROCESSED_DATA_PATH}")
    print(f"Количество строк: {frame.shape[0]}")
    print(f"Количество колонок: {frame.shape[1]}")


if __name__ == "__main__":
    main()
