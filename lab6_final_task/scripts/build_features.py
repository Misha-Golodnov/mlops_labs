"""Создать базовые train/test-наборы признаков."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "california_housing_clean.csv"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
TRAIN_PATH = FEATURES_DIR / "train.csv"
TEST_PATH = FEATURES_DIR / "test.csv"
TARGET_COLUMN = "medhouseval"


def main() -> None:
    if not PROCESSED_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Очищенный датасет не найден: {PROCESSED_DATA_PATH}. "
            "Сначала запустите scripts/preprocess_data.py."
        )

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(PROCESSED_DATA_PATH)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Целевая колонка отсутствует: {TARGET_COLUMN}")

    train, test = train_test_split(frame, test_size=0.2, random_state=42)

    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    print(f"Train-набор сохранен: {TRAIN_PATH}")
    print(f"Test-набор сохранен: {TEST_PATH}")
    print(f"Строк в train: {train.shape[0]}")
    print(f"Строк в test: {test.shape[0]}")


if __name__ == "__main__":
    main()
