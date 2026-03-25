"""
Этап конвейера: загрузка датасета Wine Quality (красное вино, UCI),
разделение признаков и целевой переменной, формирование train/test.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from sklearn.model_selection import train_test_split

WINE_QUALITY_RED_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
TARGET_COLUMN = "quality"
DEFAULT_TEST_SIZE = 0.2


def lab2_root() -> Path:
    return Path(__file__).resolve().parents[1]


def download_raw(csv_path: Path, force: bool) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and not force:
        return
    urlretrieve(WINE_QUALITY_RED_URL, csv_path)


def load_frame(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, sep=";")


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"В датасете нет колонки «{TARGET_COLUMN}»")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Wine Quality: загрузка и разбиение данных")
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Доля тестовой выборки (по умолчанию 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed для воспроизводимости split",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Перезаписать сырой CSV с UCI",
    )
    args = parser.parse_args()

    root = lab2_root()
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    raw_csv = raw_dir / "winequality-red.csv"

    download_raw(raw_csv, force=args.force_download)
    df = load_frame(raw_csv)
    X, y = split_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)
    y_train.to_frame(TARGET_COLUMN).to_csv(
        processed_dir / "y_train.csv", index=False
    )
    y_test.to_frame(TARGET_COLUMN).to_csv(processed_dir / "y_test.csv", index=False)

    print(f"Сырой файл: {raw_csv}")
    print(f"Строк: {len(df)}, признаков: {X.shape[1]}, цель: {TARGET_COLUMN}")
    print(f"Train: {len(X_train)}, test: {len(X_test)}")
    print(f"Артефакты: {processed_dir}")


if __name__ == "__main__":
    main()
