"""
Этап конвейера: базовая предобработка датасета insurance и
выделение важных признаков и разделение на train/test выборки.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "charges"
REQUIRED_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
CATEGORICAL_COLUMNS = ["sex", "smoker", "region"]
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_FEATURE_IMPORTANCE_THRESHOLD = 0.01


def lab3_root() -> Path:
    """Вернуть путь к корню lab3."""
    return Path(__file__).resolve().parents[1]


def resolve_path(path_value: str, base_dir: Path) -> Path:
    """Преобразовать путь в абсолютный относительно lab3."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return base_dir / path


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Загрузить датасет из CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Файл датасета не найден: {csv_path}")
    return pd.read_csv(csv_path)


def validate_columns(df: pd.DataFrame) -> None:
    """Проверить наличие обязательных колонок."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise KeyError(
            "В датасете отсутствуют обязательные колонки: "
            + ", ".join(missing_columns)
        )


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Заполнить пропуски в числовых и категориальных колонках."""
    filled_df = df.copy()
    numeric_columns = filled_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [
        column for column in filled_df.columns if column not in numeric_columns
    ]

    for column in numeric_columns:
        if filled_df[column].isna().any():
            filled_df[column] = filled_df[column].fillna(filled_df[column].median())

    for column in categorical_columns:
        if filled_df[column].isna().any():
            modes = filled_df[column].mode(dropna=True)
            if modes.empty:
                raise ValueError(
                    f"Невозможно заполнить пропуски в колонке '{column}': нет моды"
                )
            filled_df[column] = filled_df[column].fillna(modes.iloc[0])

    return filled_df


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Отделить признаки от целевой переменной."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    """Закодировать категориальные признаки через one-hot encoding."""
    return pd.get_dummies(X, columns=CATEGORICAL_COLUMNS, dtype=int)


def identify_important_features(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = DEFAULT_FEATURE_IMPORTANCE_THRESHOLD,
) -> tuple[list[str], dict[str, float]]:
    """Определить важные признаки через RandomForestRegressor."""
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=DEFAULT_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X, y)

    feature_importance = dict(zip(X.columns, model.feature_importances_))
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    important_features = [
        feature for feature, importance in sorted_features if importance >= threshold
    ]

    if not important_features:
        n_keep = max(1, len(sorted_features) // 2)
        important_features = [feature for feature, _ in sorted_features[:n_keep]]

    return important_features, feature_importance


def apply_feature_selection(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    important_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Оставить только важные признаки в train/test."""
    return X_train[important_features], X_test[important_features]


def save_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
) -> None:
    """Сохранить train/test выборки в CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_frame(TARGET_COLUMN).to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_frame(TARGET_COLUMN).to_csv(output_dir / "y_test.csv", index=False)


def save_feature_metadata(
    feature_importance: dict[str, float],
    important_features: list[str],
    output_dir: Path,
) -> None:
    """Сохранить важности признаков и список выбранных признаков."""
    feature_importance_df = pd.DataFrame(
        sorted(feature_importance.items(), key=lambda item: item[1], reverse=True),
        columns=["feature", "importance"],
    )
    feature_importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    (output_dir / "important_features.txt").write_text(
        "\n".join(important_features) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Предобработка insurance.csv, выделение признаков и train/test split"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Доля тестовой выборки (по умолчанию 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Seed для воспроизводимого разбиения (по умолчанию 42)",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/insurance.csv",
        help="Путь к входному CSV относительно lab3 или абсолютный путь",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Путь к директории для выходных файлов относительно lab3 или абсолютный путь",
    )
    parser.add_argument(
        "--feature-threshold",
        type=float,
        default=DEFAULT_FEATURE_IMPORTANCE_THRESHOLD,
        help="Порог важности признаков (по умолчанию 0.01)",
    )
    args = parser.parse_args()

    if not 0 < args.test_size < 1:
        raise ValueError("Параметр --test-size должен быть в диапазоне (0, 1)")
    if args.feature_threshold < 0:
        raise ValueError("Параметр --feature-threshold должен быть неотрицательным")

    root = lab3_root()
    input_path = resolve_path(args.input_path, root)
    output_dir = resolve_path(args.output_dir, root)

    df = load_dataset(input_path)
    validate_columns(df)
    missing_values_count = int(df.isna().sum().sum())
    df = fill_missing_values(df)

    X, y = split_features_and_target(df)
    X_encoded = encode_categorical_features(X)
    important_features, feature_importance = identify_important_features(
        X_encoded,
        y,
        threshold=args.feature_threshold,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    X_train_selected, X_test_selected = apply_feature_selection(
        X_train,
        X_test,
        important_features,
    )

    save_split(X_train_selected, X_test_selected, y_train, y_test, output_dir)
    save_feature_metadata(feature_importance, important_features, output_dir)

    print(f"Исходный файл: {input_path}")
    print(f"Строк: {len(df)}, признаков после кодирования: {X_encoded.shape[1]}")
    print(f"Обнаружено пропусков до заполнения: {missing_values_count}")
    print(f"Выбрано важных признаков: {len(important_features)} из {X_encoded.shape[1]}")
    print(f"Train: {X_train_selected.shape}, Test: {X_test_selected.shape}")
    print(f"Файлы сохранены в: {output_dir}")


if __name__ == "__main__":
    main()
