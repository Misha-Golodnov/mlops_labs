"""
Этап конвейера: обработка данных, выделение важных признаков,
формирование train/test датасетов для модели машинного обучения.

Задачи:
- Загрузка сырого датасета
- Обработка пропусков и аномалий
- Масштабирование признаков
- Выделение важных признаков
- Формирование train/test выборок
- Сохранение обработанных данных
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

WINE_QUALITY_RED_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
TARGET_COLUMN = "quality"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_FEATURE_IMPORTANCE_THRESHOLD = 0.01


def lab2_root() -> Path:
    return Path(__file__).resolve().parents[1]


def download_raw(csv_path: Path, force: bool) -> None:
    """Загрузить сырой датасет с UCI."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and not force:
        return
    print(f"Загрузка датасета с {WINE_QUALITY_RED_URL}...")
    urlretrieve(WINE_QUALITY_RED_URL, csv_path)
    print(f"Датасет сохранен: {csv_path}")


def load_frame(csv_path: Path) -> pd.DataFrame:
    """Загрузить датасет из CSV."""
    return pd.read_csv(csv_path, sep=";")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обработать пропуски в данных.
    Для Wine Quality датасета обычно нет пропусков,
    но добавляем для общности.
    """
    print(f"Пропусков в датасете: {df.isnull().sum().sum()}")
    # Заполняем пропуски медианой (если они есть)
    df_filled = df.fillna(df.median())
    return df_filled


def remove_outliers(df: pd.DataFrame, target_col: str, z_threshold: float = 3.0) -> pd.DataFrame:
    """
    Удалить выбросы используя метод Z-score.
    Исключаем строки, где целевая переменная более чем на z_threshold стандартных отклонений от среднего.
    """
    z_scores = np.abs((df[target_col] - df[target_col].mean()) / df[target_col].std())
    mask = z_scores < z_threshold
    print(f"Удалено выбросов: {(~mask).sum()} из {len(df)}")
    return df[mask]


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Разделить признаки и целевую переменную."""
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"В датасете нет колонки «{TARGET_COLUMN}»")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def identify_important_features(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
    threshold: float = DEFAULT_FEATURE_IMPORTANCE_THRESHOLD,
) -> tuple[list[str], dict[str, float]]:
    """
    Определить важные признаки используя Random Forest.
    Возвращает список важных признаков и словарь важностей.
    """
    print(f"\nВыявление важных признаков ({n_estimators} деревьев)...")
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf_model.fit(X, y)
    
    feature_importance = dict(zip(X.columns, rf_model.feature_importances_))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\nИмпортантность признаков:")
    for feature, importance in sorted_importance:
        print(f"  {feature}: {importance:.4f}")
    
    # Выбираем признаки с важностью >= threshold
    important_features = [f for f, imp in sorted_importance if imp >= threshold]
    
    if not important_features:
        # Если изменить порог не нашел признаки, беремtop-50%
        n_keep = max(1, len(X.columns) // 2)
        important_features = [f for f, _ in sorted_importance[:n_keep]]
    
    print(f"\nВыбрано важных признаков: {len(important_features)} из {len(X.columns)}")
    print(f"Признаки: {important_features}")
    
    return important_features, feature_importance


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Масштабировать признаки используя StandardScaler.
    Scaler обучается на train и применяется к test.
    """
    print("\nМасштабирование признаков...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled_df, X_test_scaled_df, scaler


def apply_feature_selection(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    important_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Оставить только важные признаки."""
    print(f"\nОтбор важных признаков...")
    X_train_selected = X_train[important_features]
    X_test_selected = X_test[important_features]
    print(f"X_train shape: {X_train_selected.shape}")
    print(f"X_test shape: {X_test_selected.shape}")
    return X_train_selected, X_test_selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wine Quality: обработка данных и выделение признаков"
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
        default=42,
        help="Seed для воспроизводимости split",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Перезаписать сырой CSV с UCI",
    )
    parser.add_argument(
        "--feature-threshold",
        type=float,
        default=DEFAULT_FEATURE_IMPORTANCE_THRESHOLD,
        help="Порог важности признаков (по умолчанию 0.01)",
    )
    args = parser.parse_args()

    root = lab2_root()
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    raw_csv = raw_dir / "winequality-red.csv"

    # === Этап 1: Загрузка ===
    print("=" * 60)
    print("Этап 1: Загрузка данных")
    print("=" * 60)
    download_raw(raw_csv, force=args.force_download)
    df = load_frame(raw_csv)
    print(f"Загруженные данные: {df.shape[0]} строк, {df.shape[1]} колонок")

    # === Этап 2: Обработка данных ===
    print("\n" + "=" * 60)
    print("Этап 2: Обработка данных")
    print("=" * 60)
    
    # Обработка пропусков
    df = handle_missing_values(df)
    
    # Удаление выбросов (опционально)
    df = remove_outliers(df, TARGET_COLUMN, z_threshold=3.0)
    
    # Разделение на признаки и целевую переменную
    X, y = split_features_and_target(df)
    print(f"После обработки: X.shape={X.shape}, y.shape={y.shape}")

    # === Этап 3: Выделение важных признаков ===
    print("\n" + "=" * 60)
    print("Этап 3: Выделение важных признаков")
    print("=" * 60)
    important_features, feature_importance_dict = identify_important_features(
        X, y, threshold=args.feature_threshold
    )

    # === Этап 4: Формирование train/test выборок ===
    print("\n" + "=" * 60)
    print("Этап 4: Формирование train/test выборок")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # === Этап 5: Масштабирование ===
    print("\n" + "=" * 60)
    print("Этап 5: Масштабирование признаков")
    print("=" * 60)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # === Этап 6: Отбор важных признаков ===
    print("\n" + "=" * 60)
    print("Этап 6: Отбор важных признаков")
    print("=" * 60)
    X_train_final, X_test_final = apply_feature_selection(
        X_train_scaled, X_test_scaled, important_features
    )

    # === Этап 7: Сохранение результатов ===
    print("\n" + "=" * 60)
    print("Этап 7: Сохранение результатов")
    print("=" * 60)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем обработанные датасеты
    X_train_final.to_csv(processed_dir / "X_train.csv", index=False)
    X_test_final.to_csv(processed_dir / "X_test.csv", index=False)
    y_train.to_frame(TARGET_COLUMN).to_csv(processed_dir / "y_train.csv", index=False)
    y_test.to_frame(TARGET_COLUMN).to_csv(processed_dir / "y_test.csv", index=False)
    print(f"✓ Сохранены X_train.csv, X_test.csv, y_train.csv, y_test.csv")

    # Сохраняем scaler
    scaler_path = processed_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"✓ Скейлер сохранен: {scaler_path}")

    # Сохраняем информацию о важности признаков
    feature_importance_csv = processed_dir / "feature_importance.csv"
    feature_importance_df = pd.DataFrame(
        sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True),
        columns=["feature", "importance"]
    )
    feature_importance_df.to_csv(feature_importance_csv, index=False)
    print(f"✓ Важность признаков сохранена: {feature_importance_csv}")

    # Сохраняем список важных признаков
    features_list_path = processed_dir / "important_features.txt"
    with open(features_list_path, "w") as f:
        f.write("\n".join(important_features))
    print(f"✓ Список важных признаков сохранен: {features_list_path}")

    # === Финальный отчет ===
    print("\n" + "=" * 60)
    print("ФИНАЛЬНЫЙ ОТЧЕТ")
    print("=" * 60)
    print(f"Исходные данные: {df.shape[0]} строк, {X.shape[1]} признаков")
    print(f"Выбрано признаков: {len(important_features)}")
    print(f"Train размер: {X_train_final.shape}")
    print(f"Test размер: {X_test_final.shape}")
    print(f"\nВсе артефакты сохранены в: {processed_dir}")


if __name__ == "__main__":
    main()
