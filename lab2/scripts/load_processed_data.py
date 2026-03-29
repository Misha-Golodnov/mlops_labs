"""
Утилита для загрузки обработанных данных и сохраненного scaler.

Этот модуль предоставляет удобные функции для:
- Загрузки train/test датасетов
- Загрузки сохраненного scaler
- Загрузки информации о важности признаков
- Применения scaler к новым данным
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_processed_data_dir() -> Path:
    """Получить путь к директории с обработанными данными."""
    return Path(__file__).resolve().parents[1] / "data" / "processed"


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Загрузить тренировочные данные.
    
    Returns:
        Кортеж (X_train, y_train)
            X_train: DataFrame с признаками
            y_train: Series с целевыми значениями
    """
    data_dir = get_processed_data_dir()
    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
    return X_train, y_train


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Загрузить тестовые данные.
    
    Returns:
        Кортеж (X_test, y_test)
            X_test: DataFrame с признаками
            y_test: Series с целевыми значениями
    """
    data_dir = get_processed_data_dir()
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()
    return X_test, y_test


def load_all_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Загрузить все данные (train + test).
    
    Returns:
        Кортеж (X_train, y_train, X_test, y_test)
    """
    X_train, y_train = load_training_data()
    X_test, y_test = load_test_data()
    return X_train, y_train, X_test, y_test


def load_scaler() -> StandardScaler:
    """
    Загрузить сохраненный scaler.
    
    Returns:
        StandardScaler объект, обученный на тренировочных данных
    """
    data_dir = get_processed_data_dir()
    scaler_path = data_dir / "scaler.pkl"
    
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler не найден: {scaler_path}")
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    return scaler


def load_feature_importance() -> pd.DataFrame:
    """
    Загрузить информацию о важности признаков.
    
    Returns:
        DataFrame с колонками [feature, importance], отсортированный по убыванию важности
    """
    data_dir = get_processed_data_dir()
    return pd.read_csv(data_dir / "feature_importance.csv")


def load_important_features() -> list[str]:
    """
    Загрузить список выбранных важных признаков.
    
    Returns:
        Список наименований признаков в порядке важности
    """
    data_dir = get_processed_data_dir()
    features_path = data_dir / "important_features.txt"
    
    if not features_path.exists():
        raise FileNotFoundError(f"Файл признаков не найден: {features_path}")
    
    with open(features_path, "r") as f:
        features = [line.strip() for line in f if line.strip()]
    
    return features


def scale_data(X: pd.DataFrame, scaler: Optional[StandardScaler] = None) -> pd.DataFrame:
    """
    Применить scaler к данным.
    
    Args:
        X: DataFrame с данными
        scaler: StandardScaler объект (если None, загружается сохраненный)
    
    Returns:
        Масштабированные данные
    """
    if scaler is None:
        scaler = load_scaler()
    
    X_scaled = scaler.transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)


def inverse_scale_data(X_scaled: pd.DataFrame, scaler: Optional[StandardScaler] = None) -> pd.DataFrame:
    """
    Обратное преобразование масштабированных данных в исходную шкалу.
    
    Args:
        X_scaled: Масштабированные данные
        scaler: StandardScaler объект (если None, загружается сохраненный)
    
    Returns:
        Данные в исходной шкале
    """
    if scaler is None:
        scaler = load_scaler()
    
    X_inverse = scaler.inverse_transform(X_scaled)
    return pd.DataFrame(X_inverse, columns=X_scaled.columns, index=X_scaled.index)


def get_data_summary() -> dict:
    """
    Получить сводку по обработанным данным.
    
    Returns:
        Словарь с информацией о размерах и структуре данных
    """
    X_train, y_train, X_test, y_test = load_all_data()
    features = load_important_features()
    importance = load_feature_importance()
    
    return {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": X_train.shape[1],
        "features": features,
        "feature_count": len(features),
        "train_target_shape": y_train.shape,
        "test_target_shape": y_test.shape,
        "target_name": "quality",
        "top_5_important_features": importance.head(5)["feature"].tolist(),
    }


if __name__ == "__main__":
    # Пример использования
    print("=" * 60)
    print("Загрузка обработанных данных")
    print("=" * 60)
    
    # Загрузить и показать информацию
    summary = get_data_summary()
    print(f"\nДанные:")
    print(f"  Train samples: {summary['train_samples']}")
    print(f"  Test samples: {summary['test_samples']}")
    print(f"  Features: {summary['n_features']}")
    print(f"  Target: {summary['target_name']}")
    
    print(f"\nТоп-5 важных признаков:")
    for i, feature in enumerate(summary['top_5_important_features'], 1):
        print(f"  {i}. {feature}")
    
    # Загрузить данные
    X_train, y_train, X_test, y_test = load_all_data()
    print(f"\n✓ Загружены данные)")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    
    # Загрузить scaler
    scaler = load_scaler()
    print(f"\n✓ Загружен scaler")
    print(f"  Тип: {type(scaler).__name__}")
    print(f"  Mean: {scaler.mean_[:3]}...")  # Показать первые 3 значения
    print(f"  Scale: {scaler.scale_[:3]}...")
