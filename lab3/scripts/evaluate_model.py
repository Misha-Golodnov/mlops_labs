"""
Этап конвейера: загрузка обученной модели и оценка качества
на тестовых данных с сохранением метрик в JSON (lab3 — insurance dataset).
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_MODEL_PATH = Path("data") / "processed" / "model.pkl"
DEFAULT_METRICS_PATH = Path("data") / "processed" / "model_metrics.json"

# Порог R² для признания модели качественной
R2_THRESHOLD = 0.80


def lab3_root() -> Path:
    """Получить путь к корню lab3."""
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    """Преобразовать относительный путь в абсолютный относительно lab3."""
    if path.is_absolute():
        return path
    return lab3_root() / path


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Загрузить тестовую выборку из processed-директории."""
    processed_dir = lab3_root() / "data" / "processed"
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze()
    return X_test, y_test


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Оценка качества обученной модели Insurance на тестовых данных"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Путь к сохраненной модели (по умолчанию data/processed/model.pkl)",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Путь для сохранения JSON-отчета с метриками",
    )
    return parser.parse_args()


def load_model(model_path: Path) -> Any:
    """Загрузить сохраненную модель из pickle."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {model_path}. "
            "Сначала выполните train_model.py."
        )
    with open(model_path, "rb") as f:
        return pickle.load(f)


def calculate_metrics(y_true: pd.Series, y_pred) -> dict[str, float]:
    """Посчитать основные метрики регрессии."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_metrics(metrics_report: dict[str, Any], metrics_path: Path) -> None:
    """Сохранить отчет с метриками в JSON."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, ensure_ascii=False, indent=2)
    print(f"Метрики сохранены: {metrics_path}")


def evaluate_quality(r2: float) -> None:
    """Вывести вердикт о качестве модели."""
    print("\n" + "=" * 60)
    if r2 >= R2_THRESHOLD:
        print(f"ВЕРДИКТ: Модель качественная (R² = {r2:.4f} >= {R2_THRESHOLD})")
    else:
        print(f"ВЕРДИКТ: Модель не достигает порога (R² = {r2:.4f} < {R2_THRESHOLD})")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    model_path = resolve_path(args.model_path)
    metrics_path = resolve_path(args.metrics_path)

    print("=" * 60)
    print("Этап оценки модели (Insurance)")
    print("=" * 60)

    model = load_model(model_path)
    X_test, y_test = load_test_data()
    print(f"Размер test-набора: {X_test.shape[0]} строк, {X_test.shape[1]} признаков")

    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)

    print("\nМетрики на тестовой выборке:")
    print(f"  MAE  = {metrics['mae']:>10.2f}")
    print(f"  RMSE = {metrics['rmse']:>10.2f}")
    print(f"  MSE  = {metrics['mse']:>10.2f}")
    print(f"  R²   = {metrics['r2']:>10.4f}")

    report = {
        "model": type(model).__name__,
        "test_size": int(X_test.shape[0]),
        "metrics": metrics,
        "quality_threshold_r2": R2_THRESHOLD,
        "quality_pass": metrics["r2"] >= R2_THRESHOLD,
    }
    save_metrics(report, metrics_path)
    evaluate_quality(metrics["r2"])


if __name__ == "__main__":
    main()
