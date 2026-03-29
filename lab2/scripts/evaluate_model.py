"""
Этап конвейера: загрузка обученной модели и оценка качества
на тестовых данных с сохранением метрик в JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from load_processed_data import load_test_data

DEFAULT_MODEL_PATH = Path("data") / "processed" / "model.pkl"
DEFAULT_METRICS_PATH = Path("data") / "processed" / "model_metrics.json"


def lab2_root() -> Path:
    """Получить путь к корню lab2."""
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Оценка качества обученной модели Wine Quality на тестовых данных"
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
        help=(
            "Путь для сохранения JSON-отчета с метриками "
            "(по умолчанию data/processed/model_metrics.json)"
        ),
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Преобразовать относительный путь в абсолютный относительно lab2."""
    if path.is_absolute():
        return path
    return lab2_root() / path


def load_model(model_path: Path) -> Any:
    """Загрузить сохраненную модель из pickle."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {model_path}. "
            "Сначала выполните этап обучения модели."
        )

    with open(model_path, "rb") as f:
        return pickle.load(f)


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
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


def main() -> None:
    args = parse_args()
    model_path = resolve_path(args.model_path)
    metrics_path = resolve_path(args.metrics_path)

    print("=" * 60)
    print("Этап оценки модели")
    print("=" * 60)

    model = load_model(model_path)
    X_test, y_test = load_test_data()
    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)

    metrics_report = {
        "model_path": str(model_path),
        "model_type": type(model).__name__,
        "test_samples": len(X_test),
        "n_features": X_test.shape[1],
        "metrics": metrics,
    }
    save_metrics(metrics_report, metrics_path)

    print(f"Путь к модели: {model_path}")
    print(f"Тип модели: {type(model).__name__}")
    print(f"Размер тестового набора: {len(X_test)}")
    print(f"Количество признаков: {X_test.shape[1]}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2: {metrics['r2']:.4f}")
    print(f"JSON-отчет сохранен: {metrics_path}")


if __name__ == "__main__":
    main()
