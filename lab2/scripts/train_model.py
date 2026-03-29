"""
Этап конвейера: обучение модели регрессии на подготовленных данных
и сохранение обученного артефакта.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

from load_processed_data import load_training_data

DEFAULT_MODEL_FILENAME = "model.pkl"
DEFAULT_N_ESTIMATORS = 100
DEFAULT_RANDOM_STATE = 42


def lab2_root() -> Path:
    """Получить путь к корню lab2."""
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Обучение модели регрессии Wine Quality и сохранение в pickle"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("data") / "processed" / DEFAULT_MODEL_FILENAME,
        help="Путь для сохранения модели (по умолчанию data/processed/model.pkl)",
    )
    return parser.parse_args()


def build_model() -> RandomForestRegressor:
    """Создать модель с фиксированными параметрами для воспроизводимости."""
    return RandomForestRegressor(
        n_estimators=DEFAULT_N_ESTIMATORS,
        random_state=DEFAULT_RANDOM_STATE,
        n_jobs=-1,
    )


def resolve_model_path(model_path: Path) -> Path:
    """Преобразовать относительный путь в абсолютный относительно lab2."""
    if model_path.is_absolute():
        return model_path
    return lab2_root() / model_path


def save_model(model: RandomForestRegressor, model_path: Path) -> None:
    """Сохранить обученную модель в pickle."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


def fit_model(
    model: RandomForestRegressor, X_train, y_train
) -> RandomForestRegressor:
    """
    Обучить модель.

    В ограниченных средах multiprocessing/joblib может быть недоступен,
    поэтому делаем безопасный fallback на один поток.
    """
    try:
        model.fit(X_train, y_train)
        return model
    except PermissionError:
        fallback_model = RandomForestRegressor(
            n_estimators=DEFAULT_N_ESTIMATORS,
            random_state=DEFAULT_RANDOM_STATE,
            n_jobs=1,
        )
        fallback_model.fit(X_train, y_train)
        return fallback_model


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_path)

    print("=" * 60)
    print("Этап обучения модели")
    print("=" * 60)

    X_train, y_train = load_training_data()
    model = build_model()
    model = fit_model(model, X_train, y_train)
    save_model(model, model_path)

    print(f"Путь к модели: {model_path}")
    print(f"Тип модели: {type(model).__name__}")
    print(f"Размер train-набора: {len(X_train)}")
    print(f"Количество признаков: {X_train.shape[1]}")
    print("Модель успешно обучена и сохранена.")


if __name__ == "__main__":
    main()
