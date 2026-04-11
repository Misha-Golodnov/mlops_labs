"""
Этап конвейера: обучение модели регрессии на подготовленных данных
и сохранение обученного артефакта (lab3 — insurance dataset).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

DEFAULT_MODEL_FILENAME = "model.pkl"
DEFAULT_N_ESTIMATORS = 300
DEFAULT_RANDOM_STATE = 42


def lab3_root() -> Path:
    """Получить путь к корню lab3."""
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    """Преобразовать относительный путь в абсолютный относительно lab3."""
    if path.is_absolute():
        return path
    return lab3_root() / path


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Загрузить обучающую выборку из processed-директории."""
    processed_dir = lab3_root() / "data" / "processed"
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze()
    return X_train, y_train


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Обучение модели регрессии Insurance и сохранение в pickle"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("data") / "processed" / DEFAULT_MODEL_FILENAME,
        help="Путь для сохранения модели (по умолчанию data/processed/model.pkl)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
        help="Число деревьев (по умолчанию 300)",
    )
    return parser.parse_args()


def build_model(n_estimators: int) -> GradientBoostingRegressor:
    """Создать GradientBoostingRegressor с фиксированными параметрами."""
    return GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=5,
        random_state=DEFAULT_RANDOM_STATE,
    )


def save_model(model: GradientBoostingRegressor, model_path: Path) -> None:
    """Сохранить обученную модель в pickle."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Модель сохранена: {model_path}")


def main() -> None:
    args = parse_args()
    model_path = resolve_path(args.model_path)

    print("=" * 60)
    print("Этап обучения модели (Insurance)")
    print("=" * 60)

    X_train, y_train = load_training_data()
    print(f"Размер train-набора: {X_train.shape[0]} строк, {X_train.shape[1]} признаков")
    print(f"Признаки: {list(X_train.columns)}")

    model = build_model(args.n_estimators)
    print(f"\nМодель: {type(model).__name__}")
    print("Обучение...")
    model.fit(X_train, y_train)

    save_model(model, model_path)
    print("\nОбучение завершено.")


if __name__ == "__main__":
    main()
