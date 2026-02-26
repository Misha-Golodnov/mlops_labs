"""
Скрипт создания и обучения модели машинного обучения на данных из train.csv.
Сохраняет обученную модель в файл с помощью pickle.
"""
import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# путь к обучающим данным (файл train.csv в текущей папке lab1)
TRAIN_DATA_PATH = Path("train.csv")
MODEL_PATH = Path("model.pkl")


def load_train_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Загружает обучающие данные и разделяет на признаки и целевую переменную."""
    df = pd.read_csv(path)
    if "target" not in df.columns:
        raise ValueError("В данных отсутствует столбец 'target'")
    X = df.drop(columns=["target"])
    y = df["target"]
    logger.info(f"Загружено объектов: {len(df)}, признаков: {X.shape[1]}")
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series):
    """Обучает модель классификации."""
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    logger.info("Модель обучена")
    return model


def train_model_with_grid_search(
    X: pd.DataFrame, y: pd.Series, cv: int = 10
):
    """
    Перебор моделей и гиперпараметров по сетке (GridSearchCV).
    Возвращает лучшую обученную модель по accuracy.
    """
    param_grid = [
        {
            "model": LogisticRegression(random_state=42, max_iter=2000),
            "params": {
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "model__solver": ["lbfgs", "saga", "liblinear"],
                "model__class_weight": [None, "balanced"],
            },
        },
        {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "model__n_estimators": [50, 100, 150, 200, 300],
                "model__max_depth": [5, 7, 10, 15, 20, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        },
        {
            "model": SVC(random_state=42, probability=True),
            "params": {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__kernel": ["rbf", "linear", "poly"],
                "model__gamma": ["scale", "auto", 0.01, 0.1],
                "model__class_weight": [None, "balanced"],
            },
        },
        {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "model__n_estimators": [50, 100, 150, 200],
                "model__max_depth": [3, 5, 7, 10],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__min_samples_split": [2, 5],
                "model__subsample": [0.8, 1.0],
            },
        },
    ]

    best_score = -1.0
    best_estimator = None

    for grid_item in param_grid:
        pipe = Pipeline([("model", grid_item["model"])])
        search = GridSearchCV(
            pipe,
            grid_item["params"],
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X, y)
        logger.info(
            f"{grid_item['model'].__class__.__name__}: "
            f"best score={search.best_score_:.4f}, "
            f"best params={search.best_params_}"
        )
        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_estimator = search.best_estimator_

    logger.info(f"Выбрана модель с accuracy={best_score:.4f}")
    return best_estimator


def save_model(model, path: Path) -> None:
    """Сохраняет модель в файл с помощью pickle."""
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Модель сохранена в {path}")


def main():
    X, y = load_train_data(TRAIN_DATA_PATH)
    model = train_model_with_grid_search(X, y)
    save_model(model, MODEL_PATH)


if __name__ == "__main__":
    main()
