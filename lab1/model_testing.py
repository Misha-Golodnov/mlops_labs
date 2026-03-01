"""
Скрипт тестирования обученной модели на данных из test.csv.
Загружает model.pkl и test.csv (после предобработки),
вычисляет метрики и выводит результат в стандартный поток вывода.
"""
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_DATA_PATH = Path("test.csv")
MODEL_PATH = Path("model.pkl")


def load_test_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Загружает тестовые данные и разделяет на признаки и целевую переменную."""
    if not path.exists():
        logger.error(f"Файл {path} не найден.")
        sys.exit(1)
    df = pd.read_csv(path)
    if "target" not in df.columns:
        raise ValueError("В тестовых данных отсутствует столбец 'target'")
    X = df.drop(columns=["target"])
    y = df["target"]
    logger.info(f"Загружено тестовых объектов: {len(df)}, признаков: {X.shape[1]}")
    return X, y


def load_model(path: Path):
    """Загружает модель из pickle-файла."""
    if not path.exists():
        logger.error(f"Файл модели {path} не найден. Сначала запустите model_preparation.py.")
        sys.exit(1)
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Модель загружена из {path}")
    return model


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Вычисляет метрики качества модели."""
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")

    # ROC-AUC только если модель поддерживает predict_proba
    try:
        y_proba = model.predict_proba(X)
        if y_proba.shape[1] == 2:
            auc = roc_auc_score(y, y_proba[:, 1])
        else:
            auc = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")
    except AttributeError:
        auc = None

    logger.info("\n" + classification_report(y, y_pred))

    return {"accuracy": acc, "f1": f1, "roc_auc": auc}


def main():
    X_test, y_test = load_test_data(TEST_DATA_PATH)
    model = load_model(MODEL_PATH)

    metrics = evaluate(model, X_test, y_test)

    # Итоговая строка в stdout (требование задания)
    auc_str = f", ROC-AUC: {metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else ""
    print(
        f"Accuracy: {metrics['accuracy']:.4f}"
        f" | F1-score: {metrics['f1']:.4f}"
        f"{auc_str}"
    )


if __name__ == "__main__":
    main()
