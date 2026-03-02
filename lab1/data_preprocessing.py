from collections import Counter
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

# логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# загрузка данных
df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

logger.info(
    f"форма данных на обучении: {df.shape}, форма тестовых данных: {df_test.shape}")

# выделение признаков
numerical_columns = df.select_dtypes([int, float]).columns.tolist()
if 'target' in numerical_columns:
    numerical_columns.remove('target')

categorical_columns = df.select_dtypes(
    include=['object', 'string']).columns.tolist()

logger.info(f"числовые признаки: {numerical_columns}")
logger.info(f"категориальные признаки: {categorical_columns}")

# функция для вычисления границ IQR по данным (только для train)
def compute_iqr_bounds(df: pd.DataFrame, features: list) -> dict:
    """Вычисляет границы выбросов (Q1 - 1.5*IQR, Q3 + 1.5*IQR) для каждого признака."""
    bounds = {}
    for column in features:
        if column not in df.columns:
            continue
        col_vals = df[column].dropna()
        Q1 = np.percentile(col_vals, 25)
        Q3 = np.percentile(col_vals, 75)
        iqr = Q3 - Q1
        outlier_step = 1.5 * iqr
        bounds[column] = (Q1 - outlier_step, Q3 + outlier_step)
    return bounds


def find_outliers(df: pd.DataFrame, thr: int, features: list, bounds: dict) -> list:
    """Находит индексы строк-выбросов по предвычисленным границам."""
    outlier_list = []
    for column in features:
        if column not in bounds or column not in df.columns:
            continue
        lower, upper = bounds[column]
        outlier_list_column = df[(df[column] < lower) | (df[column] > upper)].index.tolist()
        outlier_list.extend(outlier_list_column)

    outlier_counts = Counter(outlier_list)
    multiple_outliers = [k for k, v in outlier_counts.items() if v > thr]
    return multiple_outliers


# удаление выбросов
iqr_bounds = compute_iqr_bounds(df, numerical_columns)
Outliers_IQR = find_outliers(df, 1, numerical_columns, iqr_bounds)
df = df.drop(Outliers_IQR, axis=0).reset_index(drop=True)

Outliers_IQR_test = find_outliers(df_test, 1, numerical_columns, iqr_bounds)
df_test = df_test.drop(Outliers_IQR_test, axis=0).reset_index(drop=True)

# масштабирование числовых признаков
scaler = StandardScaler()
if numerical_columns:
    scaler.fit(df[numerical_columns])
    df[numerical_columns] = scaler.transform(df[numerical_columns])
    cols_to_scale = [c for c in numerical_columns if c in df_test.columns]
    df_test[cols_to_scale] = scaler.transform(df_test[cols_to_scale])

logger.info("числовые признаки масштабированы")

# кодирование категориальных признаков
if categorical_columns:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(df[categorical_columns])

    df_cat = pd.DataFrame(
        encoder.transform(df[categorical_columns]),
        columns=encoder.get_feature_names_out(categorical_columns)
    )
    df_test_cat = pd.DataFrame(
        encoder.transform(df_test[categorical_columns]),
        columns=encoder.get_feature_names_out(categorical_columns)
    )

    df = pd.concat([df.drop(categorical_columns, axis=1), df_cat], axis=1)
    df_test = pd.concat(
        [df_test.drop(categorical_columns, axis=1), df_test_cat], axis=1)

    logger.info("текстовые признаки переведены в чилосовой формат")

# сохранение предобработанных данных
df.to_csv("train.csv", index=False)
df_test.to_csv("test.csv", index=False)
logger.info(
    f"данные сохранены. Финальные размеры | обучение: {df.shape}, тест: {df_test.shape}")
