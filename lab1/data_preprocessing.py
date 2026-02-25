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

# функция для поиска выбросов


def IQR(df: pd.DataFrame, thr: int, features: list):
    outlier_list = []
    for column in features:
        col_vals = df[column].dropna()
        Q1 = np.percentile(col_vals, 25)
        Q3 = np.percentile(col_vals, 75)
        iqr = Q3 - Q1
        outlier_step = 1.5 * iqr
        outlier_list_column = df[(df[column] < Q1 - outlier_step)
                                 | (df[column] > Q3 + outlier_step)].index.tolist()
        outlier_list.extend(outlier_list_column)

    outlier_counts = Counter(outlier_list)
    multiple_outliers = [k for k, v in outlier_counts.items() if v > thr]
    logger.info(f'удалено выбросов: {len(multiple_outliers)}')
    return multiple_outliers


# удаление выбросов
Outliers_IQR = IQR(df, 1, numerical_columns)
df = df.drop(Outliers_IQR, axis=0).reset_index(drop=True)

Outliers_IQR_test = IQR(df_test, 1, numerical_columns)
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
df.to_csv("train_preprocessed.csv", index=False)
df_test.to_csv("test_preprocessed.csv", index=False)
logger.info(
    f"данные сохранены. Финальные размеры | обучение: {df.shape}, тест: {df_test.shape}")
