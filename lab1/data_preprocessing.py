from collections import Counter

import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


#Работаем с треин
healthy_or_not = df['target'].value_counts().tolist()
values = [healthy_or_not[0], healthy_or_not[1]]


numerical_columns = df.select_dtypes([int, float]).columns.tolist()
if 'target' in numerical_columns:
    numerical_columns.remove('target')


categorical_columns = df.select_dtypes(include=['object', 'string'])

#функция для поиска выбросов
def IQR (df: pd.DataFrame, thr: int, features: list):

    outlier_list = []
    for column in features:

        Q1 = np.percentile(df[column], 25)
        Q3 = np.percentile(df[column], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5*IQR
        outlier_list_column = df[(df[column]< Q1 - outlier_step) | (df[column] > Q3 + outlier_step)].index
        outlier_list.extend(outlier_list_column)

    outlier_list = Counter(outlier_list)
    multiple_outliers = list( k for k, v in outlier_list.items() if v > thr)

    out1 = df[df[column] < Q1 - outlier_step]
    out2 = df[df[column] > Q3 + outlier_step]

    print('Общее число удаленных выбросов:', out1.shape[0] + out2.shape[0])

    return multiple_outliers

Outliers_IQR = IQR(df,1,numerical_columns)

#удаляем выбросы в треин
df = df.drop(Outliers_IQR, axis = 0).reset_index(drop=True)

# Работаем с тест
healthy_or_not_test = df_test['target'].value_counts().tolist()

numerical_columns_test = df_test.select_dtypes([int, float]).columns.tolist()
if 'target' in numerical_columns_test:
    numerical_columns_test.remove('target')

categorical_columns_test = df_test.select_dtypes(include=['object', 'string']).columns.tolist()

Outliers_IQR_test = IQR(df_test, 1, numerical_columns_test)

##удаляем выбросы в треин
df_test = df_test.drop(Outliers_IQR_test, axis=0).reset_index(drop=True)



