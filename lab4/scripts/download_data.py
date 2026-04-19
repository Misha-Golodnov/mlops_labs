import pandas as pd
from catboost.datasets import titanic

train_df, test_df = titanic()

df = train_df.copy()

df.to_csv('data/titanic.csv', index=False)

print("Полный датасет Титаник сохранён: data/titanic.csv")
print(f"Размер: {df.shape}")