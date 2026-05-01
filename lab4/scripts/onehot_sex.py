import os

import pandas as pd

LAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(LAB_DIR, "data", "titanic_age_filled.csv")
OUTPUT_PATH = os.path.join(LAB_DIR, "data", "titanic_onehot.csv")

df = pd.read_csv(INPUT_PATH)

df = pd.get_dummies(df, columns=["Sex"], prefix="Sex", dtype=int)

df.to_csv(OUTPUT_PATH, index=False)

print(f"Вход: {INPUT_PATH}")
print(f"Колонки после one-hot Sex: {list(df.columns)}")
print(f"Результат: {OUTPUT_PATH}")
print(f"Размер: {df.shape}")
