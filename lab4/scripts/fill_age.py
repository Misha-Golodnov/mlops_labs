import os
import pandas as pd

LAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(LAB_DIR, "data", "titanic.csv")
OUTPUT_PATH = os.path.join(LAB_DIR, "data", "titanic_age_filled.csv")

df = pd.read_csv(INPUT_PATH)

missing_before = df["age"].isna().sum()
mean_age = df["age"].mean()

df["age"] = df["age"].fillna(mean_age)

df.to_csv(OUTPUT_PATH, index=False)

print(f"Исходный датасет: {INPUT_PATH}")
print(f"Пропущенных значений в age до заполнения: {missing_before}")
print(f"Среднее значение age: {mean_age:.2f}")
print(f"Пропущенных значений в age после заполнения: {df['age'].isna().sum()}")
print(f"Результат сохранён: {OUTPUT_PATH}")
print(f"Размер датасета: {df.shape}")
