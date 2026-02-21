import zipfile
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Загружаем датасет
zip_path = Path("heart-failure-prediction.zip")
with zipfile.ZipFile(zip_path) as z:
    z.extractall("data")


csvs = list(Path("data").rglob("*.csv"))
print("CSV:", csvs)

df  = pd.read_csv(csvs[0])

x = df.iloc[:, :-1]
y_raw = df.iloc[:, -1]

le = LabelEncoder()
y = le.fit_transform(y_raw)

#Делим на треин и тест
X_train, X_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)

#Сохраняем результат
train_df = X_train.copy()
train_df['target'] = y_train

test_df = X_test.copy()
test_df['target'] = y_test

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("- train.csv (обучающие данные + target)")
print("- test.csv (тестовые данные + target)")
print(f"Размер train: {train_df.shape}")
print(f"Размер test: {test_df.shape}")





