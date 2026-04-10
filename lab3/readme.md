# Lab 3: Insurance Data Preprocessing

В этой лабораторной используется локальный датасет `data/insurance.csv`.
Скрипт предобработки выполняет базовую очистку данных, кодирует категориальные
признаки, выделяет важные признаки и сохраняет разбиение на train/test.

## Что делает скрипт

- Загружает `lab3/data/insurance.csv`
- Проверяет наличие обязательных колонок
- Заполняет пропуски:
  - числовые признаки: медиана
  - категориальные признаки: мода
- Кодирует `sex`, `smoker`, `region` через one-hot encoding
- Оценивает важность признаков через `RandomForestRegressor`
- Отбирает признаки по порогу важности
- Разделяет данные на train/test через `train_test_split`
- Сохраняет результаты в `lab3/data/processed`

## Выходные файлы

После запуска создаются:

- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`
- `data/processed/feature_importance.csv`
- `data/processed/important_features.txt`

## Запуск

Из папки `lab3`:

```bash
python scripts/data_preprocessing.py
```

Пример с параметрами:

```bash
python scripts/data_preprocessing.py --test-size 0.25 --random-state 123
```

Пример с пользовательским порогом важности:

```bash
python scripts/data_preprocessing.py --feature-threshold 0.02
```
