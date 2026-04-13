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
- `data/processed/model.pkl`
- `data/processed/model_metrics.json`

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

## Обучение модели

```bash
python scripts/train_model.py
```

Обучает `GradientBoostingRegressor` (n_estimators=300, lr=0.05, max_depth=5) и сохраняет модель в `data/processed/model.pkl`.

## Оценка модели

```bash
python scripts/evaluate_model.py
```

Считает MAE, RMSE, R² на тестовой выборке и сохраняет отчёт в `data/processed/model_metrics.json`. Порог качества: R² >= 0.80.

Результат на тестовой выборке (268 строк):

- MAE: 2653
- RMSE: 4900
- R²: **0.845** — порог пройден


## Web app
```
pip install -r requirements.txt
python scripts/train_model.py
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```