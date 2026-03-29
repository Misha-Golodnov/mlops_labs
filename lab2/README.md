# Лабораторная работа 2: CI/CD Pipeline для ML проекта

## Описание

Лабораторная работа по разработке CI/CD pipeline с использованием Jenkins для автоматизации процесса машинного обучения на датасете Wine Quality Red.

**Статус**: ✅ Этап обработки данных **выполнен**

## Архитектура проекта

```
lab2/
├── data/
│   ├── raw/                           # Сырые данные
│   │   └── winequality-red.csv        # Исходный датасет (1599 строк)
│   └── processed/                     # Обработанные данные
│       ├── X_train.csv                # Признаки для обучения (1271 × 11)
│       ├── X_test.csv                 # Признаки для тестирования (318 × 11)
│       ├── y_train.csv                # Целевая переменная для обучения
│       ├── y_test.csv                 # Целевая переменная для тестирования
│       ├── scaler.pkl                 # Сохраненный StandardScaler
│       ├── feature_importance.csv     # Важность признаков
│       ├── important_features.txt     # Список выбранных признаков
│       └── preprocessing_config.json  # Конфигурация обработки
├── scripts/
│   ├── load_and_split_wine_quality.py # Исходный скрипт (базовая обработка)
│   ├── data_preprocessing.py          # Полная обработка данных ✨ НОВОЕ
│   └── load_processed_data.py         # Утилита загрузки данных ✨ НОВОЕ
├── Jenkinsfile                        # Pipeline конфигурация
├── requirements.txt                   # Python зависимости
├── README.md                          # Этот файл
└── DATA_PREPROCESSING.md              # Документация по обработке
```

## Выполненные работы

### ✅ Этап 2: Обработка данных и выделение признаков

#### 2.1 Загрузка данных
- **Источник**: UCI Machine Learning Repository
- **Датасет**: Wine Quality Red
- **URL**: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
- **Размер**: 1599 строк × 12 колонок

#### 2.2 Обработка данных
- ✅ Проверка и обработка пропусков (0 найдено)
- ✅ Удаление выбросов методом Z-score (удалено 10 строк)
- ✅ Разделение на train/test (80%/20%, стратификация)
- ✅ Масштабирование признаков (StandardScaler)

#### 2.3 Выделение важных признаков
- ✅ Анализ важности признаков (Random Forest)
- ✅ Отбор значимых признаков

**Топ-5 признаков по важности:**
1. **alcohol** (27.8%)
2. **sulphates** (14.0%)
3. **volatile acidity** (11.1%)
4. **total sulfur dioxide** (8.3%)
5. **chlorides** (6.5%)

#### 2.4 Генерация датасетов
- ✅ X_train.csv (1271 × 11)
- ✅ X_test.csv (318 × 11)
- ✅ y_train.csv (1271 × 1)
- ✅ y_test.csv (318 × 1)

#### 2.5 Сохранение артефактов
- ✅ StandardScaler (scaler.pkl)
- ✅ Информация о важности (feature_importance.csv)
- ✅ Список признаков (important_features.txt)
- ✅ Конфигурация обработки (preprocessing_config.json)

## Установка и запуск

### Требования
- Python 3.8+
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0

### Установка зависимостей
```bash
pip install -r requirements.txt
```

### Запуск обработки данных

**Базовый запуск:**
```bash
python scripts/data_preprocessing.py
```

**С параметрами:**
```bash
# Пользовательская доля тестовых данных (по умолчанию 0.2)
python scripts/data_preprocessing.py --test-size 0.3

# Пользовательский random state
python scripts/data_preprocessing.py --random-state 123

# Изменить порог важности признаков
python scripts/data_preprocessing.py --feature-threshold 0.02

# Переустановить сырой датасет с UCI
python scripts/data_preprocessing.py --force-download
```

### Загрузка обработанных данных

В своих скриптах используйте утилиту для загрузки:

```python
from scripts.load_processed_data import (
    load_all_data,
    load_scaler,
    load_feature_importance,
    get_data_summary
)

# Загрузить все данные
X_train, y_train, X_test, y_test = load_all_data()

# Получить информацию
summary = get_data_summary()
print(f"Train samples: {summary['train_samples']}")
print(f"Test samples: {summary['test_samples']}")

# Загрузить scaler для предсказаний
scaler = load_scaler()
```

## Структура данных

### Входные данные (raw)
- **winequality-red.csv**: 1599 строк, разделитель `;`
- **Признаки**: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
- **Целевая переменная**: quality (0-10)

### Выходные данные (processed)
- **X_train.csv**: Масштабированные признаки для обучения
- **X_test.csv**: Масштабированные признаки для тестирования
- **y_train.csv**: Целевые значения для обучения
- **y_test.csv**: Целевые значения для тестирования
- **Формат**: CSV с запятой в качестве разделителя
- **Масштабирование**: Z-score нормализация (mean=0, std=1)

## Параметры обработки

| Параметр | Значение | Описание |
|----------|----------|---------|
| Train/Test split | 80/20 | Разделение данных |
| Random state | 42 | Для воспроизводимости |
| Стратификация | Да | По целевой переменной |
| Масштабирование | StandardScaler | Z-score нормализация |
| Метод отбора признаков | Random Forest | Mean Decrease in Impurity |
| Порог удаления выбросов | Z=3.0 | 3 стандартных отклонения |

## Примечания Jenkins

Для использования в Jenkins pipeline:

```groovy
stage('Data Preprocessing') {
    steps {
        sh 'cd lab2 && python scripts/data_preprocessing.py'
    }
}
```

**Требования для агента:**
- ✅ Интернет доступ до UCI (для загрузки датасета)
- ✅ Python 3.8+ с pip
- ✅ Установленные зависимости (pandas, scikit-learn, numpy)

## Документация

- [DATA_PREPROCESSING.md](DATA_PREPROCESSING.md) - Подробное описание этапов обработки
- [preprocessing_config.json](data/processed/preprocessing_config.json) - Конфигурация обработки

## Воспроизводимость

Все обработки используют `random_state=42` для гарантированной воспроизводимости. Для переобработки данных с другим random state используйте:

```bash
python scripts/data_preprocessing.py --random-state <YOUR_VALUE>
```

## Статистика

| Метрика | Значение |
|---------|----------|
| Исходные записи | 1599 |
| После обработки | 1589 |
| Удалено выбросов | 10 |
| Признаков | 11 |
| Тренировочные образцы | 1271 |
| Тестовые образцы | 318 |
| Разделение | 80% / 20% |

## Следующие этапы

- [ ] Этап 3: Обучение модели машинного обучения
- [ ] Этап 4: Оценка качества модели
- [ ] Этап 5: Развертывание в Jenkins pipeline

## Автор

MLOps Lab 2 - Data Preprocessing Pipeline
2026-03-29


Для этапа подготовки датасета:
- У агента должен быть доступ в интернет до UCI (скачивание CSV)
- В job должен быть checkout репозитория (в declarative pipeline он обычно выполняется сам)