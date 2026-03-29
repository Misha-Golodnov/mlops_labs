# Отчет: Обработка данных Wine Quality Red

**Дата:** 2026-03-29  
**Статус:** ✅ Завершено  
**Версия:** 1.0

---

## Резюме

Успешно разработан и реализован полный конвейер обработки данных для датасета Wine Quality Red. Все этапы выполнены, все артефакты сохранены.

---

## Выполненные задачи

### 1. Загрузка данных ✅
- **Источник**: UCI Machine Learning Repository
- **Датасет**: winequality-red.csv
- **Размер**: 1599 строк, 12 колонок
- **Формат**: CSV, разделитель `;`
- **Статус**: Успешно загружен

### 2. Обработка пропусков ✅
- **Найдено пропусков**: 0
- **Метод**: fill_with_median (не потребовался)
- **Результат**: Данные готовы к обработке

### 3. Удаление выбросов ✅
- **Метод**: Z-score нормализация
- **Порог**: 3.0 стандартных отклонения
- **Удалено**: 10 строк
- **Осталось**: 1589 строк
- **Процент данных**: 99.4% сохранено

### 4. Разделение на train/test ✅
- **Метод**: Stratified train_test_split
- **Тестовая доля**: 20%
- **Стратификация**: По целевой переменной `quality`
- **Train размер**: 1271 строка (79.8%)
- **Test размер**: 318 строк (20.2%)
- **Random seed**: 42 (для воспроизводимости)

### 5. Выделение важных признаков ✅
- **Метод**: Random Forest (n_estimators=100)
- **Метрика**: Mean Decrease in Impurity
- **Всего признаков**: 11
- **Выбрано признаков**: 11 (все имеют значимость > 1%)

**Распределение важности признаков:**
```
1.  alcohol               27.82 % ███████████████████████████
2.  sulphates            13.96 % ██████████████
3.  volatile acidity     11.13 % ███████████
4.  total sulfur dioxide  8.25 % ████████
5.  chlorides             6.50 % ███████
6.  pH                    6.12 % ██████
7.  residual sugar        5.54 % ██████
8.  density               5.53 % ██████
9.  citric acid           5.27 % █████
10. fixed acidity         5.15 % █████
11. free sulfur dioxide   4.74 % █████
```

### 6. Масштабирование признаков ✅
- **Метод**: StandardScaler (Z-score нормализация)
- **Формула**: x_new = (x - mean) / std
- **Параметры**: with_mean=True, with_std=True
- **Результат**: Все признаки имеют mean≈0 и std=1

### 7. Сохранение артефактов ✅
- **X_train.csv**: Обработанные признаки для обучения
- **X_test.csv**: Обработанные признаки для тестирования
- **y_train.csv**: Целевые значения для обучения
- **y_test.csv**: Целевые значения для тестирования
- **scaler.pkl**: Сохраненный StandardScaler для future predictions
- **feature_importance.csv**: Важность каждого признака
- **important_features.txt**: Упорядоченный список признаков
- **preprocessing_config.json**: Полная конфигурация обработки

---

## Созданные скрипты

### 1. scripts/data_preprocessing.py (НОВОЕ) ✨
**Назначение**: Полная обработка данных с выделением признаков

**Функции:**
- `download_raw()` - загрузка данных с UCI
- `load_frame()` - чтение CSV файла
- `handle_missing_values()` - обработка пропусков
- `remove_outliers()` - удаление выбросов (Z-score)
- `split_features_and_target()` - разделение признаков/целевой переменной
- `identify_important_features()` - анализ важности (Random Forest)
- `scale_features()` - масштабирование (StandardScaler)
- `apply_feature_selection()` - отбор важных признаков
- `main()` - основной конвейер

**Параметры коммандной строки:**
- `--test-size` (float, default=0.2)
- `--random-state` (int, default=42)
- `--force-download` (flag)
- `--feature-threshold` (float, default=0.01)

**Запуск:**
```bash
python scripts/data_preprocessing.py
python scripts/data_preprocessing.py --test-size 0.3
python scripts/data_preprocessing.py --feature-threshold 0.02
```

### 2. scripts/load_processed_data.py (НОВОЕ) ✨
**Назначение**: Утилита для удобной загрузки обработанных данных

**Функции:**
- `load_training_data()` - загрузить train данные
- `load_test_data()` - загрузить test данные
- `load_all_data()` - загрузить все данные сразу
- `load_scaler()` - загрузить сохраненный scaler
- `load_feature_importance()` - загрузить значимость признаков
- `load_important_features()` - загрузить список признаков
- `scale_data()` - применить scaler к новым данным
- `inverse_scale_data()` - обратное преобразование
- `get_data_summary()` - получить сводку по данным

**Пример использования:**
```python
from scripts.load_processed_data import load_all_data, load_scaler

X_train, y_train, X_test, y_test = load_all_data()
scaler = load_scaler()
```

**Запуск:**
```bash
python scripts/load_processed_data.py  # Демонстрация
```

---

## Созданные документы

### 1. DATA_PREPROCESSING.md (НОВОЕ) ✨
Подробная документация по процессу обработки данных:
- Описание каждого этапа
- Таблица важности признаков
- Статистика датасета
- Примеры использования
- Формулы методов

### 2. preprocessing_config.json (НОВОЕ) ✨
Конфигурационный файл с полной информацией о обработке:
- Метаданные обработки
- Статистика данных
- Параметры train/test split
- Информация о признаках
- Пути к выходным файлам
- Примечания по использованию

---

## Структура выходных данных

```
data/processed/
├── X_train.csv                  # 1271 × 11 [признаки для обучения]
├── X_test.csv                   # 318 × 11 [признаки для тестирования]
├── y_train.csv                  # 1271 × 1 [целевые значения train]
├── y_test.csv                   # 318 × 1 [целевые значения test]
├── scaler.pkl                   # [StandardScaler для future predictions]
├── feature_importance.csv       # [важность каждого признака]
├── important_features.txt       # [список признаков]
└── preprocessing_config.json    # [конфигурация обработки]
```

**Размеры файлов:**
- X_train.csv: 277 KB
- X_test.csv: 69.6 KB
- y_train.csv: 3.8 KB
- y_test.csv: 963 bytes
- scaler.pkl: 961 bytes
- feature_importance.csv: 389 bytes
- important_features.txt: 147 bytes
- **Всего**: ~352 KB

---

## Обновленные файлы

### 1. requirements.txt
Добавлена зависимость:
- `numpy>=1.24.0`

**Все зависимости:**
```
pandas>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
```

### 2. README.md
Обновлен с информацией о:
- Полной архитектуре проекта
- Выполненных работах
- Параметрах обработки
- Инструкциях по запуску
- Примерах использования
- Требованиях для Jenkins

---

## Качество обработки

| Параметр | Значение |
|----------|----------|
| Сохранено данных | 99.4% (1589 / 1599) |
| Сохранено признаков | 100% (11 / 11) |
| Выбросов удалено | 0.63% (10 / 1589) |
| Train/Test сбалансированность | Стратифицирована по quality |
| Воспроизводимость | Гарантирована (seed=42) |

---

## Проверка целостности

✅ **Все файлы созданы и проверены**

```
✓ X_train.csv (1271 × 11)
✓ X_test.csv (318 × 11)  
✓ y_train.csv (1271 × 1)
✓ y_test.csv (318 × 1)
✓ scaler.pkl (StandardScaler)
✓ feature_importance.csv (11 записей)
✓ important_features.txt (11 признаков)
✓ preprocessing_config.json (конфигурация)
```

✅ **Все скрипты протестированы:**
```
✓ data_preprocessing.py (успешно запущен)
✓ load_processed_data.py (все функции работают)
```

✅ **Данные корректны:**
```
✓ Нет NaN значений
✓ Размеры соответствуют ожидаемым
✓ Значения масштабированы правильно
✓ Train/test split стратифицирован
```

---

## Использование в Jenkins

Для интеграции в Jenkins pipeline добавить:

```groovy
pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Install Dependencies') {
            steps {
                sh '''
                cd lab2
                pip install -r requirements.txt
                '''
            }
        }
        
        stage('Data Preprocessing') {
            steps {
                sh '''
                cd lab2
                python scripts/data_preprocessing.py
                '''
            }
        }
        
        stage('Verify Output') {
            steps {
                sh '''
                cd lab2
                ls -la data/processed/
                python scripts/load_processed_data.py
                '''
            }
        }
    }
}
```

---

## Заключение

✅ **Задача "Обработка данных" полностью выполнена**

Разработан, протестирован и документирован полный конвейер обработки данных Wine Quality Red. Все артефакты готовы к использованию в следующих этапах:
- Обучение модели ml 
- Оценка качества
- Развертывание в production

**Готово к интеграции в Jenkins pipeline.**

---

*Отчет создан: 2026-03-29*  
*версия: 1.0*
