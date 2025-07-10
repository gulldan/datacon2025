# Модули для Task 3: Машинное обучение

Этот документ описывает модули, созданные для выполнения Task 3 - обучения моделей для предсказания биологической активности.

## Структура модулей

### 1. `data_preparation.py`
**Назначение**: Подготовка данных для машинного обучения

**Основные классы и функции**:
- `DataPreparation` - класс для подготовки данных
- `prepare_data_pipeline()` - полный пайплайн подготовки данных

**Функциональность**:
- Загрузка данных из parquet/csv файлов
- Очистка данных от пропусков и некорректных значений
- Фильтрация признаков по дисперсии
- Нормализация/стандартизация признаков (StandardScaler, MinMaxScaler)
- Разделение на обучающую и тестовую выборки
- Применение PCA для снижения размерности

### 2. `classical_models.py`
**Назначение**: Обучение классических моделей машинного обучения

**Основные классы и функции**:
- `ClassicalModels` - класс для работы с классическими моделями
- `train_classical_models()` - обучение набора классических моделей

**Поддерживаемые модели**:
- Linear Regression (с использованием polars-ds)
- ElasticNet (с использованием polars-ds)
- Fallback на sklearn для совместимости

**Особенности**:
- Использует polars-ds как основную библиотеку (согласно требованиям task3)
- Автоматический fallback на sklearn при ошибках
- Отслеживание времени обучения
- Вычисление метрик MAE, MSE, RMSE, R²

### 3. `neural_models.py`
**Назначение**: Обучение нейросетевых моделей с PyTorch

**Основные классы и функции**:
- `MLPRegressor` - многослойный перцептрон
- `CNNRegressor` - сверточная нейронная сеть
- `NeuralNetworkTrainer` - класс-тренер для нейросетей
- `train_neural_models()` - обучение набора нейросетевых моделей

**Архитектуры**:
- **MLP**: Полносвязная сеть с BatchNorm, Dropout, настраиваемыми слоями
- **CNN**: 1D сверточная сеть с несколькими размерами ядер (3, 5, 7)

**Особенности**:
- Early stopping с настраиваемым patience
- Gradient clipping для стабильности обучения
- AdamW оптимизатор с learning rate scheduling
- Поддержка CUDA/CPU
- Воспроизводимые результаты через random seeds

### 4. `cross_validation.py`
**Назначение**: Кросс-валидация моделей

**Основные классы и функции**:
- `CrossValidation` - класс для проведения кросс-валидации
- `calculate_cv_stats()` - вычисление статистик CV
- `run_cross_validation()` - запуск полной кросс-валидации

**Функциональность**:
- K-fold кросс-валидация (по умолчанию 5 фолдов)
- Поддержка как классических, так и нейросетевых моделей
- Вычисление статистик: среднее, стандартное отклонение, min, max, коэффициент вариации
- Обработка ошибок и пропущенных значений

### 5. `model_comparison.py`
**Назначение**: Сравнение и визуализация результатов моделей

**Основные функции**:
- `create_performance_comparison()` - график сравнения производительности
- `create_cv_stability_plot()` - график стабильности моделей
- `create_training_time_comparison()` - сравнение времени обучения
- `create_prediction_scatter()` - scatter plot предсказаний vs истинных значений
- `create_combined_comparison_dashboard()` - комбинированный dashboard
- `create_model_ranking()` - рейтинг моделей по взвешенным метрикам
- `save_visualizations()` - сохранение графиков в HTML

**Технологии**:
- Plotly для интерактивных визуализаций (согласно требованиям task3)
- Polars DataFrame для обработки результатов
- Взвешенная система рейтинга моделей

## Использование

### Базовый пример:

```python
from src.data_preparation import prepare_data_pipeline
from src.classical_models import train_classical_models
from src.neural_models import train_neural_models
from src.cross_validation import run_cross_validation
from src.model_comparison import create_performance_comparison

# 1. Подготовка данных
data = prepare_data_pipeline(
    file_path="data/descriptors/cox2_best_combined.parquet",
    target_column="pIC50",
    test_size=0.2,
    normalization="standard"
)

# 2. Обучение моделей
classical_results = train_classical_models(
    data['X_train'], data['y_train'], 
    data['X_test'], data['y_test']
)

neural_results = train_neural_models(
    data['X_train'], data['y_train'], 
    data['X_test'], data['y_test'], 
    epochs=100
)

# 3. Кросс-валидация
cv_results, cv_stats = run_cross_validation(
    data['X_train'], data['y_train']
)

# 4. Визуализация
fig = create_performance_comparison({**classical_results, **neural_results})
fig.show()
```

## Соответствие требованиям Task 3

### ✅ Технологический стек:
- **Polars**: Используется во всех модулях для обработки данных
- **polars-ds**: Основная библиотека для классических ML моделей
- **PyTorch**: Для нейросетевых моделей (MLP, CNN)
- **Plotly**: Для всех визуализаций (вместо matplotlib)
- **Google style docstrings**: Везде используются
- **Type hints**: Полная типизация всех функций

### ✅ Модели:
- **2 классические**: LinearRegression, ElasticNet
- **2 нейросетевые**: MLP, CNN

### ✅ Функциональность:
- Подготовка данных с нормализацией/стандартизацией
- Кросс-валидация с воспроизводимыми результатами
- Сравнение моделей по метрикам MAE, RMSE, R²
- Интерактивные визуализации
- Анализ стабильности и времени обучения

### ✅ Воспроизводимость:
- Random seeds во всех модулях
- Контролируемое разделение данных
- Детерминированные результаты

## Зависимости

```toml
[dependencies]
polars = "^0.20.0"
polars-ds = "^0.1.0"  # Для классических ML моделей
torch = "^2.0.0"      # Для нейросетей
plotly = "^5.0.0"     # Для визуализаций
scikit-learn = "^1.3.0"  # Для метрик и fallback
numpy = "^1.24.0"
```

## Примечания

1. **polars-ds**: Экспериментальная библиотека, поэтому предусмотрены fallback'и на sklearn
2. **Производительность**: Все модули оптимизированы для работы с большими датасетами
3. **Расширяемость**: Архитектура позволяет легко добавлять новые модели и метрики
4. **Документация**: Все функции имеют подробные docstring'и с примерами 