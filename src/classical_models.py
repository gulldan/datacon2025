"""Модуль для обучения классических моделей машинного обучения.

Реализует Random Forest, Gradient Boosting (XGBoost/LightGBM)
и другие ensemble-методы как требуется в Task 3.
"""

import logging
import time
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class ClassicalModels:
    """Класс для работы с классическими ensemble-моделями машинного обучения."""

    def __init__(self, random_state: int = 42) -> None:
        """Инициализация с установкой random state.

        Args:
            random_state: Seed для воспроизводимости результатов.
        """
        self.random_state = random_state
        self.models: dict[str, Any] = {}
        self.training_times: dict[str, float] = {}

    def train_random_forest(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        n_estimators: int = 100,
        max_depth: int | None = None,
        model_name: str = "RandomForest",
    ) -> None:
        """Обучает модель Random Forest Regressor.

        Args:
            X_train: Обучающие признаки.
            y_train: Целевая переменная.
            n_estimators: Количество деревьев в лесу.
            max_depth: Максимальная глубина деревьев.
            model_name: Название модели для сохранения.
        """
        from sklearn.ensemble import RandomForestRegressor

        start_time = time.time()

        # Конвертируем в numpy
        X_array = X_train.to_numpy()
        y_array = y_train.to_numpy()

        # Обучаем модель
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1,  # Используем все доступные ядра
        )
        model.fit(X_array, y_array)

        # Сохраняем модель
        self.models[model_name] = {
            "type": "random_forest",
            "model": model,
            "feature_columns": X_train.columns,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
        }

        training_time = time.time() - start_time
        self.training_times[model_name] = training_time

        logger.info(
            f"Обучена модель {model_name} (n_estimators={n_estimators}, max_depth={max_depth}) за {training_time:.3f} сек"
        )

    def train_xgboost(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        model_name: str = "XGBoost",
    ) -> None:
        """Обучает модель XGBoost Regressor.

        Args:
            X_train: Обучающие признаки.
            y_train: Целевая переменная.
            n_estimators: Количество бустинг-раундов.
            max_depth: Максимальная глубина деревьев.
            learning_rate: Скорость обучения.
            model_name: Название модели для сохранения.
        """
        try:
            import xgboost as xgb  # type: ignore
        except ImportError:
            logger.warning("XGBoost не установлен, используем GradientBoostingRegressor")
            return self._train_gradient_boosting_fallback(X_train, y_train, n_estimators, max_depth, learning_rate, model_name)

        start_time = time.time()

        # Конвертируем в numpy
        X_array = X_train.to_numpy()
        y_array = y_train.to_numpy()

        # Обучаем модель
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_array, y_array)

        # Сохраняем модель
        self.models[model_name] = {
            "type": "xgboost",
            "model": model,
            "feature_columns": X_train.columns,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        }

        training_time = time.time() - start_time
        self.training_times[model_name] = training_time

        logger.info(
            f"Обучена модель {model_name} (n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}) за {training_time:.3f} сек"
        )
        return None

    def train_lightgbm(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        model_name: str = "LightGBM",
    ) -> None:
        """Обучает модель LightGBM Regressor.

        Args:
            X_train: Обучающие признаки.
            y_train: Целевая переменная.
            n_estimators: Количество бустинг-раундов.
            max_depth: Максимальная глубина деревьев (-1 для неограниченной).
            learning_rate: Скорость обучения.
            model_name: Название модели для сохранения.
        """
        try:
            import lightgbm as lgb  # type: ignore
        except ImportError:
            logger.warning("LightGBM не установлен, используем GradientBoostingRegressor")
            return self._train_gradient_boosting_fallback(X_train, y_train, n_estimators, max_depth, learning_rate, model_name)

        start_time = time.time()

        # Конвертируем в numpy
        X_array = X_train.to_numpy()
        y_array = y_train.to_numpy()

        # Обучаем модель
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(X_array, y_array)

        # Сохраняем модель
        self.models[model_name] = {
            "type": "lightgbm",
            "model": model,
            "feature_columns": X_train.columns,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        }

        training_time = time.time() - start_time
        self.training_times[model_name] = training_time

        logger.info(
            f"Обучена модель {model_name} (n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}) за {training_time:.3f} сек"
        )
        return None

    def _train_gradient_boosting_fallback(
        self, X_train: pl.DataFrame, y_train: pl.Series, n_estimators: int, max_depth: int, learning_rate: float, model_name: str
    ) -> None:
        """Fallback метод для обучения Gradient Boosting через sklearn.

        Args:
            X_train: Обучающие признаки.
            y_train: Целевая переменная.
            n_estimators: Количество бустинг-раундов.
            max_depth: Максимальная глубина деревьев.
            learning_rate: Скорость обучения.
            model_name: Название модели для сохранения.
        """
        from sklearn.ensemble import GradientBoostingRegressor

        start_time = time.time()

        # Конвертируем в numpy
        X_array = X_train.to_numpy()
        y_array = y_train.to_numpy()

        # Обучаем модель
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth > 0 else 3,  # sklearn не поддерживает -1
            learning_rate=learning_rate,
            random_state=self.random_state,
        )
        model.fit(X_array, y_array)

        # Сохраняем модель
        self.models[model_name] = {
            "type": "gradient_boosting",
            "model": model,
            "feature_columns": X_train.columns,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        }

        training_time = time.time() - start_time
        self.training_times[model_name] = training_time

        logger.info(f"Обучена sklearn модель {model_name} (GradientBoosting) за {training_time:.3f} сек")

    def predict(self, model_name: str, X_test: pl.DataFrame) -> np.ndarray:
        """Делает предсказания с помощью обученной модели.

        Args:
            model_name: Название модели.
            X_test: Тестовые признаки.

        Returns:
            Массив предсказаний.
        """
        if model_name not in self.models:
            msg = f"Модель {model_name} не найдена"
            raise ValueError(msg)

        model_info = self.models[model_name]

        # Конвертируем в numpy и делаем предсказания
        X_array = X_test.to_numpy()
        return model_info["model"].predict(X_array)

    def evaluate_model(self, model_name: str, X_test: pl.DataFrame, y_test: pl.Series) -> dict[str, float]:
        """Оценивает производительность модели.

        Args:
            model_name: Название модели.
            X_test: Тестовые признаки.
            y_test: Истинные значения.

        Returns:
            Словарь с метриками.
        """
        # Делаем предсказания
        y_pred = self.predict(model_name, X_test)
        y_true = y_test.to_numpy()

        # Вычисляем метрики
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "training_time": self.training_times.get(model_name, 0.0)}

        logger.info(f"Метрики для {model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

        return metrics

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        """Возвращает информацию о модели.

        Args:
            model_name: Название модели.

        Returns:
            Словарь с информацией о модели.
        """
        if model_name not in self.models:
            msg = f"Модель {model_name} не найдена"
            raise ValueError(msg)

        return self.models[model_name].copy()

    def list_models(self) -> list[str]:
        """Возвращает список названий обученных моделей.

        Returns:
            Список названий моделей.
        """
        return list(self.models.keys())


def train_classical_models(
    X_train: pl.DataFrame, y_train: pl.Series, X_test: pl.DataFrame, y_test: pl.Series, random_state: int = 42
) -> dict[str, dict[str, float]]:
    """Обучает и оценивает набор классических ensemble-моделей как требуется в Task 3.

    Args:
        X_train: Обучающие признаки.
        y_train: Обучающая целевая переменная.
        X_test: Тестовые признаки.
        y_test: Тестовая целевая переменная.
        random_state: Seed для воспроизводимости.

    Returns:
        Словарь с метриками для каждой модели.
    """
    models = ClassicalModels(random_state=random_state)

    # Обучаем требуемые модели согласно Task 3
    logger.info("Обучение Random Forest...")
    models.train_random_forest(X_train, y_train, n_estimators=100, max_depth=10)

    logger.info("Обучение XGBoost/Gradient Boosting...")
    models.train_xgboost(X_train, y_train, n_estimators=100, max_depth=6, learning_rate=0.1)

    # Оцениваем модели
    results = {}
    for model_name in models.list_models():
        metrics = models.evaluate_model(model_name, X_test, y_test)
        results[model_name] = metrics

    return results
