"""Модуль для кросс-валидации моделей машинного обучения.

Содержит функции для проведения k-fold кросс-валидации
как классических, так и нейросетевых моделей.
"""

import logging
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import KFold

from .classical_models import ClassicalModels
from .neural_models import NeuralNetworkTrainer

logger = logging.getLogger(__name__)


class CrossValidation:
    """Класс для проведения кросс-валидации."""

    def __init__(self, n_splits: int = 5, random_state: int = 42, shuffle: bool = True):
        """Инициализация кросс-валидации.

        Args:
            n_splits: Количество фолдов.
            random_state: Seed для воспроизводимости.
            shuffle: Перемешивать ли данные перед разделением.
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        self.kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def cross_validate_classical(
        self, X: pl.DataFrame, y: pl.Series, model_configs: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, list[float]]]:
        """Проводит кросс-валидацию для классических моделей.

        Args:
            X: Признаки.
            y: Целевая переменная.
            model_configs: Конфигурации моделей.

        Returns:
            Словарь с результатами кросс-валидации.
        """
        results = {}

        # Конвертируем в numpy для sklearn KFold
        X_array = X.to_numpy()
        y_array = y.to_numpy()

        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(X_array)):
            logger.info(f"Фолд {fold + 1}/{self.n_splits}")

            # Разделяем данные
            X_train_fold = pl.DataFrame(X_array[train_idx], schema=X.columns)
            X_val_fold = pl.DataFrame(X_array[val_idx], schema=X.columns)
            y_train_fold = pl.Series(y_array[train_idx])
            y_val_fold = pl.Series(y_array[val_idx])

            # Обучаем модели для этого фолда
            models = ClassicalModels(random_state=self.random_state)

            for model_name, config in model_configs.items():
                if model_name not in results:
                    results[model_name] = {"mae": [], "mse": [], "rmse": [], "r2": [], "training_time": []}

                try:
                    if config["type"] == "linear_regression":
                        models.train_linear_regression(X_train_fold, y_train_fold, f"{model_name}_fold_{fold}")
                    elif config["type"] == "elastic_net":
                        models.train_elastic_net(
                            X_train_fold,
                            y_train_fold,
                            alpha=config.get("alpha", 1.0),
                            l1_ratio=config.get("l1_ratio", 0.5),
                            model_name=f"{model_name}_fold_{fold}",
                        )
                    else:
                        logger.warning(f"Неизвестный тип модели: {config['type']}")
                        continue

                    # Оцениваем модель
                    metrics = models.evaluate_model(f"{model_name}_fold_{fold}", X_val_fold, y_val_fold)

                    for metric_name, value in metrics.items():
                        results[model_name][metric_name].append(value)

                except Exception as e:
                    logger.error(f"Ошибка при обучении {model_name} на фолде {fold}: {e}")
                    # Добавляем NaN для этого фолда
                    for metric_name in ["mae", "mse", "rmse", "r2", "training_time"]:
                        results[model_name][metric_name].append(np.nan)

        return results

    def cross_validate_neural(
        self, X: pl.DataFrame, y: pl.Series, model_configs: dict[str, dict[str, Any]], epochs: int = 50
    ) -> dict[str, dict[str, list[float]]]:
        """Проводит кросс-валидацию для нейросетевых моделей.

        Args:
            X: Признаки.
            y: Целевая переменная.
            model_configs: Конфигурации моделей.
            epochs: Количество эпох обучения.

        Returns:
            Словарь с результатами кросс-валидации.
        """
        results = {}

        # Конвертируем в numpy для sklearn KFold
        X_array = X.to_numpy()
        y_array = y.to_numpy()

        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(X_array)):
            logger.info(f"Фолд {fold + 1}/{self.n_splits} (нейросети)")

            # Разделяем данные
            X_train_fold = pl.DataFrame(X_array[train_idx], schema=X.columns)
            X_val_fold = pl.DataFrame(X_array[val_idx], schema=X.columns)
            y_train_fold = pl.Series(y_array[train_idx])
            y_val_fold = pl.Series(y_array[val_idx])

            # Обучаем модели для этого фолда
            trainer = NeuralNetworkTrainer(random_state=self.random_state + fold)

            for model_name, config in model_configs.items():
                if model_name not in results:
                    results[model_name] = {"mae": [], "mse": [], "rmse": [], "r2": [], "training_time": []}

                try:
                    if config["type"] == "mlp":
                        trainer.train_mlp(
                            X_train_fold,
                            y_train_fold,
                            hidden_sizes=config.get("hidden_sizes", [512, 256, 128]),
                            dropout_rate=config.get("dropout_rate", 0.3),
                            epochs=epochs,
                            patience=config.get("patience", 15),
                            model_name=f"{model_name}_fold_{fold}",
                        )
                    elif config["type"] == "cnn":
                        trainer.train_cnn(
                            X_train_fold,
                            y_train_fold,
                            num_filters=config.get("num_filters", 64),
                            kernel_sizes=config.get("kernel_sizes", [3, 5, 7]),
                            dropout_rate=config.get("dropout_rate", 0.3),
                            epochs=epochs,
                            patience=config.get("patience", 15),
                            model_name=f"{model_name}_fold_{fold}",
                        )
                    else:
                        logger.warning(f"Неизвестный тип модели: {config['type']}")
                        continue

                    # Оцениваем модель
                    metrics = trainer.evaluate_model(f"{model_name}_fold_{fold}", X_val_fold, y_val_fold)

                    for metric_name, value in metrics.items():
                        results[model_name][metric_name].append(value)

                except Exception as e:
                    logger.error(f"Ошибка при обучении {model_name} на фолде {fold}: {e}")
                    # Добавляем NaN для этого фолда
                    for metric_name in ["mae", "mse", "rmse", "r2", "training_time"]:
                        results[model_name][metric_name].append(np.nan)

        return results

    def cross_validate_all(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        classical_configs: dict[str, dict[str, Any]] | None = None,
        neural_configs: dict[str, dict[str, Any]] | None = None,
        neural_epochs: int = 50,
    ) -> dict[str, dict[str, list[float]]]:
        """Проводит кросс-валидацию для всех типов моделей.

        Args:
            X: Признаки.
            y: Целевая переменная.
            classical_configs: Конфигурации классических моделей.
            neural_configs: Конфигурации нейросетевых моделей.
            neural_epochs: Количество эпох для нейросетей.

        Returns:
            Объединенные результаты кросс-валидации.
        """
        results = {}

        # Дефолтные конфигурации
        if classical_configs is None:
            classical_configs = {
                "LinearRegression": {"type": "linear_regression"},
                "ElasticNet": {"type": "elastic_net", "alpha": 0.1, "l1_ratio": 0.5},
            }

        if neural_configs is None:
            neural_configs = {
                "MLP": {"type": "mlp", "hidden_sizes": [512, 256, 128], "dropout_rate": 0.3},
                "CNN": {"type": "cnn", "num_filters": 64, "kernel_sizes": [3, 5, 7], "dropout_rate": 0.3},
            }

        # Классические модели
        if classical_configs:
            logger.info("Начинаем кросс-валидацию классических моделей")
            classical_results = self.cross_validate_classical(X, y, classical_configs)
            results.update(classical_results)

        # Нейросетевые модели
        if neural_configs:
            logger.info("Начинаем кросс-валидацию нейросетевых моделей")
            neural_results = self.cross_validate_neural(X, y, neural_configs, neural_epochs)
            results.update(neural_results)

        return results


def calculate_cv_stats(cv_results: dict[str, dict[str, list[float]]]) -> pl.DataFrame:
    """Вычисляет статистики кросс-валидации.

    Args:
        cv_results: Результаты кросс-валидации.

    Returns:
        DataFrame со статистиками.
    """
    stats_data = []

    for model_name, metrics in cv_results.items():
        model_stats: dict[str, Any] = {"model": model_name}

        for metric_name, values in metrics.items():
            # Фильтруем NaN значения
            clean_values = [v for v in values if not np.isnan(v)]

            if clean_values:
                model_stats[f"{metric_name}_mean"] = np.mean(clean_values)
                model_stats[f"{metric_name}_std"] = np.std(clean_values)
                model_stats[f"{metric_name}_min"] = np.min(clean_values)
                model_stats[f"{metric_name}_max"] = np.max(clean_values)
                model_stats[f"{metric_name}_cv"] = (
                    np.std(clean_values) / np.mean(clean_values) if np.mean(clean_values) != 0 else 0
                )
            else:
                model_stats[f"{metric_name}_mean"] = np.nan
                model_stats[f"{metric_name}_std"] = np.nan
                model_stats[f"{metric_name}_min"] = np.nan
                model_stats[f"{metric_name}_max"] = np.nan
                model_stats[f"{metric_name}_cv"] = np.nan

        stats_data.append(model_stats)

    return pl.DataFrame(stats_data)


def run_cross_validation(
    X: pl.DataFrame, y: pl.Series, n_splits: int = 5, neural_epochs: int = 50, random_state: int = 42
) -> tuple[dict[str, dict[str, list[float]]], pl.DataFrame]:
    """Запускает полную кросс-валидацию всех моделей.

    Args:
        X: Признаки.
        y: Целевая переменная.
        n_splits: Количество фолдов.
        neural_epochs: Количество эпох для нейросетей.
        random_state: Seed для воспроизводимости.

    Returns:
        Кортеж (результаты CV, статистики).
    """
    cv = CrossValidation(n_splits=n_splits, random_state=random_state)

    # Проводим кросс-валидацию
    cv_results = cv.cross_validate_all(X, y, neural_epochs=neural_epochs)

    # Вычисляем статистики
    cv_stats = calculate_cv_stats(cv_results)

    logger.info(f"Кросс-валидация завершена для {len(cv_results)} моделей")

    return cv_results, cv_stats
