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

    def __init__(self, n_splits: int = 5, random_state: int = 42, shuffle: bool = True) -> None:
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
        """Проводит кросс-валидацию для классических ensemble-моделей.

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
            models = ClassicalModels(random_state=self.random_state + fold)

            for model_name, config in model_configs.items():
                if model_name not in results:
                    results[model_name] = {"mae": [], "mse": [], "rmse": [], "r2": [], "training_time": []}

                try:
                    if config["type"] == "random_forest":
                        models.train_random_forest(
                            X_train_fold,
                            y_train_fold,
                            n_estimators=config.get("n_estimators", 100),
                            max_depth=config.get("max_depth", None),
                            model_name=f"{model_name}_fold_{fold}",
                        )
                    elif config["type"] == "xgboost":
                        models.train_xgboost(
                            X_train_fold,
                            y_train_fold,
                            n_estimators=config.get("n_estimators", 100),
                            max_depth=config.get("max_depth", 6),
                            learning_rate=config.get("learning_rate", 0.1),
                            model_name=f"{model_name}_fold_{fold}",
                        )
                    elif config["type"] == "lightgbm":
                        models.train_lightgbm(
                            X_train_fold,
                            y_train_fold,
                            n_estimators=config.get("n_estimators", 100),
                            max_depth=config.get("max_depth", -1),
                            learning_rate=config.get("learning_rate", 0.1),
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

        # Дефолтные конфигурации для ensemble-моделей согласно Task 3
        if classical_configs is None:
            classical_configs = {
                "random_forest": {"type": "random_forest", "n_estimators": 100, "max_depth": 10},
                "xgboost": {"type": "xgboost", "n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
            }

        if neural_configs is None:
            neural_configs = {
                "mlp": {"type": "mlp", "hidden_sizes": [512, 256, 128], "dropout_rate": 0.3},
                "cnn": {"type": "cnn", "num_filters": 64, "kernel_sizes": [3, 5, 7], "dropout_rate": 0.3},
            }

        # Классические модели
        if classical_configs:
            logger.info("Начинаем кросс-валидацию ensemble-моделей")
            classical_results = self.cross_validate_classical(X, y, classical_configs)
            results.update(classical_results)

        # Нейросетевые модели
        if neural_configs:
            logger.info("Начинаем кросс-валидацию нейросетевых моделей")
            neural_results = self.cross_validate_neural(X, y, neural_configs, neural_epochs)
            results.update(neural_results)

        return results


def calculate_cv_stats(cv_results: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, float]]:
    """Вычисляет статистики кросс-валидации.

    Args:
        cv_results: Результаты кросс-валидации.

    Returns:
        Словарь со статистиками для каждой модели.
    """
    stats = {}

    for model_name, metrics in cv_results.items():
        model_stats: dict[str, float] = {}

        for metric_name, values in metrics.items():
            # Фильтруем NaN значения
            clean_values = [v for v in values if not np.isnan(v)]

            if clean_values:
                model_stats[f"{metric_name}_mean"] = float(np.mean(clean_values))
                model_stats[f"{metric_name}_std"] = float(np.std(clean_values))
                model_stats[f"{metric_name}_min"] = float(np.min(clean_values))
                model_stats[f"{metric_name}_max"] = float(np.max(clean_values))
                model_stats[f"{metric_name}_cv"] = (
                    float(np.std(clean_values) / np.mean(clean_values)) if np.mean(clean_values) != 0 else 0.0
                )
            else:
                model_stats[f"{metric_name}_mean"] = float("nan")
                model_stats[f"{metric_name}_std"] = float("nan")
                model_stats[f"{metric_name}_min"] = float("nan")
                model_stats[f"{metric_name}_max"] = float("nan")
                model_stats[f"{metric_name}_cv"] = float("nan")

        stats[model_name] = model_stats

    return stats


def run_cross_validation(
    X: pl.DataFrame,
    y: pl.Series,
    model_types: list[str] | None = None,
    cv_folds: int = 5,
    neural_epochs: int = 50,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """Запускает кросс-валидацию для указанных типов моделей.

    Args:
        X: Признаки.
        y: Целевая переменная.
        model_types: Список типов моделей для валидации.
        cv_folds: Количество фолдов.
        neural_epochs: Количество эпох для нейросетей.
        random_state: Seed для воспроизводимости.

    Returns:
        Словарь со статистиками кросс-валидации.
    """
    cv = CrossValidation(n_splits=cv_folds, random_state=random_state)

    # Определяем конфигурации на основе запрошенных типов моделей
    classical_configs = {}
    neural_configs = {}

    if model_types is None:
        model_types = ["random_forest", "xgboost", "mlp"]

    for model_type in model_types:
        if model_type == "random_forest":
            classical_configs["random_forest"] = {"type": "random_forest", "n_estimators": 100, "max_depth": 10}
        elif model_type == "xgboost":
            classical_configs["xgboost"] = {"type": "xgboost", "n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
        elif model_type == "lightgbm":
            classical_configs["lightgbm"] = {"type": "lightgbm", "n_estimators": 100, "max_depth": -1, "learning_rate": 0.1}
        elif model_type == "mlp":
            neural_configs["mlp"] = {"type": "mlp", "hidden_sizes": [512, 256, 128], "dropout_rate": 0.3}
        elif model_type == "cnn":
            neural_configs["cnn"] = {"type": "cnn", "num_filters": 64, "kernel_sizes": [3, 5, 7], "dropout_rate": 0.3}
        else:
            logger.warning(f"Неизвестный тип модели: {model_type}")

    # Проводим кросс-валидацию
    cv_results = cv.cross_validate_all(
        X,
        y,
        classical_configs=classical_configs if classical_configs else None,
        neural_configs=neural_configs if neural_configs else None,
        neural_epochs=neural_epochs,
    )

    # Вычисляем статистики
    cv_stats = calculate_cv_stats(cv_results)

    logger.info(f"Кросс-валидация завершена для {len(cv_stats)} моделей")

    return cv_stats
