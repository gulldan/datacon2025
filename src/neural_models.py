"""Модуль для обучения нейросетевых моделей.

Содержит реализации MLP и CNN моделей с использованием PyTorch
для регрессионных задач предсказания биологической активности.
"""

import logging
import time
from typing import Any

import numpy as np
import polars as pl
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class MLPRegressor(nn.Module):
    """Многослойный перцептрон для регрессии."""

    def __init__(self, input_size: int, hidden_sizes: list[int] | None = None, dropout_rate: float = 0.3) -> None:
        """Инициализация MLP модели.

        Args:
            input_size: Размер входного слоя.
            hidden_sizes: Список размеров скрытых слоев.
            dropout_rate: Вероятность dropout.
        """
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]
        super().__init__()

        layers = []
        prev_size = input_size

        # Создаем скрытые слои
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(prev_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(), nn.Dropout(dropout_rate)])
            prev_size = hidden_size

        # Выходной слой
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход через сеть.

        Args:
            x: Входные данные.

        Returns:
            Предсказания модели.
        """
        return self.network(x).squeeze()


class CNNRegressor(nn.Module):
    """Сверточная нейронная сеть для регрессии."""

    def __init__(
        self, input_size: int, num_filters: int = 64, kernel_sizes: list[int] | None = None, dropout_rate: float = 0.3
    ) -> None:
        """Инициализация CNN модели.

        Args:
            input_size: Размер входного слоя.
            num_filters: Количество фильтров в каждом слое.
            kernel_sizes: Размеры ядер свертки.
            dropout_rate: Вероятность dropout.
        """
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]
        super().__init__()

        self.input_size = input_size

        # Сверточные слои с разными размерами ядер
        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, num_filters, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(num_filters),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(input_size // 4),
                )
                for k in kernel_sizes
            ]
        )

        # Полносвязные слои
        conv_output_size = num_filters * len(kernel_sizes) * (input_size // 4)

        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(conv_output_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход через сеть.

        Args:
            x: Входные данные.

        Returns:
            Предсказания модели.
        """
        # Добавляем размерность канала для Conv1d
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)

        # Применяем сверточные слои
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_out = conv_layer(x)
            conv_outputs.append(conv_out)

        # Объединяем выходы всех сверточных слоев
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(kernel_sizes), pooled_size)

        # Выпрямляем для полносвязных слоев
        x = x.view(x.size(0), -1)

        # Применяем полносвязные слои
        x = self.fc_layers(x)

        return x.squeeze()


class NeuralNetworkTrainer:
    """Класс для обучения нейронных сетей."""

    def __init__(self, random_state: int = 42, device: str | None = None) -> None:
        """Инициализация тренера.

        Args:
            random_state: Seed для воспроизводимости.
            device: Устройство для вычислений (cuda/cpu).
        """
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models: dict[str, dict[str, Any]] = {}
        self.training_times: dict[str, float] = {}

        # Устанавливаем seeds для воспроизводимости
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)

    def _prepare_data(self, X: pl.DataFrame, y: pl.Series, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """Подготавливает данные для обучения.

        Args:
            X: Признаки.
            y: Целевая переменная.
            batch_size: Размер батча.
            shuffle: Перемешивать ли данные.

        Returns:
            DataLoader с подготовленными данными.
        """
        # Конвертируем в numpy и затем в tensors
        X_array = X.to_numpy().astype(np.float32)
        y_array = y.to_numpy().astype(np.float32)

        X_tensor = torch.from_numpy(X_array)
        y_tensor = torch.from_numpy(y_array)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

    def train_mlp(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        X_val: pl.DataFrame | None = None,
        y_val: pl.Series | None = None,
        hidden_sizes: list[int] | None = None,
        dropout_rate: float = 0.3,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        patience: int = 15,
        model_name: str = "MLP",
    ) -> None:
        """Обучает MLP модель.

        Args:
            X_train: Обучающие признаки.
            y_train: Обучающая целевая переменная.
            X_val: Валидационные признаки.
            y_val: Валидационная целевая переменная.
            hidden_sizes: Размеры скрытых слоев.
            dropout_rate: Вероятность dropout.
            epochs: Количество эпох.
            batch_size: Размер батча.
            learning_rate: Скорость обучения.
            patience: Терпение для early stopping.
            model_name: Название модели.
        """
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]
        start_time = time.time()

        # Создаем модель
        input_size = X_train.shape[1]
        model = MLPRegressor(input_size, hidden_sizes, dropout_rate).to(self.device)

        # Оптимизатор и функция потерь
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # Подготавливаем данные
        train_loader = self._prepare_data(X_train, y_train, batch_size, shuffle=True)

        # Early stopping
        best_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        # История обучения
        train_losses = []
        val_losses = []

        logger.info(f"Начинаем обучение {model_name} на {self.device}")

        for epoch in range(epochs):
            # Обучение
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Валидация
            if X_val is not None and y_val is not None:
                model.eval()
                with torch.no_grad():
                    val_loader = self._prepare_data(X_val, y_val, batch_size, shuffle=False)
                    val_loss = 0.0

                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        predictions = model(batch_X)
                        val_loss += criterion(predictions, batch_y).item()

                    val_loss /= len(val_loader)
                    val_losses.append(val_loss)

                    scheduler.step(val_loss)

                    # Early stopping
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_counter = 0
                        best_model_state = model.state_dict().copy()
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        logger.info(f"Early stopping на эпохе {epoch + 1}")
                        break

            if (epoch + 1) % 10 == 0:
                if X_val is not None:
                    logger.info(f"Эпоха {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                else:
                    logger.info(f"Эпоха {epoch + 1}/{epochs}: train_loss={train_loss:.4f}")

        # Загружаем лучшую модель
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        training_time = time.time() - start_time

        # Сохраняем модель
        self.models[model_name] = {
            "type": "mlp",
            "model": model,
            "hidden_sizes": hidden_sizes,
            "dropout_rate": dropout_rate,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

        self.training_times[model_name] = training_time

        logger.info(f"Обучена модель {model_name} за {training_time:.3f} сек")

    def train_cnn(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        X_val: pl.DataFrame | None = None,
        y_val: pl.Series | None = None,
        num_filters: int = 64,
        kernel_sizes: list[int] | None = None,
        dropout_rate: float = 0.3,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        patience: int = 15,
        model_name: str = "CNN",
    ) -> None:
        """Обучает CNN модель.

        Args:
            X_train: Обучающие признаки.
            y_train: Обучающая целевая переменная.
            X_val: Валидационные признаки.
            y_val: Валидационная целевая переменная.
            num_filters: Количество фильтров.
            kernel_sizes: Размеры ядер свертки.
            dropout_rate: Вероятность dropout.
            epochs: Количество эпох.
            batch_size: Размер батча.
            learning_rate: Скорость обучения.
            patience: Терпение для early stopping.
            model_name: Название модели.
        """
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]
        start_time = time.time()

        # Создаем модель
        input_size = X_train.shape[1]
        model = CNNRegressor(input_size, num_filters, kernel_sizes, dropout_rate).to(self.device)

        # Оптимизатор и функция потерь
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # Подготавливаем данные
        train_loader = self._prepare_data(X_train, y_train, batch_size, shuffle=True)

        # Early stopping
        best_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        # История обучения
        train_losses = []
        val_losses = []

        logger.info(f"Начинаем обучение {model_name} на {self.device}")

        for epoch in range(epochs):
            # Обучение
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Валидация
            if X_val is not None and y_val is not None:
                model.eval()
                with torch.no_grad():
                    val_loader = self._prepare_data(X_val, y_val, batch_size, shuffle=False)
                    val_loss = 0.0

                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        predictions = model(batch_X)
                        val_loss += criterion(predictions, batch_y).item()

                    val_loss /= len(val_loader)
                    val_losses.append(val_loss)

                    scheduler.step(val_loss)

                    # Early stopping
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_counter = 0
                        best_model_state = model.state_dict().copy()
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        logger.info(f"Early stopping на эпохе {epoch + 1}")
                        break

            if (epoch + 1) % 10 == 0:
                if X_val is not None:
                    logger.info(f"Эпоха {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                else:
                    logger.info(f"Эпоха {epoch + 1}/{epochs}: train_loss={train_loss:.4f}")

        # Загружаем лучшую модель
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        training_time = time.time() - start_time

        # Сохраняем модель
        self.models[model_name] = {
            "type": "cnn",
            "model": model,
            "num_filters": num_filters,
            "kernel_sizes": kernel_sizes,
            "dropout_rate": dropout_rate,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

        self.training_times[model_name] = training_time

        logger.info(f"Обучена модель {model_name} за {training_time:.3f} сек")

    def predict(self, model_name: str, X_test: pl.DataFrame, batch_size: int = 32) -> np.ndarray:
        """Делает предсказания с помощью обученной модели.

        Args:
            model_name: Название модели.
            X_test: Тестовые признаки.
            batch_size: Размер батча.

        Returns:
            Массив предсказаний.
        """
        if model_name not in self.models:
            msg = f"Модель {model_name} не найдена"
            raise ValueError(msg)

        model = self.models[model_name]["model"]
        model.eval()

        # Подготавливаем данные (без shuffle)
        X_array = X_test.to_numpy().astype(np.float32)
        X_tensor = torch.from_numpy(X_array)

        predictions = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i : i + batch_size].to(self.device)
                pred = model(batch)
                predictions.append(pred.cpu().numpy())

        return np.concatenate(predictions)

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


def train_neural_models(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_test: pl.DataFrame,
    y_test: pl.Series,
    X_val: pl.DataFrame | None = None,
    y_val: pl.Series | None = None,
    epochs: int = 100,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """Обучает и оценивает набор нейросетевых моделей.

    Args:
        X_train: Обучающие признаки.
        y_train: Обучающая целевая переменная.
        X_test: Тестовые признаки.
        y_test: Тестовая целевая переменная.
        X_val: Валидационные признаки.
        y_val: Валидационная целевая переменная.
        epochs: Количество эпох обучения.
        random_state: Seed для воспроизводимости.

    Returns:
        Словарь с метриками для каждой модели.
    """
    trainer = NeuralNetworkTrainer(random_state=random_state)

    # Обучаем модели
    trainer.train_mlp(X_train, y_train, X_val, y_val, epochs=epochs, model_name="MLP")
    trainer.train_cnn(X_train, y_train, X_val, y_val, epochs=epochs, model_name="CNN")

    # Оцениваем модели
    results = {}
    for model_name in trainer.list_models():
        metrics = trainer.evaluate_model(model_name, X_test, y_test)
        results[model_name] = metrics

    return results
