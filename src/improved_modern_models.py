"""Оптимизированная версия современных моделей для молекулярного предсказания.

Улучшенная производительность с GPU, многопоточностью и эффективными DataLoader'ами.
"""

import time
import warnings
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as GeometricDataLoader
from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool
from tqdm import tqdm

from .logging_config import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


class MolecularDataset(Dataset):
    """Оптимизированный датасет для молекулярных данных."""

    def __init__(self, graphs: list[Data], targets: list[float]) -> None:
        self.graphs = graphs
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.targets[idx]


def create_molecular_graph_fast(smiles: str) -> Data | None:
    """Быстрое создание молекулярного графа с кешированием."""
    try:
        mol = Chem.MolFromSmiles(smiles)  # type: ignore
        if mol is None:
            return None

        # Упрощенные атомные признаки для скорости
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
            ]
            atom_features.append(features)

        # Простые связи
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.extend([(i, j), (j, i)])

        if not edge_indices:
            num_atoms = len(atom_features)
            edge_indices = [(i, i) for i in range(num_atoms)]

        # Создаем граф
        x = torch.tensor(atom_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index, num_nodes=x.size(0))

    except Exception:
        return None


def prepare_graph_data_optimized(
    df: pl.DataFrame,
    smiles_column: str = "canonical_smiles",
    target_column: str = "pic50",
    train_indices: list[int] | None = None,
    test_indices: list[int] | None = None,
    max_samples: int = 2000,  # Ограничиваем для скорости
) -> tuple[list[Data], list[float], list[Data], list[float]]:
    """Оптимизированная подготовка графовых данных."""
    logger.info("Оптимизированная подготовка графовых данных...")

    # Ограничиваем размер для тестирования
    if len(df) > max_samples:
        logger.info(f"Ограничиваем размер до {max_samples} образцов")
        df = df.head(max_samples)

    smiles_list = df[smiles_column].to_list()
    targets = df[target_column].to_list()

    # Создаем графы с прогресс-баром
    graphs = []
    valid_indices = []

    logger.info(f"Создание графов для {len(smiles_list)} молекул...")

    for i, smiles in enumerate(tqdm(smiles_list, desc="Creating graphs")):
        graph = create_molecular_graph_fast(smiles)
        if graph is not None:
            graphs.append(graph)
            valid_indices.append(i)

    # Фильтруем целевые значения
    valid_targets = [targets[i] for i in valid_indices]

    logger.info(f"Создано {len(graphs)} валидных графов из {len(smiles_list)}")

    # Простое разделение если индексы не заданы
    if train_indices is None or test_indices is None:
        n_train = int(0.8 * len(graphs))
        train_graphs = graphs[:n_train]
        train_targets = valid_targets[:n_train]
        test_graphs = graphs[n_train:]
        test_targets = valid_targets[n_train:]
    else:
        # Используем предоставленные индексы
        train_graphs = [graphs[i] for i in train_indices if i < len(graphs)]
        train_targets = [valid_targets[i] for i in train_indices if i < len(valid_targets)]
        test_graphs = [graphs[i] for i in test_indices if i < len(graphs)]
        test_targets = [valid_targets[i] for i in test_indices if i < len(valid_targets)]

    logger.info(f"Разделение: {len(train_graphs)} train, {len(test_graphs)} test")

    return train_graphs, train_targets, test_graphs, test_targets


class OptimizedSimpleGCN(nn.Module):
    """Оптимизированная простая GCN модель."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()

        self.conv_layers = nn.ModuleList([GCNConv(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])

        self.dropout = nn.Dropout(0.3)
        self.predictor = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)
        return self.predictor(x).squeeze()


class OptimizedSimpleGAT(nn.Module):
    """Оптимизированная простая GAT модель."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, num_layers: int = 2, heads: int = 4) -> None:
        super().__init__()

        self.gat_layers = nn.ModuleList(
            [
                GATv2Conv(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim // heads if i < num_layers - 1 else hidden_dim,
                    heads=heads if i < num_layers - 1 else 1,
                    dropout=0.3,
                    concat=i < num_layers - 1,
                )
                for i in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(0.3)
        self.predictor = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for gat in self.gat_layers:
            x = gat(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)
        return self.predictor(x).squeeze()


class OptimizedMLPBaseline(nn.Module):
    """Оптимизированная MLP baseline модель."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 128, num_layers: int = 3) -> None:
        super().__init__()

        layers = []
        prev_dim = input_dim

        for _i in range(num_layers):
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, data):
        x, batch = data.x, data.batch
        x = global_mean_pool(x, batch)
        return self.network(x).squeeze()


def train_model_optimized(
    model: nn.Module,
    train_graphs: list[Data],
    train_targets: list[float],
    test_graphs: list[Data],
    test_targets: list[float],
    epochs: int = 25,  # Уменьшили количество эпох
    lr: float = 0.001,
    batch_size: int = 32,  # Увеличили batch size
    device: str = "auto",
    random_state: int = 42,
) -> dict[str, Any]:
    """Оптимизированная функция обучения модели."""
    # Автоматическое определение устройства
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Используется устройство: {device}")
    logger.info(f"Batch size: {batch_size}, Epochs: {epochs}")

    # Нормализация целевых значений
    target_scaler = StandardScaler()
    train_targets_2d = target_scaler.fit_transform(np.array(train_targets).reshape(-1, 1))
    test_targets_2d = target_scaler.transform(np.array(test_targets).reshape(-1, 1))

    # Преобразуем в 1D массивы
    train_targets_scaled = np.asarray(train_targets_2d).ravel()
    test_targets_scaled = np.asarray(test_targets_2d).ravel()

    # Создание DataLoader'ов
    train_loader = GeometricDataLoader(
        list(zip(train_graphs, train_targets_scaled, strict=False)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # Многопоточность
        pin_memory=device == "cuda",
    )

    test_loader = GeometricDataLoader(
        list(zip(test_graphs, test_targets_scaled, strict=False)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device == "cuda",
    )

    # Перемещение модели на устройство
    model = model.to(device)

    # Оптимизатор и планировщик
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    # Обучение
    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_graphs, batch_targets in train_loader:
            batch_graphs = batch_graphs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            predictions = model(batch_graphs)
            loss = criterion(predictions, batch_targets)

            # Проверка на NaN
            if torch.isnan(loss):
                logger.warning("NaN loss detected, skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        scheduler.step(avg_loss)

        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    training_time = time.time() - start_time
    logger.info(f"Обучение завершено за {training_time:.2f} секунд")

    # Оценка
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_graphs, batch_targets in test_loader:
            batch_graphs = batch_graphs.to(device)
            batch_targets = batch_targets.to(device)

            predictions = model(batch_graphs)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_targets.cpu().numpy())

    # Обратное масштабирование
    predictions_original = target_scaler.inverse_transform(np.array(all_predictions).reshape(-1, 1))
    targets_original = target_scaler.inverse_transform(np.array(all_targets).reshape(-1, 1))

    # Преобразуем в 1D массивы
    predictions_original = np.asarray(predictions_original).ravel()
    targets_original = np.asarray(targets_original).ravel()

    # Метрики
    from sklearn.metrics import mean_absolute_error, r2_score

    mae = mean_absolute_error(targets_original, predictions_original)
    r2 = r2_score(targets_original, predictions_original)

    results = {
        "model_name": model.__class__.__name__,
        "r2": r2,
        "mae": mae,
        "training_time": training_time,
        "device": device,
        "batch_size": batch_size,
        "epochs": epochs,
    }

    logger.info(f"Результаты: R² = {r2:.4f}, MAE = {mae:.4f}")

    return results
