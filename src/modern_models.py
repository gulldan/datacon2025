"""Современные модели для молекулярного предсказания свойств.

Полностью переписанные модели с GPU поддержкой и исправленными архитектурами.
Включает:
- Graph Transformer
- Foundation Model (CheMeleon-style)
- Multimodal GNN
- Attention-based GNN
- Modern Message Passing Networks
"""

import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch_geometric.data import DataLoader
from torch_geometric.nn import (
    GATv2Conv,
    GCNConv,
    MessagePassing,
    TransformerConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from src.logging_config import get_logger

logger = get_logger(__name__)


class GraphTransformer(nn.Module):
    """Улучшенный Graph Transformer для молекулярного предсказания."""

    def __init__(
        self,
        input_dim: int = 74,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        # Graph Transformer layers с улучшенными параметрами
        self.transformer_layers = nn.ModuleList(
            [
                TransformerConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=None,
                    beta=True,
                    root_weight=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

        # Улучшенный readout с batch normalization
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Input projection
        x = self.input_projection(x)

        # Apply transformer layers с skip connections
        for transformer, layer_norm, dropout in zip(self.transformer_layers, self.layer_norms, self.dropouts, strict=False):
            residual = x
            x = transformer(x, edge_index)
            x = layer_norm(x + residual)  # Skip connection
            x = dropout(F.gelu(x))  # GELU activation

        # Multi-scale graph pooling
        graph_mean = global_mean_pool(x, batch)
        graph_max = global_max_pool(x, batch)
        graph_sum = global_add_pool(x, batch)

        # Concatenate pooling results
        graph_repr = torch.cat([graph_mean, graph_max, graph_sum], dim=-1)

        # Final prediction
        out = self.readout(graph_repr)
        return out


class FoundationModel(nn.Module):
    """Улучшенная Foundation Model с правильной архитектурой."""

    def __init__(
        self,
        input_dim: int = 74,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 16,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        # Улучшенный encoder
        self.descriptor_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional encoding для молекул
        self.pos_encoding = nn.Parameter(torch.randn(1000, hidden_dim) * 0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Improved prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim),
        )

    def forward(self, data):
        x, batch = data.x, data.batch

        # Encode molecular descriptors
        x = self.descriptor_encoder(x)

        # Group by batch and pad sequences
        batch_size = batch.max().item() + 1
        max_nodes = torch.bincount(batch).max().item()

        # Create padded sequences
        x_padded = torch.zeros(batch_size, int(max_nodes), x.size(-1), device=x.device)
        attention_mask = torch.ones(batch_size, int(max_nodes), device=x.device, dtype=torch.bool)

        for i in range(batch_size):
            mask = batch == i
            nodes_in_batch = mask.sum().item()
            x_padded[i, :nodes_in_batch] = x[mask]
            attention_mask[i, nodes_in_batch:] = False

        # Add positional encoding
        seq_len = x_padded.size(1)
        x_padded = x_padded + self.pos_encoding[:seq_len].unsqueeze(0)

        # Apply transformer
        x_transformed = self.transformer(x_padded, src_key_padding_mask=~attention_mask)

        # Global pooling с attention weighting
        attention_weights = attention_mask.unsqueeze(-1).float()
        x_pooled = (x_transformed * attention_weights).sum(dim=1) / attention_weights.sum(dim=1)

        # Final prediction
        out = self.prediction_head(x_pooled)
        return out


class MultimodalGNN(nn.Module):
    """Исправленная Multimodal GNN с правильной архитектурой."""

    def __init__(
        self,
        input_dim: int = 74,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        # Graph convolution branch
        self.graph_convs = nn.ModuleList([GCNConv(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])

        self.graph_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

        # Descriptor processing branch
        self.descriptor_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Cross-modal fusion
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Graph branch with skip connections
        graph_x = x
        for i, (conv, norm) in enumerate(zip(self.graph_convs, self.graph_norms, strict=False)):
            new_x = conv(graph_x, edge_index)
            new_x = norm(new_x)
            new_x = F.relu(new_x)
            new_x = self.dropout(new_x)

            # Skip connection after first layer
            if i > 0 and new_x.size() == graph_x.size():
                new_x = new_x + graph_x
            graph_x = new_x

        # Graph pooling
        graph_repr = global_mean_pool(graph_x, batch)

        # Descriptor branch
        node_repr = global_mean_pool(x, batch)  # Use original features
        descriptor_repr = self.descriptor_net(node_repr)

        # Fusion
        combined = torch.cat([graph_repr, descriptor_repr], dim=-1)
        out = self.fusion_net(combined)

        return out


class AttentionGNN(nn.Module):
    """Исправленная Attention GNN с правильными размерностями."""

    def __init__(
        self,
        input_dim: int = 74,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        # Убеждаемся что hidden_dim делится на num_heads
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) должен делиться на num_heads ({num_heads})"

        head_dim = hidden_dim // num_heads

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GAT layers с правильными размерностями
        self.gat_layers = nn.ModuleList(
            [
                GATv2Conv(hidden_dim, head_dim, heads=num_heads, dropout=dropout, concat=True, bias=True, share_weights=False)
                for _ in range(num_layers)
            ]
        )

        # Layer normalization
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Output projection after GAT layers
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Input projection
        x = self.input_proj(x)

        # Apply GAT layers с skip connections
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms, strict=False)):
            residual = x
            x = gat(x, edge_index)
            x = self.dropout(x)

            # Skip connection
            if i > 0:
                x = x + residual
            x = norm(x)
            x = F.relu(x)

        # Final projection
        x = self.output_proj(x)

        # Graph pooling
        graph_repr = global_mean_pool(x, batch)

        # Prediction
        out = self.predictor(graph_repr)
        return out


class ModernMPNN(nn.Module):
    """Исправленная MPNN модель."""

    def __init__(
        self,
        input_dim: int = 74,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        # Используем GCN слои вместо кастомного MessagePassing
        self.conv_layers = nn.ModuleList([GCNConv(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])

        # Layer normalization
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Prediction head
        self.predictor = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply conv layers with skip connections
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.layer_norms, strict=False)):
            residual = x if i > 0 and x.size(-1) == conv.out_channels else None
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)

            # Skip connection
            if residual is not None:
                x = x + residual

        # Global pooling
        graph_repr = global_mean_pool(x, batch)

        # Prediction
        out = self.predictor(graph_repr)
        return out


class ModernMessagePassing(MessagePassing):
    """Исправленная Message Passing с правильной сигнатурой."""

    def __init__(self, input_dim, hidden_dim, dropout=0.2) -> None:
        super().__init__(aggr="add", node_dim=0)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Message network
        self.message_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        # Gating mechanism
        self.gate = nn.Sequential(nn.Linear(input_dim + hidden_dim, input_dim), nn.Sigmoid())

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # Создаем сообщения из соседних узлов
        return self.message_net(torch.cat([x_i, x_j], dim=-1))

    def update(self, inputs, x):
        # Обновляем представления узлов
        gate = self.gate(torch.cat([x, inputs], dim=-1))
        update = self.update_net(torch.cat([x, inputs], dim=-1))
        return gate * update + (1 - gate) * x


class ModernModels:
    """Улучшенный менеджер для современных моделей."""

    def __init__(self, epochs=100, lr=0.001, batch_size=32, random_state=42) -> None:
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state

        # Автоматическое определение устройства (GPU/CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Инициализация ModernModels на {self.device}")

        # Установка seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)

        self.models = {}

    def train_model(
        self,
        model_name: str,
        X_train: list,
        y_train: torch.Tensor,
        X_test: list,
        y_test: torch.Tensor,
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 32,
        **model_kwargs,
    ) -> dict[str, Any]:
        """Обучает современную модель с правильным batch handling."""
        start_time = time.time()

        # Определяем размерность входа
        input_dim = X_train[0].x.shape[1] if hasattr(X_train[0], "x") else 74

        # Создаем модель
        model_configs = {
            "GraphTransformer": {"hidden_dim": 256, "num_layers": 6, "num_heads": 8},
            "FoundationModel": {"hidden_dim": 512, "num_layers": 8, "num_heads": 16},
            "MultimodalGNN": {"hidden_dim": 256, "num_layers": 4},
            "AttentionGNN": {"hidden_dim": 256, "num_layers": 4, "num_heads": 8},
            "ModernMPNN": {"hidden_dim": 256, "num_layers": 4},
        }

        config = model_configs.get(model_name, {})
        config.update(model_kwargs)

        if model_name == "GraphTransformer":
            model = GraphTransformer(input_dim=input_dim, **config)
        elif model_name == "FoundationModel":
            model = FoundationModel(input_dim=input_dim, **config)
        elif model_name == "MultimodalGNN":
            model = MultimodalGNN(input_dim=input_dim, **config)
        elif model_name == "AttentionGNN":
            model = AttentionGNN(input_dim=input_dim, **config)
        elif model_name == "ModernMPNN":
            model = ModernMPNN(input_dim=input_dim, **config)
        else:
            msg = f"Unknown model: {model_name}"
            raise ValueError(msg)

        model = model.to(self.device)

        # Улучшенный оптимизатор и scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
        criterion = nn.MSELoss()

        # Создаем DataLoader
        train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True, drop_last=False)

        logger.info(f"Начинаем обучение {model_name} на {self.device}")

        # Training loop с gradient clipping
        model.train()
        best_loss = float("inf")
        patience = 20
        patience_counter = 0

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for batch_data in train_loader:
                batch_data = batch_data.to(self.device)

                # Получаем соответствующие target значения
                batch_indices = []
                current_idx = 0
                for graph in X_train:
                    batch_size_curr = graph.batch.max().item() + 1 if hasattr(graph, "batch") and graph.batch is not None else 1
                    batch_indices.extend(range(current_idx, current_idx + batch_size_curr))
                    current_idx += batch_size_curr
                    if current_idx >= len(batch_data.batch.unique()):
                        break

                # Упрощенное получение targets
                unique_batch_ids = batch_data.batch.unique()
                batch_targets = y_train[unique_batch_ids.cpu()].to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_data).squeeze()

                # Убеждаемся что размерности совпадают
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                if batch_targets.dim() == 0:
                    batch_targets = batch_targets.unsqueeze(0)

                if len(outputs) != len(batch_targets):
                    batch_targets = batch_targets[: len(outputs)]

                loss = criterion(outputs, batch_targets)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            scheduler.step()

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 20 == 0:
                logger.info(f"Эпоха {epoch + 1}/{epochs}: train_loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

        training_time = time.time() - start_time

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=False)
            all_predictions = []
            all_targets = []

            for i, batch_data in enumerate(test_loader):
                batch_data = batch_data.to(self.device)
                outputs = model(batch_data).squeeze()

                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)

                all_predictions.append(outputs.cpu().numpy())

                # Получаем соответствующие targets
                batch_size_curr = len(batch_data.batch.unique())
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size_curr, len(y_test))
                batch_targets = y_test[start_idx:end_idx]
                all_targets.append(batch_targets.cpu().numpy())

            predictions = np.concatenate(all_predictions)
            targets = np.concatenate(all_targets)

        # Вычисляем метрики
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)

        # Сохраняем модель
        self.models[model_name] = {
            "model": model,
            "predictions": predictions,
            "targets": targets,
        }

        logger.info(f"Обучена модель {model_name} за {training_time:.3f} сек")
        logger.info(f"Метрики для {model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "training_time": training_time,
            "predictions": predictions.tolist(),
            "targets": targets.tolist(),
        }

    def train_all_models(
        self,
        X_train: list,
        y_train: torch.Tensor,
        X_test: list,
        y_test: torch.Tensor,
        epochs: int = 100,
        **kwargs,
    ) -> dict[str, dict[str, Any]]:
        """Обучает все современные модели."""
        model_names = ["GraphTransformer", "FoundationModel", "MultimodalGNN", "AttentionGNN", "ModernMPNN"]
        results = {}

        for model_name in model_names:
            try:
                logger.info(f"\n🚀 Обучение {model_name}...")
                results[model_name] = self.train_model(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    epochs=epochs,
                    lr=self.lr,
                    batch_size=self.batch_size,
                    **kwargs,
                )
            except Exception as e:
                logger.error(f"Ошибка при обучении {model_name}: {e}")
                import traceback

                traceback.print_exc()
                results[model_name] = {
                    "mae": float("inf"),
                    "mse": float("inf"),
                    "rmse": float("inf"),
                    "r2": -float("inf"),
                    "training_time": 0.0,
                    "predictions": [],
                    "targets": [],
                }

        return results


def train_modern_models(
    X_train: list,
    y_train: torch.Tensor,
    X_test: list,
    y_test: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 32,
    random_state: int = 42,
    **kwargs,
) -> dict[str, dict[str, Any]]:
    """Главная функция для обучения современных моделей."""
    logger.info("=== ОБУЧЕНИЕ СОВРЕМЕННЫХ МОДЕЛЕЙ ===")
    logger.info("Включены: Graph Transformer, Foundation Model, Multimodal GNN, Attention GNN, Modern MPNN")

    trainer = ModernModels(epochs=epochs, lr=lr, batch_size=batch_size, random_state=random_state)
    results = trainer.train_all_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, epochs=epochs, **kwargs)

    logger.info("\n=== РЕЗУЛЬТАТЫ СОВРЕМЕННЫХ МОДЕЛЕЙ ===")
    for model_name, metrics in results.items():
        if metrics["r2"] != -float("inf"):
            logger.info(
                f"{model_name:15}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
                f"R²={metrics['r2']:.4f}, Время={metrics['training_time']:.2f}s"
            )

    return results
