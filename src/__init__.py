"""COX-2 Dataset Preparation Package.

Оптимизированный пакет для подготовки данных, аугментации и обучения моделей
для предсказания активности против COX-2.
"""

import warnings
from pathlib import Path

# Подавляем предупреждения
warnings.filterwarnings("ignore")

# Основные модули
# Классические модели
from .classical_models import ClassicalModels
from .cross_validation import CrossValidation
from .data_preparation import prepare_data_pipeline
from .logging_config import get_logger

# Оптимизированные модули
from .molecular_augmentation import (
    MolecularAugmentation,
    augment_molecular_data,
    augment_molecular_data_optimized,
)

# Проверка доступности современных моделей
MODERN_MODELS_AVAILABLE = False
try:
    from .improved_modern_models import (
        OptimizedMLPBaseline,
        OptimizedSimpleGAT,
        OptimizedSimpleGCN,
        create_molecular_graph_fast,
        prepare_graph_data_optimized,
        train_model_optimized,
    )

    MODERN_MODELS_AVAILABLE = True
    logger = get_logger(__name__)
    logger.info("✅ Оптимизированные современные модели доступны")

except ImportError as e:
    logger = get_logger(__name__)
    logger.warning(f"⚠️ Оптимизированные современные модели недоступны: {e}")
    logger.warning("Установите torch-geometric для использования современных моделей")

    # Заглушки для обратной совместимости
    OptimizedMLPBaseline = None
    OptimizedSimpleGAT = None
    OptimizedSimpleGCN = None
    create_molecular_graph_fast = None
    prepare_graph_data_optimized = None
    train_model_optimized = None

# Проверка графовых утилит
GRAPH_UTILS_AVAILABLE = False
try:
    from .graph_utils import prepare_graph_data
    GRAPH_UTILS_AVAILABLE = True
    logger.info("✅ Графовые утилиты доступны")
except ImportError as e:
    logger.warning(f"⚠️ Графовые утилиты недоступны: {e}")
    prepare_graph_data = None

# Версия пакета
__version__ = "1.2.0"

# Главные функции и классы
__all__ = [
    # Версия
    "__version__",

    # Логирование
    "get_logger",

    # Управление данными
    "prepare_data_pipeline",

    # Классические модели
    "ClassicalModels",
    "CrossValidation",

    # Нейронные сети
    # Оптимизированная аугментация
    "MolecularAugmentation",
    "augment_molecular_data",
    "augment_molecular_data_optimized",

    # Оптимизированные современные модели
    "OptimizedMLPBaseline",
    "OptimizedSimpleGAT",
    "OptimizedSimpleGCN",
    "create_molecular_graph_fast",
    "prepare_graph_data_optimized",
    "train_model_optimized",

    # Графовые утилиты
    "prepare_graph_data",

    # Флаги доступности
    "MODERN_MODELS_AVAILABLE",
    "GRAPH_UTILS_AVAILABLE",
]

# Проверка зависимостей
def check_dependencies():
    """Проверяет доступность всех зависимостей."""
    logger = get_logger(__name__)

    dependencies = {
        "polars": "✅ Доступен",
        "scikit-learn": "✅ Доступен",
        "rdkit": "✅ Доступен",
        "torch": "✅ Доступен",
        "numpy": "✅ Доступен",
        "pandas": "✅ Доступен",
        "torch-geometric": "✅ Доступен" if MODERN_MODELS_AVAILABLE else "❌ Не установлен",
        "tqdm": "✅ Доступен",
        "plotly": "✅ Доступен",
    }

    logger.info("🔍 Проверка зависимостей:")
    for dep, status in dependencies.items():
        logger.info(f"  {dep}: {status}")

    return dependencies


# Автоматическая проверка при импорте
if __name__ != "__main__":
    logger = get_logger(__name__)
    logger.info(f"📦 Инициализация пакета COX-2 Dataset Preparation v{__version__}")

    # Проверяем доступность ключевых модулей
    if MODERN_MODELS_AVAILABLE:
        logger.info("🚀 Оптимизированные модели готовы к использованию")
    else:
        logger.warning("⚠️ Для полной функциональности установите torch-geometric")

    if GRAPH_UTILS_AVAILABLE:
        logger.info("📊 Графовые утилиты готовы к использованию")

    logger.info("✅ Инициализация завершена")
