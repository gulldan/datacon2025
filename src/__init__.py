"""Пакет для анализа молекулярной активности и машинного обучения."""

# Импорты из существующих модулей (task 1-2)
from .classical_models import (
    ClassicalModels,
    train_classical_models,
)
from .cross_validation import (
    CrossValidation,
    calculate_cv_stats,
    run_cross_validation,
)

# Импорты для task 3 (машинное обучение)
from .data_preparation import (
    DataPreparation,
    prepare_data_pipeline,
)
from .descriptor_calculator import (
    calculate_fingerprints,
    calculate_mordred_descriptors,
    calculate_padel_descriptors,
    calculate_rdkit_descriptors,
    clear_cache,
    get_cache_info,
    get_optimal_threads,
    save_descriptors_to_data,
)
from .model_comparison import (
    create_combined_comparison_dashboard,
    create_cv_stability_plot,
    create_model_ranking,
    create_performance_comparison,
    create_prediction_scatter,
    create_training_time_comparison,
    save_visualizations,
)
from .neural_models import (
    CNNRegressor,
    MLPRegressor,
    NeuralNetworkTrainer,
    train_neural_models,
)

__all__ = [
    # Task 1-2 функции
    "calculate_fingerprints",
    "calculate_mordred_descriptors",
    "calculate_padel_descriptors",
    "calculate_rdkit_descriptors",
    "clear_cache",
    "get_cache_info",
    "get_optimal_threads",
    "save_descriptors_to_data",
    # Task 3 классы и функции
    "DataPreparation",
    "prepare_data_pipeline",
    "ClassicalModels",
    "train_classical_models",
    "MLPRegressor",
    "CNNRegressor",
    "NeuralNetworkTrainer",
    "train_neural_models",
    "CrossValidation",
    "calculate_cv_stats",
    "run_cross_validation",
    "create_performance_comparison",
    "create_cv_stability_plot",
    "create_training_time_comparison",
    "create_prediction_scatter",
    "create_combined_comparison_dashboard",
    "create_model_ranking",
    "save_visualizations",
]
