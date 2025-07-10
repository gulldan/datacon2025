"""Модуль для сравнения и визуализации результатов моделей.

Содержит функции для создания интерактивных визуализаций
результатов обучения и кросс-валидации с использованием plotly.
"""

import logging

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def create_performance_comparison(
    results: dict[str, dict[str, float]],
    metrics: list[str] | None = None,
    title: str = "Сравнение производительности моделей",
) -> go.Figure:
    """Создает график сравнения производительности моделей.

    Args:
        results: Словарь с результатами моделей.
        metrics: Список метрик для сравнения.
        title: Заголовок графика.

    Returns:
        Plotly figure с графиком сравнения.
    """
    # Подготавливаем данные
    if metrics is None:
        metrics = ["mae", "rmse", "r2"]
    models = list(results.keys())

    # Создаем подграфики
    fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=metrics, shared_yaxes=False)

    colors = px.colors.qualitative.Set1[: len(models)]

    for i, metric in enumerate(metrics):
        values = [results[model].get(metric, 0) for model in models]

        fig.add_trace(
            go.Bar(
                x=models,
                y=values,
                name=metric,
                marker_color=colors,
                showlegend=(i == 0),
                text=[f"{v:.4f}" for v in values],
                textposition="auto",
            ),
            row=1,
            col=i + 1,
        )

        # Настраиваем оси
        fig.update_xaxes(title_text="Модель", row=1, col=i + 1)
        fig.update_yaxes(title_text=metric.upper(), row=1, col=i + 1)

    fig.update_layout(title=title, height=400, showlegend=False)

    return fig


def create_cv_stability_plot(
    cv_results: dict[str, dict[str, list[float]]], metric: str = "mae", title: str = "Стабильность моделей (кросс-валидация)"
) -> go.Figure:
    """Создает график стабильности моделей на основе кросс-валидации.

    Args:
        cv_results: Результаты кросс-валидации.
        metric: Метрика для анализа.
        title: Заголовок графика.

    Returns:
        Plotly figure с графиком стабильности.
    """
    models = []
    means = []
    stds = []
    values_lists = []

    for model_name, metrics in cv_results.items():
        if metric in metrics:
            # Фильтруем NaN значения
            clean_values = [v for v in metrics[metric] if not np.isnan(v)]
            if clean_values:
                models.append(model_name)
                means.append(np.mean(clean_values))
                stds.append(np.std(clean_values))
                values_lists.append(clean_values)

    fig = go.Figure()

    # Добавляем error bars
    fig.add_trace(
        go.Scatter(
            x=models,
            y=means,
            error_y={"type": "data", "array": stds, "visible": True},
            mode="markers+lines",
            marker={"size": 10},
            name=f"{metric.upper()} (среднее ± std)",
            line={"width": 2},
        )
    )

    # Добавляем точки для каждого фолда
    colors = px.colors.qualitative.Pastel1
    for i, values in enumerate(values_lists):
        for j, value in enumerate(values):
            fig.add_trace(
                go.Scatter(
                    x=[models[i]],
                    y=[value],
                    mode="markers",
                    marker={"size": 6, "color": colors[j % len(colors)], "opacity": 0.7},
                    showlegend=(i == 0),
                    name=f"Фолд {j + 1}" if i == 0 else "",
                    legendgroup=f"fold_{j}",
                )
            )

    fig.update_layout(title=title, xaxis_title="Модель", yaxis_title=f"{metric.upper()}", height=500)

    return fig


def create_training_time_comparison(results: dict[str, dict[str, float]], title: str = "Сравнение времени обучения") -> go.Figure:
    """Создает график сравнения времени обучения моделей.

    Args:
        results: Словарь с результатами моделей.
        title: Заголовок графика.

    Returns:
        Plotly figure с графиком времени обучения.
    """
    models = list(results.keys())
    times = [results[model].get("training_time", 0) for model in models]

    # Создаем цветовую схему
    colors = ["#1f77b4" if "Linear" in model or "Elastic" in model else "#ff7f0e" for model in models]

    fig = go.Figure(data=[go.Bar(x=models, y=times, marker_color=colors, text=[f"{t:.2f}s" for t in times], textposition="auto")])

    fig.update_layout(title=title, xaxis_title="Модель", yaxis_title="Время обучения (сек)", height=400)

    return fig


def create_prediction_scatter(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, title: str | None = None) -> go.Figure:
    """Создает scatter plot предсказаний vs истинных значений.

    Args:
        y_true: Истинные значения.
        y_pred: Предсказанные значения.
        model_name: Название модели.
        title: Заголовок графика.

    Returns:
        Plotly figure со scatter plot.
    """
    if title is None:
        title = f"Предсказания vs Истинные значения ({model_name})"

    # Вычисляем метрики
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    # Линия идеального предсказания
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    fig = go.Figure()

    # Scatter plot
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            marker={"size": 4, "opacity": 0.6, "color": "blue"},
            name="Предсказания",
            text=[f"True: {t:.2f}<br>Pred: {p:.2f}" for t, p in zip(y_true, y_pred, strict=False)],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # Линия идеального предсказания
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line={"dash": "dash", "color": "red", "width": 2},
            name="Идеальное предсказание",
        )
    )

    fig.update_layout(
        title=f"{title}<br><sub>MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}</sub>",
        xaxis_title="Истинные значения",
        yaxis_title="Предсказанные значения",
        height=500,
        width=500,
    )

    return fig


def create_combined_comparison_dashboard(
    test_results: dict[str, dict[str, float]],
    cv_results: dict[str, dict[str, list[float]]],
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> go.Figure:
    """Создает комбинированный dashboard для сравнения моделей.

    Args:
        test_results: Результаты на тестовой выборке.
        cv_results: Результаты кросс-валидации.
        predictions: Словарь с предсказаниями (y_true, y_pred) для каждой модели.

    Returns:
        Plotly figure с комбинированным dashboard.
    """
    # Создаем подграфики
    if predictions:
        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=[
                "MAE (тест vs CV)",
                "RMSE (тест vs CV)",
                "R² (тест vs CV)",
                "Время обучения",
                "Стабильность CV",
                "Предсказания vs Истинные",
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "scatter"}],
            ],
        )
    else:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=["MAE (тест vs CV)", "RMSE (тест vs CV)", "Время обучения", "Стабильность CV"],
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "scatter"}]],
        )

    models = list(test_results.keys())
    colors = px.colors.qualitative.Set1[: len(models)]

    # 1. MAE сравнение
    test_mae = [test_results[model].get("mae", 0) for model in models]
    cv_mae_mean = []
    cv_mae_std = []

    for model in models:
        if model in cv_results and "mae" in cv_results[model]:
            clean_values = [v for v in cv_results[model]["mae"] if not np.isnan(v)]
            cv_mae_mean.append(np.mean(clean_values) if clean_values else 0)
            cv_mae_std.append(np.std(clean_values) if clean_values else 0)
        else:
            cv_mae_mean.append(0)
            cv_mae_std.append(0)

    fig.add_trace(go.Bar(x=models, y=test_mae, name="Тест MAE", marker_color="lightblue"), row=1, col=1)
    fig.add_trace(
        go.Bar(x=models, y=cv_mae_mean, name="CV MAE", marker_color="darkblue", error_y={"type": "data", "array": cv_mae_std}),
        row=1,
        col=1,
    )

    # 2. RMSE сравнение
    test_rmse = [test_results[model].get("rmse", 0) for model in models]
    cv_rmse_mean = []
    cv_rmse_std = []

    for model in models:
        if model in cv_results and "rmse" in cv_results[model]:
            clean_values = [v for v in cv_results[model]["rmse"] if not np.isnan(v)]
            cv_rmse_mean.append(np.mean(clean_values) if clean_values else 0)
            cv_rmse_std.append(np.std(clean_values) if clean_values else 0)
        else:
            cv_rmse_mean.append(0)
            cv_rmse_std.append(0)

    fig.add_trace(go.Bar(x=models, y=test_rmse, name="Тест RMSE", marker_color="lightcoral", showlegend=False), row=1, col=2)
    fig.add_trace(
        go.Bar(
            x=models,
            y=cv_rmse_mean,
            name="CV RMSE",
            marker_color="darkred",
            error_y={"type": "data", "array": cv_rmse_std},
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 3. R² сравнение
    test_r2 = [test_results[model].get("r2", 0) for model in models]
    cv_r2_mean = []
    cv_r2_std = []

    for model in models:
        if model in cv_results and "r2" in cv_results[model]:
            clean_values = [v for v in cv_results[model]["r2"] if not np.isnan(v)]
            cv_r2_mean.append(np.mean(clean_values) if clean_values else 0)
            cv_r2_std.append(np.std(clean_values) if clean_values else 0)
        else:
            cv_r2_mean.append(0)
            cv_r2_std.append(0)

    fig.add_trace(go.Bar(x=models, y=test_r2, name="Тест R²", marker_color="lightgreen", showlegend=False), row=1, col=3)
    fig.add_trace(
        go.Bar(
            x=models,
            y=cv_r2_mean,
            name="CV R²",
            marker_color="darkgreen",
            error_y={"type": "data", "array": cv_r2_std},
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    # 4. Время обучения
    times = [test_results[model].get("training_time", 0) for model in models]
    fig.add_trace(go.Bar(x=models, y=times, name="Время обучения", marker_color=colors, showlegend=False), row=2, col=1)

    # 5. Стабильность CV (коэффициент вариации MAE)
    cv_stability = []
    for model in models:
        if model in cv_results and "mae" in cv_results[model]:
            clean_values = [v for v in cv_results[model]["mae"] if not np.isnan(v)]
            if clean_values and np.mean(clean_values) != 0:
                cv_stability.append(np.std(clean_values) / np.mean(clean_values))
            else:
                cv_stability.append(0)
        else:
            cv_stability.append(0)

    fig.add_trace(
        go.Scatter(
            x=models, y=cv_stability, mode="markers+lines", marker={"size": 10}, name="CV коэф. вариации", showlegend=False
        ),
        row=2,
        col=2,
    )

    # 6. Предсказания (если есть)
    if predictions:
        # Берем первую модель для примера
        first_model = next(iter(predictions.keys()))
        y_true, y_pred = predictions[first_model]

        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode="markers",
                marker={"size": 4, "opacity": 0.6},
                name=f"Предсказания {first_model}",
                showlegend=False,
            ),
            row=2,
            col=3,
        )

        # Линия идеального предсказания
        min_val, max_val = y_true.min(), y_true.max()
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line={"dash": "dash", "color": "red"},
                name="Идеальное",
                showlegend=False,
            ),
            row=2,
            col=3,
        )

    # Обновляем layout
    fig.update_layout(title="Dashboard сравнения моделей", height=800, showlegend=True)

    # Настраиваем оси
    fig.update_xaxes(title_text="Модель", row=1, col=1)
    fig.update_xaxes(title_text="Модель", row=1, col=2)
    fig.update_xaxes(title_text="Модель", row=1, col=3)
    fig.update_xaxes(title_text="Модель", row=2, col=1)
    fig.update_xaxes(title_text="Модель", row=2, col=2)

    fig.update_yaxes(title_text="MAE", row=1, col=1)
    fig.update_yaxes(title_text="RMSE", row=1, col=2)
    fig.update_yaxes(title_text="R²", row=1, col=3)
    fig.update_yaxes(title_text="Время (сек)", row=2, col=1)
    fig.update_yaxes(title_text="Коэф. вариации", row=2, col=2)

    if predictions:
        fig.update_xaxes(title_text="Истинные значения", row=2, col=3)
        fig.update_yaxes(title_text="Предсказания", row=2, col=3)

    return fig


def create_model_ranking(
    test_results: dict[str, dict[str, float]],
    cv_results: dict[str, dict[str, list[float]]],
    weights: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Создает рейтинг моделей на основе взвешенных метрик.

    Args:
        test_results: Результаты на тестовой выборке.
        cv_results: Результаты кросс-валидации.
        weights: Веса для разных метрик.

    Returns:
        DataFrame с рейтингом моделей.
    """
    if weights is None:
        weights = {"mae": 0.4, "rmse": 0.3, "r2": 0.2, "stability": 0.1}
    ranking_data = []

    # Нормализуем метрики для сравнения
    all_mae = [test_results[model].get("mae", float("inf")) for model in test_results]
    all_rmse = [test_results[model].get("rmse", float("inf")) for model in test_results]
    all_r2 = [test_results[model].get("r2", 0) for model in test_results]

    max_mae = max(all_mae) if all_mae else 1
    max_rmse = max(all_rmse) if all_rmse else 1
    max_r2 = max(all_r2) if all_r2 else 1

    for model_name in test_results:
        # Нормализованные метрики (чем меньше MAE/RMSE, тем лучше; чем больше R², тем лучше)
        norm_mae = 1 - (test_results[model_name].get("mae", float("inf")) / max_mae)
        norm_rmse = 1 - (test_results[model_name].get("rmse", float("inf")) / max_rmse)
        norm_r2 = test_results[model_name].get("r2", 0) / max_r2 if max_r2 > 0 else 0

        # Стабильность (меньше коэффициент вариации = выше стабильность)
        stability = 1.0
        if model_name in cv_results and "mae" in cv_results[model_name]:
            clean_values = [v for v in cv_results[model_name]["mae"] if not np.isnan(v)]
            if clean_values and np.mean(clean_values) != 0:
                cv_coef = np.std(clean_values) / np.mean(clean_values)
                stability = 1 / (1 + cv_coef)  # Чем меньше CV, тем выше стабильность

        # Взвешенная оценка
        score = (
            norm_mae * weights.get("mae", 0.4)
            + norm_rmse * weights.get("rmse", 0.3)
            + norm_r2 * weights.get("r2", 0.2)
            + stability * weights.get("stability", 0.1)
        )

        ranking_data.append(
            {
                "model": model_name,
                "score": score,
                "mae": test_results[model_name].get("mae", float("inf")),
                "rmse": test_results[model_name].get("rmse", float("inf")),
                "r2": test_results[model_name].get("r2", 0),
                "stability": stability,
                "training_time": test_results[model_name].get("training_time", 0),
            }
        )

    # Сортируем по убыванию score
    ranking_df = pl.DataFrame(ranking_data).sort("score", descending=True)

    logger.info(f"Рейтинг моделей создан, лучшая модель: {ranking_df.row(0)[0]}")

    return ranking_df


def save_visualizations(figures: dict[str, go.Figure], output_dir: str = "data/visualizations/") -> None:
    """Сохраняет визуализации в HTML файлы.

    Args:
        figures: Словарь с фигурами plotly.
        output_dir: Директория для сохранения.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    for name, fig in figures.items():
        filepath = os.path.join(output_dir, f"{name}.html")
        fig.write_html(filepath)
        logger.info(f"Сохранена визуализация: {filepath}")
