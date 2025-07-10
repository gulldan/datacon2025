"""Модуль для визуализации дескрипторов.

Этот модуль содержит функции для создания интерактивных графиков
для анализа молекулярных дескрипторов и фингерпринтов.
"""

import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def create_descriptors_visualization(
    datasets: dict[str, pl.DataFrame],
    original_descriptors: dict[str, pl.DataFrame],
    filtered_descriptors: dict[str, pl.DataFrame],
    output_dir: Path,
) -> None:
    """Создаёт комплексную визуализацию дескрипторов и фингерпринтов.

    Args:
        datasets: Словарь с финальными датасетами.
        original_descriptors: Словарь с исходными дескрипторами.
        filtered_descriptors: Словарь с отфильтрованными дескрипторами.
        output_dir: Директория для сохранения графиков.

    Raises:
        OSError: Если не удалось создать директорию или сохранить файлы.
    """
    try:
        output_dir.mkdir(exist_ok=True)
    except OSError as e:
        logger.error(f"Не удалось создать директорию {output_dir}: {e}")
        raise

    # 1. Диаграмма количества признаков до и после фильтрации
    create_feature_counts_chart(original_descriptors, filtered_descriptors, output_dir)

    # 2. Анализ пропущенных значений
    create_missing_values_analysis(original_descriptors, output_dir)

    # 3. Распределение размеров датасетов
    create_dataset_sizes_chart(datasets, output_dir)

    # 4. Анализ фингерпринтов
    create_fingerprint_analysis(filtered_descriptors, output_dir)

    logger.info(f"Статистика: Визуализации сохранены в {output_dir}")


def create_feature_counts_chart(original: dict[str, pl.DataFrame], filtered: dict[str, pl.DataFrame], output_dir: Path) -> None:
    """Создаёт диаграмму количества признаков до и после фильтрации.

    Args:
        original: Исходные дескрипторы.
        filtered: Отфильтрованные дескрипторы.
        output_dir: Директория для сохранения.
    """
    # Подготавливаем данные для графика
    feature_types = []
    original_counts = []
    filtered_counts = []

    for key in original:
        if key in filtered:
            feature_types.append(key)
            original_counts.append(original[key].shape[1])
            filtered_counts.append(filtered[key].shape[1])

    fig = go.Figure(
        data=[
            go.Bar(
                name="Исходное количество",
                x=feature_types,
                y=original_counts,
                marker_color="lightblue",
                text=original_counts,
                textposition="outside",
            ),
            go.Bar(
                name="После фильтрации",
                x=feature_types,
                y=filtered_counts,
                marker_color="darkblue",
                text=filtered_counts,
                textposition="outside",
            ),
        ]
    )

    fig.update_layout(
        title="Количество признаков до и после фильтрации",
        xaxis_title="Тип признаков",
        yaxis_title="Количество признаков",
        barmode="group",
        height=500,
        font={"size": 12},
    )

    # Сохраняем график
    fig.write_html(output_dir / "feature_counts.html")
    logger.info("Успешно: График количества признаков сохранён")


def create_missing_values_analysis(original: dict[str, pl.DataFrame], output_dir: Path) -> None:
    """Создаёт анализ пропущенных значений в исходных дескрипторах.

    Args:
        original: Исходные дескрипторы.
        output_dir: Директория для сохранения.
    """
    # Создаём subplot с несколькими графиками
    fig = make_subplots(
        rows=len(original), cols=1, subplot_titles=[f"Пропущенные значения: {key}" for key in original], vertical_spacing=0.1
    )

    for i, (key, df) in enumerate(original.items(), 1):
        # Считаем процент пропущенных значений для каждой колонки
        null_percentages = []
        column_names = []

        for col in df.columns:
            null_count = df.select(pl.col(col).is_null().sum()).item()
            null_percentage = (null_count / df.shape[0]) * 100
            null_percentages.append(null_percentage)
            column_names.append(col)

        # Создаём гистограмму распределения null значений
        fig.add_trace(go.Histogram(x=null_percentages, name=f"{key}", nbinsx=20, showlegend=False), row=i, col=1)

    fig.update_layout(
        title="Распределение пропущенных значений по типам дескрипторов", height=300 * len(original), font={"size": 10}
    )

    fig.update_xaxes(title_text="Процент пропущенных значений")
    fig.update_yaxes(title_text="Количество признаков")

    fig.write_html(output_dir / "missing_values_analysis.html")
    logger.info("Успешно: Анализ пропущенных значений сохранён")


def create_dataset_sizes_chart(datasets: dict[str, pl.DataFrame], output_dir: Path) -> None:
    """Создаёт график размеров финальных датасетов.

    Args:
        datasets: Финальные датасеты.
        output_dir: Директория для сохранения.
    """
    dataset_names = list(datasets.keys())
    molecules_counts = [df.shape[0] for df in datasets.values()]
    features_counts = [df.shape[1] for df in datasets.values()]

    # Создаём комбинированную диаграмму
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Количество молекул", "Количество признаков"],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]],
    )

    # График количества молекул
    fig.add_trace(
        go.Bar(
            x=dataset_names,
            y=molecules_counts,
            name="Молекулы",
            marker_color="lightgreen",
            text=molecules_counts,
            textposition="outside",
        ),
        row=1,
        col=1,
    )

    # График количества признаков
    fig.add_trace(
        go.Bar(
            x=dataset_names,
            y=features_counts,
            name="Признаки",
            marker_color="orange",
            text=features_counts,
            textposition="outside",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(title="Размеры финальных датасетов", height=500, showlegend=False, font={"size": 10})

    # Поворачиваем подписи по оси X для лучшей читаемости
    fig.update_xaxes(tickangle=45)

    fig.write_html(output_dir / "dataset_sizes.html")
    logger.info("Успешно: График размеров датасетов сохранён")


def create_fingerprint_analysis(filtered: dict[str, pl.DataFrame], output_dir: Path) -> None:
    """Создаёт анализ плотности фингерпринтов.

    Args:
        filtered: Отфильтрованные дескрипторы и фингерпринты.
        output_dir: Директория для сохранения.
    """
    # Фильтруем только фингерпринты
    fingerprint_data = {
        k: v for k, v in filtered.items() if "morgan" in k or "maccs" in k or "atompairs" in k or "topological" in k
    }

    if not fingerprint_data:
        logger.info("Фингерпринты не найдены для анализа")
        return

    fig = make_subplots(
        rows=len(fingerprint_data),
        cols=1,
        subplot_titles=[f"Плотность бит: {key}" for key in fingerprint_data],
        vertical_spacing=0.1,
    )

    for i, (key, df) in enumerate(fingerprint_data.items(), 1):
        # Считаем плотность (долю единиц) для каждой молекулы
        densities = []
        for row_idx in range(df.shape[0]):
            row_data = df.row(row_idx)
            density = sum(row_data) / len(row_data) * 100
            densities.append(density)

        fig.add_trace(go.Histogram(x=densities, name=key, nbinsx=30, showlegend=False), row=i, col=1)

    fig.update_layout(
        title="Распределение плотности битов в фингерпринтах", height=300 * len(fingerprint_data), font={"size": 10}
    )

    fig.update_xaxes(title_text="Плотность битов (%)")
    fig.update_yaxes(title_text="Количество молекул")

    fig.write_html(output_dir / "fingerprint_analysis.html")
    logger.info("Успешно: Анализ фингерпринтов сохранён")


def create_correlation_heatmap(df: pl.DataFrame, title: str, output_path: Path, max_features: int = 50) -> None:
    """Создаёт тепловую карту корреляций между признаками.

    Args:
        df: DataFrame с числовыми признаками.
        title: Заголовок графика.
        output_path: Путь для сохранения файла.
        max_features: Максимальное количество признаков для отображения.
    """
    # Выбираем подмножество признаков если их слишком много
    if df.shape[1] > max_features:
        selected_columns = df.columns[:max_features]
        df_subset = df.select(selected_columns)
        logger.info(f"Выбрано {max_features} признаков из {df.shape[1]} для корреляционной матрицы")
    else:
        df_subset = df

    # Рассчитываем корреляционную матрицу
    try:
        # Используем простой способ через numpy
        numeric_data = df_subset.to_numpy()
        corr_matrix = np.corrcoef(numeric_data.T)

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix,
                x=df_subset.columns,
                y=df_subset.columns,
                colorscale="RdBu",
                zmid=0,
                colorbar={"title": "Корреляция"},
            )
        )

        fig.update_layout(title=title, xaxis_title="Признаки", yaxis_title="Признаки", width=800, height=800)

        fig.write_html(output_path)
        logger.info(f"Успешно: Корреляционная матрица сохранена: {output_path}")

    except Exception as e:
        logger.warning(f"Не удалось создать корреляционную матрицу: {e}")


def create_summary_dashboard(datasets: dict[str, pl.DataFrame], statistics: dict[str, dict[str, int]], output_path: Path) -> None:
    """Создаёт сводную панель с основной статистикой.

    Args:
        datasets: Финальные датасеты.
        statistics: Статистика по датасетам.
        output_path: Путь для сохранения панели.
    """
    # Создаём таблицу со статистикой
    table_data = []
    for name, stats in statistics.items():
        table_data.append([name, stats["molecules"], stats["features"], stats["null_values"]])

    fig = go.Figure(
        data=[
            go.Table(
                header={
                    "values": ["Датасет", "Молекулы", "Признаки", "Null значения"],
                    "fill_color": "lightblue",
                    "align": "left",
                    "font": {"size": 12, "color": "black"},
                },
                cells={
                    "values": list(zip(*table_data, strict=False)),
                    "fill_color": "white",
                    "align": "left",
                    "font": {"size": 11},
                },
            )
        ]
    )

    fig.update_layout(title="Сводная статистика по датасетам COX-2", height=400, font={"size": 10})

    fig.write_html(output_path)
    logger.info(f"Успешно: Сводная панель сохранена: {output_path}")
