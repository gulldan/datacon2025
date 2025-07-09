"""Модуль для фильтрации молекулярных признаков.

Этот модуль содержит функции для фильтрации дескрипторов и фингерпринтов
по критериям качества: пропущенные значения, дисперсия, корреляция.
"""

import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def filter_descriptors(
    descriptors_df: pl.DataFrame,
    name_prefix: str = "",
    null_threshold: float = 0.1,
    variance_threshold: float = 0.001,
    correlation_threshold: float = 0.7,
) -> pl.DataFrame:
    """Фильтрация дескрипторов согласно критериям задания.

    Args:
        descriptors_df: Polars DataFrame с дескрипторами для фильтрации.
        name_prefix: Префикс для логгирования.
        null_threshold: Максимальная доля null значений (по умолчанию 10%).
        variance_threshold: Минимальная дисперсия.
        correlation_threshold: Максимальная корреляция между признаками.

    Returns:
        Polars DataFrame с отфильтрованными дескрипторами.

    Raises:
        ValueError: Если DataFrame пуст или не содержит численных колонок.
    """
    if descriptors_df.shape[0] == 0:
        raise ValueError("DataFrame не может быть пустым")

    logger.info(f"Фильтрация дескрипторов {name_prefix}...")
    logger.info(f"Исходное количество дескрипторов: {descriptors_df.shape[1]}")

    # 1. Удаляем дескрипторы с большим количеством null значений
    null_counts = descriptors_df.null_count()
    total_rows = descriptors_df.shape[0]

    # Получаем имена колонок с допустимым количеством null значений
    good_columns = []
    for col in descriptors_df.columns:
        null_ratio = null_counts.select(pl.col(col)).item() / total_rows
        if null_ratio <= null_threshold:
            good_columns.append(col)

    filtered_df = descriptors_df.select(good_columns)
    removed_null = descriptors_df.shape[1] - len(good_columns)
    logger.info(f"Удалено дескрипторов с >{null_threshold * 100}% null: {removed_null}")

    # 2. Заполняем оставшиеся null медианными значениями
    for col in filtered_df.columns:
        if filtered_df.select(pl.col(col).is_null().any()).item():
            median_val = filtered_df.select(pl.col(col).median()).item()
            filtered_df = filtered_df.with_columns(pl.col(col).fill_null(median_val))

    # 3. Удаляем дескрипторы с нулевой/низкой дисперсией
    variance_stats = filtered_df.select([pl.col(col).var().alias(f"{col}_var") for col in filtered_df.columns])

    # Выбираем колонки с достаточной дисперсией
    high_variance_cols = []
    for col in filtered_df.columns:
        var_value = variance_stats.select(pl.col(f"{col}_var")).item()
        if var_value is not None and var_value > variance_threshold:
            high_variance_cols.append(col)

    variance_filtered_df = filtered_df.select(high_variance_cols)
    removed_variance = filtered_df.shape[1] - variance_filtered_df.shape[1]
    logger.info(f"Удалено дескрипторов с низкой дисперсией (<{variance_threshold}): {removed_variance}")

    # 4. Удаляем высококоррелированные признаки
    if variance_filtered_df.shape[1] > 1:
        try:
            # Конвертируем в numpy для расчета корреляции
            numpy_data = variance_filtered_df.to_numpy()

            # Рассчитываем корреляционную матрицу
            correlation_matrix = np.corrcoef(numpy_data.T)

            # Находим пары с высокой корреляцией
            high_corr_features = set()
            n_features = len(variance_filtered_df.columns)

            for i in range(n_features):
                for j in range(i + 1, n_features):
                    corr_val = correlation_matrix[i, j]
                    if not np.isnan(corr_val) and abs(corr_val) > correlation_threshold:
                        # Удаляем второй признак из пары
                        high_corr_features.add(variance_filtered_df.columns[j])

            # Удаляем высококоррелированные признаки
            final_features = [col for col in variance_filtered_df.columns if col not in high_corr_features]
            final_df = variance_filtered_df.select(final_features)

            removed_corr = len(high_corr_features)
            logger.info(f"Удалено высококоррелированных дескрипторов (r>{correlation_threshold}): {removed_corr}")

        except Exception as e:
            logger.warning(f"Не удалось рассчитать корреляцию: {e}, пропускаем фильтрацию по корреляции")
            final_df = variance_filtered_df
    else:
        final_df = variance_filtered_df

    logger.info(f"Финальное количество дескрипторов {name_prefix}: {final_df.shape[1]}")

    return final_df


def filter_fingerprints(fingerprints_df: pl.DataFrame, name_prefix: str = "", variance_threshold: float = 0.01) -> pl.DataFrame:
    """Фильтрация фингерпринтов (удаляем биты, которые всегда 0 или всегда 1).

    Args:
        fingerprints_df: Polars DataFrame с фингерпринтами для фильтрации.
        name_prefix: Префикс для логгирования.
        variance_threshold: Минимальная дисперсия для бинарных данных.

    Returns:
        Polars DataFrame с отфильтрованными фингерпринтами.

    Raises:
        ValueError: Если DataFrame пуст.
    """
    if fingerprints_df.shape[0] == 0:
        raise ValueError("DataFrame не может быть пустым")

    # Для фингерпринтов используем только фильтр по дисперсии
    informative_cols = []
    for col in fingerprints_df.columns:
        variance = fingerprints_df.select(pl.col(col).var()).item()
        if variance is not None and variance > variance_threshold:
            informative_cols.append(col)

    filtered_df = fingerprints_df.select(informative_cols)
    removed_bits = fingerprints_df.shape[1] - filtered_df.shape[1]

    logger.info(f"{name_prefix}: удалено {removed_bits} неинформативных битов, осталось {filtered_df.shape[1]}")

    return filtered_df
