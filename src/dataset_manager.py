"""Модуль для управления датасетами.

Этот модуль содержит функции для создания различных комбинаций датасетов,
их сохранения и создания отчётов.
"""

import datetime
import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def create_dataset_combinations(
    base_data: pl.DataFrame,
    rdkit_filtered: pl.DataFrame,
    mordred_filtered: pl.DataFrame | None,
    filtered_fingerprints: dict[str, pl.DataFrame],
    padel_filtered: pl.DataFrame | None = None,
) -> dict[str, pl.DataFrame]:
    """Создаёт различные комбинации датасетов для мини-таска 3.

    Args:
        base_data: Базовые данные (SMILES, активность).
        rdkit_filtered: Отфильтрованные RDKit дескрипторы.
        mordred_filtered: Отфильтрованные Mordred дескрипторы (если доступны).
        filtered_fingerprints: Словарь с отфильтрованными фингерпринтами.
        padel_filtered: Отфильтрованные PaDEL дескрипторы (если доступны).

    Returns:
        Словарь с различными комбинациями датасетов.

    Raises:
        ValueError: Если базовые данные пусты.
    """
    if base_data.shape[0] == 0:
        raise ValueError("Базовые данные не могут быть пустыми")

    datasets_to_save = {}

    # 1. Датасет только с RDKit дескрипторами
    rdkit_with_prefix = rdkit_filtered.rename({col: f"rdkit_{col}" for col in rdkit_filtered.columns})
    rdkit_dataset = pl.concat([base_data, rdkit_with_prefix], how="horizontal")
    datasets_to_save["rdkit_descriptors"] = rdkit_dataset
    logger.info(f"Успешно: RDKit датасет: {rdkit_dataset.shape}")

    # 2. Датасет с Mordred дескрипторами (если доступны)
    if mordred_filtered is not None:
        mordred_with_prefix = mordred_filtered.rename({col: f"mordred_{col}" for col in mordred_filtered.columns})
        mordred_dataset = pl.concat([base_data, mordred_with_prefix], how="horizontal")
        datasets_to_save["mordred_descriptors"] = mordred_dataset
        logger.info(f"Успешно: Mordred датасет: {mordred_dataset.shape}")

    # 2a. Датасет с PaDEL дескрипторами (если доступны)
    if padel_filtered is not None:
        padel_with_prefix = padel_filtered.rename({col: f"padel_{col}" for col in padel_filtered.columns})
        padel_dataset = pl.concat([base_data, padel_with_prefix], how="horizontal")
        datasets_to_save["padel_descriptors"] = padel_dataset
        logger.info(f"Успешно: PaDEL датасет: {padel_dataset.shape}")

    # 3. Датасеты с фингерпринтами
    for fp_type, fp_df in filtered_fingerprints.items():
        fp_with_prefix = fp_df.rename({col: f"{fp_type}_{col}" for col in fp_df.columns})
        fp_dataset = pl.concat([base_data, fp_with_prefix], how="horizontal")
        datasets_to_save[f"fingerprints_{fp_type}"] = fp_dataset
        logger.info(f"Успешно: {fp_type} датасет: {fp_dataset.shape}")

        # 4. Комбинированный датасет (RDKit + лучший фингерпринт)
    # Выберем Morgan 1024 как компромисс между информативностью и размером
    if "morgan_1024" in filtered_fingerprints:
        # Добавляем префиксы для избежания дубликатов колонок
        rdkit_prefixed = rdkit_filtered.rename({col: f"rdkit_{col}" for col in rdkit_filtered.columns})
        morgan_prefixed = filtered_fingerprints["morgan_1024"].rename(
            {col: f"morgan_1024_{col}" for col in filtered_fingerprints["morgan_1024"].columns}
        )

        combined_dataset = pl.concat([base_data, rdkit_prefixed, morgan_prefixed], how="horizontal")
        datasets_to_save["combined_rdkit_morgan"] = combined_dataset
        logger.info(f"Успешно: Комбинированный датасет (RDKit + Morgan): {combined_dataset.shape}")

    # 5. Создаём лучший комбинированный датасет с всеми доступными дескрипторами
    if "morgan_1024" in filtered_fingerprints:
        components_to_combine = [base_data]

        # Отбираем топ дескрипторы из каждого типа для оптимального баланса
        if rdkit_filtered.shape[1] >= 50:
            rdkit_top50 = rdkit_filtered.select(rdkit_filtered.columns[:50])
            rdkit_top50_prefixed = rdkit_top50.rename({col: f"rdkit_{col}" for col in rdkit_top50.columns})
            components_to_combine.append(rdkit_top50_prefixed)
        else:
            rdkit_prefixed = rdkit_filtered.rename({col: f"rdkit_{col}" for col in rdkit_filtered.columns})
            components_to_combine.append(rdkit_prefixed)

        # Добавляем Mordred если доступен
        if mordred_filtered is not None:
            if mordred_filtered.shape[1] >= 100:
                mordred_top100 = mordred_filtered.select(mordred_filtered.columns[:100])
                mordred_top100_prefixed = mordred_top100.rename({col: f"mordred_{col}" for col in mordred_top100.columns})
            else:
                mordred_top100_prefixed = mordred_filtered.rename({col: f"mordred_{col}" for col in mordred_filtered.columns})
            components_to_combine.append(mordred_top100_prefixed)

        # Добавляем PaDEL если доступен
        if padel_filtered is not None:
            if padel_filtered.shape[1] >= 150:
                padel_top150 = padel_filtered.select(padel_filtered.columns[:150])
                padel_top150_prefixed = padel_top150.rename({col: f"padel_{col}" for col in padel_top150.columns})
            else:
                padel_top150_prefixed = padel_filtered.rename({col: f"padel_{col}" for col in padel_filtered.columns})
            components_to_combine.append(padel_top150_prefixed)

        # Добавляем Morgan фингерпринты
        morgan_prefixed = filtered_fingerprints["morgan_1024"].rename(
            {col: f"morgan_1024_{col}" for col in filtered_fingerprints["morgan_1024"].columns}
        )
        components_to_combine.append(morgan_prefixed)

        best_combined = pl.concat(components_to_combine, how="horizontal")
        datasets_to_save["best_combined"] = best_combined
        logger.info(f"Успешно: Лучший комбинированный датасет: {best_combined.shape}")

    return datasets_to_save


def save_datasets(datasets: dict[str, pl.DataFrame], output_dir: Path) -> None:
    """Сохраняет датасеты в CSV и Parquet форматах.

    Args:
        datasets: Словарь с датасетами для сохранения.
        output_dir: Директория для сохранения файлов.

    Raises:
        OSError: Если не удалось создать директорию или сохранить файлы.
    """
    try:
        output_dir.mkdir(exist_ok=True)
    except OSError as e:
        logger.error(f"Не удалось создать директорию {output_dir}: {e}")
        raise

    for name, dataset in datasets.items():
        try:
            # Сохраняем в CSV и Parquet
            csv_path = output_dir / f"cox2_{name}.csv"
            parquet_path = output_dir / f"cox2_{name}.parquet"

            dataset.write_csv(csv_path)
            dataset.write_parquet(parquet_path)

            logger.info(f"Сохранено: Сохранён датасет {name}: {dataset.shape}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении датасета {name}: {e}")
            raise


def create_report(
    datasets: dict[str, pl.DataFrame],
    smiles_list: list[str],
    rdkit_descriptors: pl.DataFrame,
    rdkit_filtered: pl.DataFrame,
    mordred_descriptors: pl.DataFrame | None,
    mordred_filtered: pl.DataFrame | None,
    fingerprint_dfs: dict[str, pl.DataFrame],
    filtered_fingerprints: dict[str, pl.DataFrame],
    output_path: Path,
    padel_descriptors: pl.DataFrame | None = None,
    padel_filtered: pl.DataFrame | None = None,
) -> None:
    """Создаёт сводный отчёт по расчёту дескрипторов.

    Args:
        datasets: Словарь с созданными датасетами.
        smiles_list: Исходный список SMILES.
        rdkit_descriptors: Исходные RDKit дескрипторы.
        rdkit_filtered: Отфильтрованные RDKit дескрипторы.
        mordred_descriptors: Исходные Mordred дескрипторы.
        mordred_filtered: Отфильтрованные Mordred дескрипторы.
        fingerprint_dfs: Исходные фингерпринты.
        filtered_fingerprints: Отфильтрованные фингерпринты.
        output_path: Путь для сохранения отчёта.
        padel_descriptors: Исходные PaDEL дескрипторы.
        padel_filtered: Отфильтрованные PaDEL дескрипторы.

    Raises:
        OSError: Если не удалось сохранить отчёт.
    """
    report_lines = [
        "# Отчёт по расчёту дескрипторов COX-2",
        f"Дата: {datetime.datetime.now()}",
        f"Исходное количество молекул: {len(smiles_list)}",
        "",
        "## Рассчитанные дескрипторы:",
        f"- RDKit: {rdkit_descriptors.shape[1]} → {rdkit_filtered.shape[1]} (после фильтрации)",
    ]

    if mordred_descriptors is not None and mordred_filtered is not None:
        report_lines.append(f"- Mordred: {mordred_descriptors.shape[1]} → {mordred_filtered.shape[1]} (после фильтрации)")

    if padel_descriptors is not None and padel_filtered is not None:
        report_lines.append(f"- PaDEL: {padel_descriptors.shape[1]} → {padel_filtered.shape[1]} (после фильтрации)")

    report_lines.extend(["", "## Фингерпринты:"])

    for fp_type, fp_df in fingerprint_dfs.items():
        filtered_fp = filtered_fingerprints[fp_type]
        report_lines.append(f"- {fp_type}: {fp_df.shape[1]} → {filtered_fp.shape[1]} битов (после фильтрации)")

    report_lines.extend(["", "## Созданные датасеты:"])

    for name, dataset in datasets.items():
        report_lines.append(f"- {name}: {dataset.shape[0]} молекул × {dataset.shape[1]} признаков")

    # Сохраняем отчёт
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        logger.info(f"Отчет: Отчёт сохранён: {output_path}")
    except OSError as e:
        logger.error(f"Не удалось сохранить отчёт {output_path}: {e}")
        raise


def get_dataset_statistics(datasets: dict[str, pl.DataFrame]) -> dict[str, dict[str, int]]:
    """Получает статистику по созданным датасетам.

    Args:
        datasets: Словарь с датасетами.

    Returns:
        Словарь со статистикой для каждого датасета.
    """
    statistics = {}

    for name, dataset in datasets.items():
        statistics[name] = {
            "molecules": dataset.shape[0],
            "features": dataset.shape[1],
            "null_values": dataset.null_count().sum_horizontal().item(),
        }

    return statistics
