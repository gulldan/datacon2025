"""Модуль для аугментации молекулярных данных.

Оптимизированная версия с многопоточностью и улучшенной производительностью.
"""

import random
import warnings
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

import numpy as np
import polars as pl
import torch
from rdkit import Chem
from tqdm import tqdm

from .logging_config import get_logger

# Подавляем предупреждения RDKit
warnings.filterwarnings("ignore")

logger = get_logger(__name__)


def process_smiles_batch(smiles_batch: list[str], n_variants: int = 3) -> list[str]:
    """Обрабатывает батч SMILES для enumeration."""
    results = []

    for smiles in smiles_batch:
        results.append(smiles)  # Добавляем исходный

        try:
            mol = Chem.MolFromSmiles(smiles)  # type: ignore
            if mol is None:
                results.extend([smiles] * n_variants)
                continue

            variants = set()
            attempts = 0
            max_attempts = n_variants * 5  # Уменьшили количество попыток

            while len(variants) < n_variants and attempts < max_attempts:
                try:
                    new_smiles = Chem.MolToSmiles(mol, doRandom=True)  # type: ignore
                    if new_smiles != smiles and Chem.MolFromSmiles(new_smiles) is not None:  # type: ignore
                        variants.add(new_smiles)
                except Exception:
                    pass
                attempts += 1

            variants_list = list(variants)
            results.extend(variants_list)

            if len(variants_list) < n_variants:
                results.extend([smiles] * (n_variants - len(variants_list)))

        except Exception:
            results.extend([smiles] * n_variants)

    return results


class MolecularAugmentation:
    """Оптимизированный класс для аугментации молекулярных данных."""

    def __init__(self, random_state: int = 42, n_workers: int | None = None):
        """Инициализация с оптимизацией ресурсов."""
        self.random_state = random_state
        self.n_workers = n_workers or min(cpu_count(), 8)  # Ограничиваем количество процессов

        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)

        logger.info(f"Инициализация с {self.n_workers} workers")

    def smiles_enumeration(self, smiles_list: list[str], n_variants: int = 3) -> list[str]:
        """Оптимизированный SMILES enumeration с многопоточностью."""
        logger.info(f"SMILES enumeration для {len(smiles_list)} молекул с {self.n_workers} workers...")

        # Разбиваем на батчи
        batch_size = max(10, len(smiles_list) // (self.n_workers * 4))
        batches = [smiles_list[i:i + batch_size] for i in range(0, len(smiles_list), batch_size)]

        logger.info(f"Обработка {len(batches)} батчей размером ~{batch_size}")

        all_results = []

        # Используем ThreadPoolExecutor для I/O операций
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(process_smiles_batch, batch, n_variants) for batch in batches]

            for future in tqdm(futures, desc="Processing batches"):
                try:
                    batch_results = future.result(timeout=60)  # Таймаут 60 секунд
                    all_results.extend(batch_results)
                except Exception as e:
                    logger.warning(f"Batch processing failed: {e}")
                    # Fallback - обрабатываем последовательно
                    failed_batch = batches[futures.index(future)]
                    all_results.extend(process_smiles_batch(failed_batch, n_variants))

        logger.info(f"SMILES enumeration завершен: {len(smiles_list)} -> {len(all_results)}")
        return all_results

    def descriptor_noise_injection_vectorized(
        self, descriptors_df: pl.DataFrame, noise_factor: float = 0.05, n_variants: int = 2
    ) -> pl.DataFrame:
        """Векторизованное добавление шума к дескрипторам."""
        logger.info("Векторизованный descriptor noise injection...")

        # Находим числовые колонки
        numeric_columns = [
            col for col in descriptors_df.columns
            if descriptors_df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]

        if not numeric_columns:
            return descriptors_df

        logger.info(f"Обработка {len(numeric_columns)} числовых колонок")

        # Конвертируем в numpy для векторных операций
        numeric_data = descriptors_df.select(numeric_columns).to_numpy()

        # Заменяем NaN на медианы
        for i, col in enumerate(numeric_columns):
            col_data = numeric_data[:, i]
            mask = ~np.isnan(col_data)
            if mask.sum() > 0:
                median_val = np.median(col_data[mask])
                numeric_data[~mask, i] = median_val

        # Вычисляем статистики векторно
        means = np.nanmean(numeric_data, axis=0)
        stds = np.nanstd(numeric_data, axis=0)
        mins = np.nanmin(numeric_data, axis=0)
        maxs = np.nanmax(numeric_data, axis=0)

        # Создаем аугментированные данные
        augmented_data = []

        # Добавляем исходные данные
        augmented_data.append(numeric_data)

        # Добавляем варианты с шумом
        for _ in range(n_variants):
            # Генерируем шум векторно
            noise = np.random.normal(0, stds * noise_factor, numeric_data.shape)
            noisy_data = numeric_data + noise

            # Ограничиваем значения векторно
            noisy_data = np.clip(noisy_data, mins, maxs)
            augmented_data.append(noisy_data)

        # Объединяем все данные
        final_data = np.vstack(augmented_data)

        # Создаем DataFrame
        augmented_df = pl.DataFrame(final_data, schema=numeric_columns)

        # Добавляем нечисловые колонки
        non_numeric_columns = [col for col in descriptors_df.columns if col not in numeric_columns]
        if non_numeric_columns:
            for col in non_numeric_columns:
                col_values = descriptors_df[col].to_list()
                extended_values = col_values * (n_variants + 1)
                augmented_df = augmented_df.with_columns(
                    pl.Series(name=col, values=extended_values[:len(augmented_df)])
                )

        logger.info(f"Векторизованная аугментация завершена: {len(descriptors_df)} -> {len(augmented_df)}")
        return augmented_df

    def augment_dataset_optimized(
        self,
        df: pl.DataFrame,
        smiles_column: str = "canonical_smiles",
        target_column: str = "pic50",
        augmentation_factor: int = 2,
        techniques: list[str] | None = None,
    ) -> pl.DataFrame:
        """Оптимизированная аугментация датасета."""
        if techniques is None:
            techniques = ["smiles_enumeration"]  # По умолчанию только быстрые техники

        logger.info("Оптимизированная аугментация датасета...")
        logger.info(f"Исходный размер: {len(df)} образцов")
        logger.info(f"Техники: {techniques}")

        # Ограничиваем размер для тестирования
        if len(df) > 1000:
            logger.info("Ограничиваем размер датасета до 1000 образцов для оптимизации")
            df = df.head(1000)

        augmented_df = df.clone()

        if "smiles_enumeration" in techniques and smiles_column in df.columns:
            logger.info("Применяем оптимизированный SMILES enumeration...")

            original_smiles = df[smiles_column].to_list()
            augmented_smiles = self.smiles_enumeration(original_smiles, augmentation_factor)

            # Создаем соответствующие целевые значения
            original_targets = df[target_column].to_list()
            augmented_targets = []

            for target in original_targets:
                augmented_targets.extend([target] * (augmentation_factor + 1))

            # Обрезаем до нужной длины
            min_len = min(len(augmented_smiles), len(augmented_targets))
            augmented_smiles = augmented_smiles[:min_len]
            augmented_targets = augmented_targets[:min_len]

            augmented_df = pl.DataFrame({
                smiles_column: augmented_smiles,
                target_column: augmented_targets
            })

        if "descriptor_noise" in techniques:
            logger.info("Применяем векторизованный descriptor noise injection...")

            descriptor_columns = [
                col for col in augmented_df.columns
                if col not in [smiles_column, target_column]
            ]

            if descriptor_columns:
                descriptors_df = augmented_df.select(descriptor_columns)
                augmented_descriptors = self.descriptor_noise_injection_vectorized(
                    descriptors_df, noise_factor=0.03, n_variants=1
                )

                # Расширяем остальные колонки
                n_multiplier = len(augmented_descriptors) // len(df)

                if smiles_column in augmented_df.columns:
                    original_smiles = df[smiles_column].to_list()
                    extended_smiles = (original_smiles * n_multiplier)[:len(augmented_descriptors)]
                    augmented_descriptors = augmented_descriptors.with_columns(
                        pl.Series(name=smiles_column, values=extended_smiles)
                    )

                original_targets = df[target_column].to_list()
                extended_targets = (original_targets * n_multiplier)[:len(augmented_descriptors)]
                augmented_df = augmented_descriptors.with_columns(
                    pl.Series(name=target_column, values=extended_targets)
                )

        logger.info("Оптимизированная аугментация завершена")
        logger.info(f"Финальный размер: {len(augmented_df)} образцов")
        logger.info(f"Коэффициент увеличения: {len(augmented_df) / len(df):.2f}x")

        return augmented_df


def augment_molecular_data_optimized(
    df: pl.DataFrame,
    smiles_column: str = "canonical_smiles",
    target_column: str = "pic50",
    augmentation_factor: int = 2,
    techniques: list[str] | None = None,
    random_state: int = 42,
    n_workers: int | None = None,
) -> pl.DataFrame:
    """Оптимизированная функция для аугментации молекулярных данных."""
    augmenter = MolecularAugmentation(random_state=random_state, n_workers=n_workers)

    return augmenter.augment_dataset_optimized(
        df=df,
        smiles_column=smiles_column,
        target_column=target_column,
        augmentation_factor=augmentation_factor,
        techniques=techniques,
    )


# Обратная совместимость
augment_molecular_data = augment_molecular_data_optimized
