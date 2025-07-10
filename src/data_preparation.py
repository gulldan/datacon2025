"""Модуль для подготовки данных для машинного обучения.

Содержит функции для нормализации, стандартизации, разделения данных
и предобработки признаков для моделей машинного обучения.
"""

import logging

import polars as pl
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


class DataPreparation:
    """Класс для подготовки данных к обучению моделей."""

    def __init__(self, random_state: int = 42) -> None:
        """Инициализация с установкой random state.

        Args:
            random_state: Seed для воспроизводимости результатов.
        """
        self.random_state = random_state
        self.scaler: StandardScaler | MinMaxScaler | None = None
        self.pca: PCA | None = None

    def load_data(self, file_path: str, target_column: str = "pIC50") -> pl.DataFrame:
        """Загружает данные из файла.

        Args:
            file_path: Путь к файлу с данными.
            target_column: Название колонки с целевой переменной.

        Returns:
            DataFrame с загруженными данными.
        """
        try:
            if file_path.endswith(".parquet"):
                df = pl.read_parquet(file_path)
            elif file_path.endswith(".csv"):
                df = pl.read_csv(file_path)
            else:
                msg = f"Неподдерживаемый формат файла: {file_path}"
                raise ValueError(msg)

            logger.info(f"Загружены данные: {df.shape[0]} строк, {df.shape[1]} колонок")
            return df

        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}")
            raise

    def clean_data(self, df: pl.DataFrame, target_column: str = "pIC50") -> pl.DataFrame:
        """Очищает данные от пропусков и некорректных значений.

        Args:
            df: Исходный DataFrame.
            target_column: Название колонки с целевой переменной.

        Returns:
            Очищенный DataFrame.
        """
        initial_rows = df.shape[0]

        # Удаляем строки с пропусками в целевой переменной
        df_clean = df.filter(pl.col(target_column).is_not_null())

        # Удаляем строки где все признаки равны null
        feature_columns = [col for col in df.columns if col not in [target_column, "smiles", "molecule_chembl_id"]]
        df_clean = df_clean.filter(
            pl.fold(
                acc=pl.lit(False), function=lambda acc, x: acc | x.is_not_null(), exprs=[pl.col(col) for col in feature_columns]
            )
        )

        cleaned_rows = df_clean.shape[0]
        logger.info(f"Очищено данных: {initial_rows} -> {cleaned_rows} строк ({initial_rows - cleaned_rows} удалено)")

        return df_clean

    def filter_features(self, df: pl.DataFrame, target_column: str = "pIC50", variance_threshold: float = 0.01) -> pl.DataFrame:
        """Фильтрует признаки по дисперсии.

        Args:
            df: DataFrame с данными.
            target_column: Название колонки с целевой переменной.
            variance_threshold: Минимальная дисперсия для сохранения признака.

        Returns:
            DataFrame с отфильтрованными признаками.
        """
        feature_columns = [col for col in df.columns if col not in [target_column, "smiles", "molecule_chembl_id"]]

        # Вычисляем дисперсию для каждого признака
        variances = {}
        for col in feature_columns:
            try:
                variance = df.select(pl.col(col).var()).item()
                if variance is not None and variance >= variance_threshold:
                    variances[col] = variance
            except Exception:
                # Пропускаем проблемные колонки
                continue

        # Сохраняем колонки с достаточной дисперсией
        keep_columns = [target_column, *list(variances.keys())]
        if "smiles" in df.columns:
            keep_columns.append("smiles")
        if "molecule_chembl_id" in df.columns:
            keep_columns.append("molecule_chembl_id")

        df_filtered = df.select(keep_columns)

        logger.info(
            f"Отфильтровано признаков: {len(feature_columns)} -> {len(variances)} (порог дисперсии: {variance_threshold})"
        )

        return df_filtered

    def normalize_features(self, df: pl.DataFrame, target_column: str = "pIC50", method: str = "standard") -> pl.DataFrame:
        """Нормализует признаки.

        Args:
            df: DataFrame с данными.
            target_column: Название колонки с целевой переменной.
            method: Метод нормализации ('standard' или 'minmax').

        Returns:
            DataFrame с нормализованными признаками.
        """
        feature_columns = [col for col in df.columns if col not in [target_column, "smiles", "molecule_chembl_id"]]

        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            msg = f"Неподдерживаемый метод нормализации: {method}"
            raise ValueError(msg)

        # Конвертируем в numpy для обработки
        features_array = df.select(feature_columns).to_numpy()

        # Заменяем NaN на медианные значения перед нормализацией
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer(strategy="median")
        features_array = imputer.fit_transform(features_array)

        # Нормализуем
        features_normalized = self.scaler.fit_transform(features_array)

        # Создаем новый DataFrame
        normalized_df = pl.DataFrame(features_normalized, schema=feature_columns)

        # Добавляем обратно целевую переменную и другие колонки
        result_df = normalized_df
        result_df = result_df.with_columns(df.select(target_column).to_series().alias(target_column))

        if "smiles" in df.columns:
            result_df = result_df.with_columns(df.select("smiles").to_series().alias("smiles"))
        if "molecule_chembl_id" in df.columns:
            result_df = result_df.with_columns(df.select("molecule_chembl_id").to_series().alias("molecule_chembl_id"))

        logger.info(f"Нормализованы признаки методом: {method}")

        return result_df

    def split_data(
        self, df: pl.DataFrame, target_column: str = "pIC50", test_size: float = 0.2
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
        """Разделяет данные на обучающую и тестовую выборки.

        Args:
            df: DataFrame с данными.
            target_column: Название колонки с целевой переменной.
            test_size: Доля тестовой выборки.

        Returns:
            Кортеж (X_train, X_test, y_train, y_test).
        """
        feature_columns = [col for col in df.columns if col not in [target_column, "smiles", "molecule_chembl_id"]]

        X = df.select(feature_columns)
        y = df.select(target_column).to_series()

        # Конвертируем в pandas для sklearn
        X_pandas = X.to_pandas()
        y_pandas = y.to_pandas()

        X_train, X_test, y_train, y_test = train_test_split(
            X_pandas, y_pandas, test_size=test_size, random_state=self.random_state
        )

        # Конвертируем обратно в polars
        X_train_pl = pl.from_pandas(X_train)
        X_test_pl = pl.from_pandas(X_test)
        y_train_pl = pl.from_pandas(y_train.to_frame()).to_series()
        y_test_pl = pl.from_pandas(y_test.to_frame()).to_series()

        logger.info(f"Данные разделены: обучение {X_train_pl.shape[0]}, тест {X_test_pl.shape[0]}")

        return X_train_pl, X_test_pl, y_train_pl, y_test_pl

    def apply_pca(
        self, X_train: pl.DataFrame, X_test: pl.DataFrame, n_components: float = 0.95
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Применяет PCA для снижения размерности.

        Args:
            X_train: Обучающие признаки.
            X_test: Тестовые признаки.
            n_components: Количество компонент или доля объясненной дисперсии.

        Returns:
            Кортеж (X_train_pca, X_test_pca).
        """
        self.pca = PCA(n_components=n_components, random_state=self.random_state)

        # Конвертируем в numpy
        X_train_array = X_train.to_numpy()
        X_test_array = X_test.to_numpy()

        # Применяем PCA
        X_train_pca = self.pca.fit_transform(X_train_array)
        X_test_pca = self.pca.transform(X_test_array)

        # Создаем колонки для PCA компонент
        pca_columns = [f"PC_{i + 1}" for i in range(X_train_pca.shape[1])]

        # Конвертируем обратно в polars
        X_train_pca_pl = pl.DataFrame(X_train_pca, schema=pca_columns)
        X_test_pca_pl = pl.DataFrame(X_test_pca, schema=pca_columns)

        explained_variance = self.pca.explained_variance_ratio_.sum()
        logger.info(
            f"PCA применен: {X_train.shape[1]} -> {X_train_pca.shape[1]} признаков, объяснено {explained_variance:.3f} дисперсии"
        )

        return X_train_pca_pl, X_test_pca_pl


def prepare_data_pipeline(
    file_path: str,
    target_column: str = "pIC50",
    test_size: float = 0.2,
    variance_threshold: float = 0.01,
    normalization: str = "standard",
    apply_pca: bool = False,
    pca_components: float = 0.95,
    random_state: int = 42,
) -> dict:
    """Полный пайплайн подготовки данных.

    Args:
        file_path: Путь к файлу с данными.
        target_column: Название колонки с целевой переменной.
        test_size: Доля тестовой выборки.
        variance_threshold: Минимальная дисперсия для сохранения признака.
        normalization: Метод нормализации ('standard' или 'minmax').
        apply_pca: Применять ли PCA.
        pca_components: Количество PCA компонент.
        random_state: Seed для воспроизводимости.

    Returns:
        Словарь с подготовленными данными и метаинформацией.
    """
    prep = DataPreparation(random_state=random_state)

    # Загружаем данные
    df = prep.load_data(file_path, target_column)

    # Очищаем данные
    df_clean = prep.clean_data(df, target_column)

    # Фильтруем признаки
    df_filtered = prep.filter_features(df_clean, target_column, variance_threshold)

    # Нормализуем признаки
    df_normalized = prep.normalize_features(df_filtered, target_column, normalization)

    # Разделяем данные
    X_train, X_test, y_train, y_test = prep.split_data(df_normalized, target_column, test_size)

    # Применяем PCA если нужно
    if apply_pca:
        X_train, X_test = prep.apply_pca(X_train, X_test, pca_components)

    result = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": prep.scaler,
        "pca": prep.pca,
        "original_shape": df.shape,
        "final_shape": (X_train.shape[0] + X_test.shape[0], X_train.shape[1]),
        "target_column": target_column,
    }

    return result
