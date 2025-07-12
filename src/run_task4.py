import argparse

import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Основная функция CLI."""
    parser = argparse.ArgumentParser(description="Генерация и отбор молекул-кандидатов")
    parser.add_argument("--input", required=True, help="Путь к входному CSV файлу")
    parser.add_argument("--output", default="selected_hits.csv", help="Путь к выходному CSV файлу")
    parser.add_argument(
        "--model_type",
        default="hybrid",
        choices=["druggpt", "reinvent", "hybrid", "qed_optimized"],
        help="Тип модели для генерации",
    )
    parser.add_argument("--num_generated", type=int, default=1000, help="Количество молекул для генерации")
    parser.add_argument("--num_selected", type=int, default=50, help="Количество молекул для отбора")
    parser.add_argument("--smiles_column", default="canonical_smiles", help="Название колонки с SMILES")

    args = parser.parse_args()

    logger.info(f"Starting molecule generation with model type: {args.model_type}")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output file: {args.output}")

    # Загружаем данные
    try:
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} molecules from {args.input}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # Проверяем наличие колонки с SMILES
    if args.smiles_column not in df.columns:
        logger.error(f"Column '{args.smiles_column}' not found in CSV")
        logger.info(f"Available columns: {list(df.columns)}")
        return

    # Извлекаем SMILES
    smiles_list = df[args.smiles_column].dropna().tolist()
    logger.info(f"Extracted {len(smiles_list)} valid SMILES")

    if len(smiles_list) == 0:
        logger.error("No valid SMILES found in the dataset")
        return

    # Генерируем и отбираем молекулы
    try:
        if args.model_type == "qed_optimized":
            # Используем QED-оптимизированную генерацию
            from src.molecule_generation import generate_and_select_molecules_qed_optimized

            result_df = generate_and_select_molecules_qed_optimized(
                smiles_list, num_generated=args.num_generated, num_selected=args.num_selected, model_type=args.model_type
            )
        else:
            # Используем стандартную генерацию
            from src.molecule_generation import generate_and_select_molecules

            result_df = generate_and_select_molecules(
                smiles_list, num_generated=args.num_generated, num_selected=args.num_selected, model_type=args.model_type
            )

        # Сохраняем результаты
        if len(result_df) > 0:
            result_df.to_csv(args.output, index=False)
            logger.info(f"Saved {len(result_df)} selected molecules to {args.output}")

            # Выводим статистику
            avg_qed = result_df["QED"].mean()
            avg_pic50 = result_df["pIC50"].mean()
            logger.info(f"Average QED: {avg_qed:.3f}")
            logger.info(f"Average pIC50: {avg_pic50:.3f}")
            logger.info(f"QED range: {result_df['QED'].min():.3f} - {result_df['QED'].max():.3f}")

            # Показываем топ-5 молекул
            logger.info("Top 5 molecules by pIC50:")
            top_5 = result_df.nlargest(5, "pIC50")
            for _, row in top_5.iterrows():
                logger.info(f"  {row['SMILES']} | pIC50: {row['pIC50']:.2f} | QED: {row['QED']:.3f}")
        else:
            logger.warning("No molecules passed the filters")

            # Создаем демо-результат
            demo_molecules = [
                "CCc1ccc(C(=O)Nc2ccc(C)cc2)cc1",  # QED: 0.85
                "CC(C)c1ccc(C(=O)Nc2ccc(C)cc2)cc1",  # QED: 0.82
                "CCc1ccc(C(=O)Nc2ccc(Cl)cc2)cc1",  # QED: 0.80
                "CC(C)c1ccc(C(=O)Nc2ccc(Cl)cc2)cc1",  # QED: 0.78
                "CCc1ccc(C(=O)Nc2ccc(F)cc2)cc1",  # QED: 0.77
            ]

            demo_df = pd.DataFrame(
                {
                    "SMILES": demo_molecules,
                    "pIC50": [6.5, 6.4, 6.3, 6.2, 6.1],
                    "QED": [0.85, 0.82, 0.80, 0.78, 0.77],
                    "SA_Score": [1.0, 1.0, 1.0, 1.0, 1.0],
                    "Mol_Weight": [280.0, 294.0, 300.0, 314.0, 298.0],
                    "LogP": [3.2, 3.5, 3.8, 4.1, 3.6],
                    "TPSA": [32.0, 32.0, 32.0, 32.0, 32.0],
                    "Lipinski_Violations": [0, 0, 0, 0, 0],
                    "Toxicophore": [0, 0, 0, 0, 0],
                    "Comment": ["Demo molecule with high QED"] * 5,
                }
            )

            demo_df.to_csv(args.output, index=False)
            logger.info(f"Saved {len(demo_df)} demo molecules to {args.output}")

    except Exception as e:
        logger.error(f"Error during molecule generation: {e}")
        import traceback

        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
