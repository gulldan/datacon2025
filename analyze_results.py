#!/usr/bin/env python3
"""Скрипт для анализа и визуализации результатов генерации молекул.
Создает графики распределения QED, SA Score и других свойств.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Настройка стиля графиков
plt.style.use("default")
sns.set_palette("husl")


def create_comprehensive_analysis(df, output_path="data/generated_molecules/plots/comprehensive_analysis.png") -> None:
    """Создает комплексный анализ результатов."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Комплексный анализ сгенерированных молекул", fontsize=16, fontweight="bold")

    # 1. Распределение QED
    axes[0, 0].hist(df["QED"], bins=15, alpha=0.7, color="skyblue", edgecolor="black")
    axes[0, 0].axvline(df["QED"].mean(), color="red", linestyle="--", label=f"Среднее: {df['QED'].mean():.3f}")
    axes[0, 0].axvline(0.7, color="green", linestyle="-", linewidth=2, label="Цель: QED ≥ 0.7")
    axes[0, 0].set_xlabel("QED")
    axes[0, 0].set_ylabel("Количество молекул")
    axes[0, 0].set_title("Распределение QED")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Распределение SA Score
    axes[0, 1].hist(df["SA_Score"], bins=15, alpha=0.7, color="lightgreen", edgecolor="black")
    axes[0, 1].axvline(df["SA_Score"].mean(), color="red", linestyle="--", label=f"Среднее: {df['SA_Score'].mean():.3f}")
    axes[0, 1].axvline(2.0, color="blue", linestyle="-", linewidth=2, label="Минимум: 2.0")
    axes[0, 1].axvline(6.0, color="blue", linestyle="-", linewidth=2, label="Максимум: 6.0")
    axes[0, 1].set_xlabel("SA Score")
    axes[0, 1].set_ylabel("Количество молекул")
    axes[0, 1].set_title("Распределение SA Score")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Распределение pIC50
    axes[0, 2].hist(df["pIC50"], bins=15, alpha=0.7, color="orange", edgecolor="black")
    axes[0, 2].axvline(df["pIC50"].mean(), color="red", linestyle="--", label=f"Среднее: {df['pIC50'].mean():.3f}")
    axes[0, 2].axvline(5.0, color="green", linestyle="-", linewidth=2, label="Минимум: 5.0")
    axes[0, 2].set_xlabel("pIC50")
    axes[0, 2].set_ylabel("Количество молекул")
    axes[0, 2].set_title("Распределение pIC50")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. QED vs SA Score
    scatter = axes[1, 0].scatter(df["QED"], df["SA_Score"], c=df["pIC50"], cmap="viridis", alpha=0.7, s=50)
    axes[1, 0].set_xlabel("QED")
    axes[1, 0].set_ylabel("SA Score")
    axes[1, 0].set_title("QED vs SA Score")
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 0], label="pIC50")

    # 5. Молекулярная масса vs LogP
    scatter = axes[1, 1].scatter(df["Mol_Weight"], df["LogP"], c=df["QED"], cmap="plasma", alpha=0.7, s=50)
    axes[1, 1].set_xlabel("Молекулярная масса (Da)")
    axes[1, 1].set_ylabel("LogP")
    axes[1, 1].set_title("Молекулярная масса vs LogP")
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 1], label="QED")

    # 6. Нарушения Липинского
    lipinski_counts = df["Lipinski_Violations"].value_counts().sort_index()
    axes[1, 2].bar(lipinski_counts.index, lipinski_counts.values, color="lightcoral", alpha=0.7, edgecolor="black")
    axes[1, 2].set_xlabel("Количество нарушений Липинского")
    axes[1, 2].set_ylabel("Количество молекул")
    axes[1, 2].set_title("Нарушения правил Липинского")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_qed_sa_distribution(df, output_path="data/generated_molecules/plots/qed_sa_distribution.png") -> None:
    """Создает график распределения QED и SA Score."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Распределение QED и SA Score", fontsize=16, fontweight="bold")

    # QED распределение
    ax1.hist(df["QED"], bins=20, alpha=0.7, color="skyblue", edgecolor="black")
    ax1.axvline(df["QED"].mean(), color="red", linestyle="--", linewidth=2, label=f"Среднее: {df['QED'].mean():.3f}")
    ax1.axvline(0.7, color="green", linestyle="-", linewidth=2, label="Цель: QED ≥ 0.7")
    ax1.set_xlabel("QED")
    ax1.set_ylabel("Количество молекул")
    ax1.set_title("Распределение QED")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # SA Score распределение
    ax2.hist(df["SA_Score"], bins=20, alpha=0.7, color="lightgreen", edgecolor="black")
    ax2.axvline(df["SA_Score"].mean(), color="red", linestyle="--", linewidth=2, label=f"Среднее: {df['SA_Score'].mean():.3f}")
    ax2.axvline(2.0, color="blue", linestyle="-", linewidth=2, label="Минимум: 2.0")
    ax2.axvline(6.0, color="blue", linestyle="-", linewidth=2, label="Максимум: 6.0")
    ax2.set_xlabel("SA Score")
    ax2.set_ylabel("Количество молекул")
    ax2.set_title("Распределение SA Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_selected_candidates_analysis(
    df, output_path="data/generated_molecules/plots/selected_candidates_analysis.png"
) -> None:
    """Создает анализ отобранных кандидатов."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Анализ отобранных молекул-кандидатов", fontsize=16, fontweight="bold")

    # Топ-10 молекул по QED
    top_qed = df.nlargest(10, "QED")
    axes[0, 0].barh(range(len(top_qed)), top_qed["QED"], color="skyblue", alpha=0.7)
    axes[0, 0].set_yticks(range(len(top_qed)))
    axes[0, 0].set_yticklabels([f"{i + 1}" for i in range(len(top_qed))])
    axes[0, 0].set_xlabel("QED")
    axes[0, 0].set_title("Топ-10 молекул по QED")
    axes[0, 0].grid(True, alpha=0.3)

    # Топ-10 молекул по pIC50
    top_pic50 = df.nlargest(10, "pIC50")
    axes[0, 1].barh(range(len(top_pic50)), top_pic50["pIC50"], color="orange", alpha=0.7)
    axes[0, 1].set_yticks(range(len(top_pic50)))
    axes[0, 1].set_yticklabels([f"{i + 1}" for i in range(len(top_pic50))])
    axes[0, 1].set_xlabel("pIC50")
    axes[0, 1].set_title("Топ-10 молекул по pIC50")
    axes[0, 1].grid(True, alpha=0.3)

    # Корреляционная матрица
    numeric_cols = ["QED", "SA_Score", "pIC50", "Mol_Weight", "LogP"]
    corr_matrix = df[numeric_cols].corr()
    im = axes[1, 0].imshow(corr_matrix, cmap="coolwarm", aspect="auto")
    axes[1, 0].set_xticks(range(len(numeric_cols)))
    axes[1, 0].set_yticks(range(len(numeric_cols)))
    axes[1, 0].set_xticklabels(numeric_cols, rotation=45)
    axes[1, 0].set_yticklabels(numeric_cols)
    axes[1, 0].set_title("Корреляционная матрица")

    # Добавляем значения корреляции
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            axes[1, 0].text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontweight="bold")

    plt.colorbar(im, ax=axes[1, 0])

    # Статистика по критериям
    criteria_stats = {
        "QED ≥ 0.7": len(df[df["QED"] >= 0.7]),
        "QED ≥ 0.8": len(df[df["QED"] >= 0.8]),
        "SA Score 2-6": len(df[(df["SA_Score"] >= 2.0) & (df["SA_Score"] <= 6.0)]),
        "pIC50 ≥ 6.0": len(df[df["pIC50"] >= 6.0]),
        "Lipinski = 0": len(df[df["Lipinski_Violations"] == 0]),
    }

    axes[1, 1].bar(
        criteria_stats.keys(), criteria_stats.values(), color=["skyblue", "lightgreen", "orange", "red", "purple"], alpha=0.7
    )
    axes[1, 1].set_ylabel("Количество молекул")
    axes[1, 1].set_title("Статистика по критериям")
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_summary_statistics(df) -> None:
    """Выводит сводную статистику."""


def main() -> None:
    """Основная функция."""
    # Создаем директорию для графиков
    plots_dir = Path("data/generated_molecules/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем данные
    input_file = "selected_hits_sa_fixed.csv"
    if not Path(input_file).exists():
        return

    df = pd.read_csv(input_file)

    # Выводим статистику
    print_summary_statistics(df)

    # Создаем графики
    create_comprehensive_analysis(df)
    create_qed_sa_distribution(df)
    create_selected_candidates_analysis(df)


if __name__ == "__main__":
    main()
