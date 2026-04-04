from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "clean_train.csv"
TEST_FILE = BASE_DIR / "clean_test.csv"
TRAIN_HEATMAP_PATH = BASE_DIR / "correlation_heatmap_train.png"
TEST_HEATMAP_PATH = BASE_DIR / "correlation_heatmap_test.png"


def load_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Cannot find training file: {TRAIN_FILE}")
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Cannot find testing file: {TEST_FILE}")

    return pd.read_csv(TRAIN_FILE), pd.read_csv(TEST_FILE)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    correlation_matrix = df.corr(numeric_only=True)

    plt.figure(figsize=(20, 20))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        cbar=True,
    )
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def calculate_feature_churn_correlations(
    df: pd.DataFrame,
    target_column: str = "churn",
) -> pd.Series:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")

    return (
        df.corr(numeric_only=True)[target_column]
        .drop(labels=[target_column])
        .sort_values(key=lambda series: series.abs(), ascending=False)
    )


def build_correlation_comparison(
    train_correlations: pd.Series,
    test_correlations: pd.Series,
) -> pd.DataFrame:
    comparison_df = pd.DataFrame(
        {
            "train_corr": train_correlations,
            "test_corr": test_correlations.reindex(train_correlations.index),
        }
    )
    comparison_df["abs_train_corr"] = comparison_df["train_corr"].abs()
    comparison_df["abs_test_corr"] = comparison_df["test_corr"].abs()
    comparison_df["corr_gap"] = (comparison_df["test_corr"] - comparison_df["train_corr"]).abs()
    return comparison_df.sort_values(by="corr_gap", ascending=False)


def print_correlation_list(title: str, correlations: pd.Series, top_n: int | None = None) -> None:
    print(f"\n{title}")
    print("-" * len(title))

    series_to_print = correlations.head(top_n) if top_n is not None else correlations
    for feature, value in series_to_print.items():
        print(f"{feature:<20} {value:>8.4f}")


def print_analysis_summary(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    train_churn_rate = df_train["churn"].mean()
    test_churn_rate = df_test["churn"].mean()
    top_shift = comparison_df.head(5)

    print("\nPossible reason for low model accuracy")
    print("-------------------------------------")
    print(
        "The train and test sets do not appear to follow the same pattern, "
        "so the models learn relationships from train that weaken or change in test."
    )
    print(
        f"- Churn rate shifts from {train_churn_rate:.4f} in train to {test_churn_rate:.4f} in test."
    )
    print(
        "- Several important feature-to-churn correlations change a lot between the two files."
    )

    for feature, row in top_shift.iterrows():
        print(
            f"  {feature}: train={row['train_corr']:.4f}, "
            f"test={row['test_corr']:.4f}, gap={row['corr_gap']:.4f}"
        )

    print(
        "- This kind of distribution shift can hurt many different algorithms at the same time "
        "(KNN, Decision Tree, Random Forest, Logistic Regression, Naive Bayes), "
        "which matches the fact that all of them are only around 0.51-0.58 accuracy."
    )
    print(
        "- A concrete sign is that some strong train signals become much weaker or even change direction in test, "
        "for example contract_length, total_spend, tenure, and last_interaction."
    )


def main() -> None:
    df_train, df_test = load_clean_data()

    print("Generating correlation heatmaps...")
    plot_correlation_heatmap(
        df_train,
        "Correlation Heatmap - Clean Train",
        TRAIN_HEATMAP_PATH,
    )
    plot_correlation_heatmap(
        df_test,
        "Correlation Heatmap - Clean Test",
        TEST_HEATMAP_PATH,
    )

    train_correlations = calculate_feature_churn_correlations(df_train)
    test_correlations = calculate_feature_churn_correlations(df_test)
    comparison_df = build_correlation_comparison(train_correlations, test_correlations)

    print_correlation_list("Train feature correlations with churn", train_correlations)
    print_correlation_list("Test feature correlations with churn", test_correlations)

    print("\nLargest correlation changes between train and test")
    print("-------------------------------------------------")
    print(
        comparison_df[["train_corr", "test_corr", "corr_gap"]]
        .round(4)
        .to_string()
    )

    print_analysis_summary(df_train, df_test, comparison_df)

    print(f"\nSaved train heatmap to: {TRAIN_HEATMAP_PATH}")
    print(f"Saved test heatmap to: {TEST_HEATMAP_PATH}")


if __name__ == "__main__":
    main()
