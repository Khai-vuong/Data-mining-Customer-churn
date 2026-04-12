from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_FILE = Path("test.csv")
OUTPUT_DIR = Path("eda_test_outputs")


def encode_for_correlation(df: pd.DataFrame) -> pd.DataFrame:
    encoded_df = pd.DataFrame(index=df.index)

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_numeric_dtype(series):
            encoded_df[column] = series
        else:
            filled = series.fillna("Missing").astype(str)
            encoded_df[column] = pd.factorize(filled, sort=True)[0]

    return encoded_df


def save_correlation_outputs(df: pd.DataFrame, output_dir: Path) -> None:
    encoded_df = encode_for_correlation(df)
    correlation_df = encoded_df.corr(numeric_only=False)

    correlation_df.to_csv(
        output_dir / "correlation_matrix.csv",
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(12, 9))
    ax = sns.heatmap(
        correlation_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Heatmap - test.csv", pad=12)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()

    if "churn" in correlation_df.columns:
        churn_corr = correlation_df["churn"].sort_values(ascending=False)
        churn_corr.to_csv(
            output_dir / "correlation_with_churn.csv",
            header=["correlation_with_churn"],
            encoding="utf-8-sig",
        )


def describe_column(df: pd.DataFrame, column: str) -> dict:
    series = df[column]
    missing_count = int(series.isna().sum())
    unique_count = int(series.nunique(dropna=True))

    info = {
        "column": column,
        "dtype": str(series.dtype),
        "missing_count": missing_count,
        "missing_percent": round((missing_count / len(df)) * 100, 2),
        "unique_count": unique_count,
    }

    if pd.api.types.is_numeric_dtype(series):
        info.update(
            {
                "column_type": "numeric",
                "mean": round(float(series.mean()), 4) if series.notna().any() else None,
                "median": round(float(series.median()), 4) if series.notna().any() else None,
                "std": round(float(series.std()), 4) if series.notna().sum() > 1 else None,
                "min": round(float(series.min()), 4) if series.notna().any() else None,
                "max": round(float(series.max()), 4) if series.notna().any() else None,
            }
        )
    else:
        top_values = series.fillna("Missing").value_counts().head(5)
        info.update(
            {
                "column_type": "categorical",
                "top_values": "; ".join(f"{idx}: {value}" for idx, value in top_values.items()),
            }
        )

    return info


def plot_numeric_column(df: pd.DataFrame, column: str, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    clean_series = df[column].dropna()

    sns.histplot(clean_series, kde=True, bins=30, ax=axes[0], color="#2a9d8f")
    axes[0].set_title(f"Histogram - {column}")
    axes[0].set_xlabel(column)

    sns.boxplot(x=clean_series, ax=axes[1], color="#e9c46a")
    axes[1].set_title(f"Boxplot - {column}")
    axes[1].set_xlabel(column)

    fig.suptitle(f"EDA for numeric column: {column}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"{column}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_categorical_column(df: pd.DataFrame, column: str, output_dir: Path) -> None:
    plot_df = df.copy()
    plot_df[column] = plot_df[column].fillna("Missing").astype(str)
    order = plot_df[column].value_counts().index

    plt.figure(figsize=(12, 6))
    ax = sns.countplot(data=plot_df, x=column, order=order, color="#8ecae6")
    ax.set_title(f"Count plot - {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    plt.xticks(rotation=30, ha="right")

    for patch in ax.patches:
        height = int(patch.get_height())
        ax.annotate(
            str(height),
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(output_dir / f"{column}.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_column_descriptions(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    descriptions = [describe_column(df, column) for column in df.columns]
    summary_df = pd.DataFrame(descriptions)
    summary_df.to_csv(output_dir / "column_summary.csv", index=False, encoding="utf-8-sig")

    report_lines = [
        "EDA REPORT FOR TEST.CSV",
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        "",
    ]

    for item in descriptions:
        report_lines.append(f"Column: {item['column']}")
        report_lines.append(f"- Dtype: {item['dtype']}")
        report_lines.append(f"- Type: {item['column_type']}")
        report_lines.append(
            f"- Missing: {item['missing_count']} ({item['missing_percent']}%)"
        )
        report_lines.append(f"- Unique values: {item['unique_count']}")

        if item["column_type"] == "numeric":
            report_lines.append(f"- Mean: {item['mean']}")
            report_lines.append(f"- Median: {item['median']}")
            report_lines.append(f"- Std: {item['std']}")
            report_lines.append(f"- Min: {item['min']}")
            report_lines.append(f"- Max: {item['max']}")
        else:
            report_lines.append(f"- Top values: {item['top_values']}")

        report_lines.append("")

    (output_dir / "column_summary.txt").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    return summary_df


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Cannot find input file: {DATA_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = pd.read_csv(DATA_FILE)
    summary_df = save_column_descriptions(df, OUTPUT_DIR)
    save_correlation_outputs(df, OUTPUT_DIR)

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            plot_numeric_column(df, column, OUTPUT_DIR)
        else:
            plot_categorical_column(df, column, OUTPUT_DIR)

    print("EDA completed for test.csv")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")
    print("Saved correlation heatmap: correlation_heatmap.png")
    print("Column overview:")
    print(summary_df[["column", "column_type", "dtype", "missing_count", "unique_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
