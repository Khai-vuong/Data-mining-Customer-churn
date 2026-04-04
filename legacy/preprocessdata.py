from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RAW_TRAIN_FILE = BASE_DIR / "customer_churn_dataset-training-master.csv"
RAW_TEST_FILE = BASE_DIR / "customer_churn_dataset-testing-master.csv"

OUTPUT_TRAIN_FILE = BASE_DIR / "clean_train.csv"
OUTPUT_TEST_FILE = BASE_DIR / "clean_test.csv"

# Keep the old filenames too so the existing model scripts still work.
LEGACY_TRAIN_FILE = BASE_DIR / "clean_data.csv"
LEGACY_TEST_FILE = BASE_DIR / "clean_test_data.csv"

ONE_HOT_COLUMNS = ["gender"]
ORDINAL_MAPPINGS = {
    "subscription_type": {
        "Basic": 0,
        "Standard": 1,
        "Premium": 2,
    },
    "contract_length": {
        "Monthly": 0,
        "Quarterly": 1,
        "Annual": 2,
    },
}
DISCRETE_INT_COLUMNS = [
    "age",
    "tenure",
    "usage_frequency",
    "support_calls",
    "payment_delay",
    "last_interaction",
    "churn",
]


def load_raw_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find raw data file: {file_path}")
    return pd.read_csv(file_path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [col.strip().lower().replace(" ", "_") for col in normalized.columns]
    return normalized


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = normalize_columns(df)

    if "customerid" in cleaned.columns:
        cleaned = cleaned.drop(columns="customerid")

    cleaned = cleaned.dropna().copy()

    for column in DISCRETE_INT_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype(int)

    return cleaned


def align_categories(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned_train = train_df.copy()
    aligned_test = test_df.copy()

    for column in ONE_HOT_COLUMNS:
        if column not in aligned_train.columns or column not in aligned_test.columns:
            raise ValueError(f"Missing categorical column '{column}' in input data")

        categories = sorted(aligned_train[column].dropna().unique().tolist())
        aligned_train[column] = pd.Categorical(aligned_train[column], categories=categories)
        aligned_test[column] = pd.Categorical(aligned_test[column], categories=categories)

    return aligned_train, aligned_test


def apply_ordinal_encoding(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy()

    for column, mapping in ORDINAL_MAPPINGS.items():
        if column not in encoded.columns:
            raise ValueError(f"Missing ordinal column '{column}' in input data")

        unknown_values = set(encoded[column].dropna().unique()) - set(mapping.keys())
        if unknown_values:
            raise ValueError(
                f"Unexpected values found in '{column}': {sorted(unknown_values)}"
            )

        encoded[column] = encoded[column].map(mapping).astype(int)

    return encoded


def encode_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned_train, aligned_test = align_categories(train_df, test_df)
    aligned_train = apply_ordinal_encoding(aligned_train)
    aligned_test = apply_ordinal_encoding(aligned_test)

    encoded_train = pd.get_dummies(
        aligned_train,
        columns=ONE_HOT_COLUMNS,
        drop_first=False,
        dtype=int,
    )
    encoded_test = pd.get_dummies(
        aligned_test,
        columns=ONE_HOT_COLUMNS,
        drop_first=False,
        dtype=int,
    )

    target_column = "churn"
    feature_columns = [col for col in encoded_train.columns if col != target_column]
    encoded_train = encoded_train.reindex(columns=feature_columns + [target_column], fill_value=0)
    encoded_test = encoded_test.reindex(columns=feature_columns + [target_column], fill_value=0)

    return encoded_train, encoded_test


def export_outputs(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_df.to_csv(OUTPUT_TRAIN_FILE, index=False)
    test_df.to_csv(OUTPUT_TEST_FILE, index=False)

    try:
        train_df.to_csv(LEGACY_TRAIN_FILE, index=False)
        print(f"Saved: {LEGACY_TRAIN_FILE.name}")
    except PermissionError:
        print(f"Skipped: {LEGACY_TRAIN_FILE.name} is currently locked by another process.")

    try:
        test_df.to_csv(LEGACY_TEST_FILE, index=False)
        print(f"Saved: {LEGACY_TEST_FILE.name}")
    except PermissionError:
        print(f"Skipped: {LEGACY_TEST_FILE.name} is currently locked by another process.")


def main() -> None:
    raw_train = load_raw_data(RAW_TRAIN_FILE)
    raw_test = load_raw_data(RAW_TEST_FILE)

    clean_train = basic_clean(raw_train)
    clean_test = basic_clean(raw_test)
    encoded_train, encoded_test = encode_features(clean_train, clean_test)

    export_outputs(encoded_train, encoded_test)

    print("Preprocessing completed successfully.")
    print(f"Train shape: {encoded_train.shape}")
    print(f"Test shape: {encoded_test.shape}")
    print(f"Saved: {OUTPUT_TRAIN_FILE.name}")
    print(f"Saved: {OUTPUT_TEST_FILE.name}")


if __name__ == "__main__":
    main()
