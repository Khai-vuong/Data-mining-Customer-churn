from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "train.csv"
TEST_FILE = BASE_DIR / "test.csv"
OUTPUT_TRAIN_FILE = BASE_DIR / "clean_train.csv"
OUTPUT_TEST_FILE = BASE_DIR / "clean_test.csv"

BOOLEAN_COLUMNS = ["churn", "online_security", "tech_support"]
BOOLEAN_MAPPING = {"No": 0, "Yes": 1}

ONE_HOT_COLUMNS = ["payment_method", "internet_service"]

CONTRACT_MAPPING = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2,
}


def load_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find input file: {file_path}")
    return pd.read_csv(file_path)


def validate_same_schema(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    if list(train_df.columns) != list(test_df.columns):
        raise ValueError("train.csv and test.csv do not have the same column names/order.")

    if not train_df.dtypes.astype(str).equals(test_df.dtypes.astype(str)):
        raise ValueError("train.csv and test.csv do not have matching data types.")


def transform_common(df: pd.DataFrame) -> pd.DataFrame:
    transformed = df.copy()

    transformed = transformed.drop(columns=["customer_id"])
    transformed["internet_service"] = transformed["internet_service"].fillna("Unknown")

    for column in BOOLEAN_COLUMNS:
        unknown_values = set(transformed[column].dropna().unique()) - set(BOOLEAN_MAPPING.keys())
        if unknown_values:
            raise ValueError(
                f"Unexpected values in boolean column '{column}': {sorted(unknown_values)}"
            )
        transformed[column] = transformed[column].map(BOOLEAN_MAPPING).astype(int)

    unknown_contracts = set(transformed["contract"].dropna().unique()) - set(CONTRACT_MAPPING.keys())
    if unknown_contracts:
        raise ValueError(f"Unexpected values in 'contract': {sorted(unknown_contracts)}")

    transformed["contract"] = transformed["contract"].map(CONTRACT_MAPPING).astype(int)
    return transformed


def apply_one_hot_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoded_train = pd.get_dummies(train_df, columns=ONE_HOT_COLUMNS, dtype=int)
    encoded_test = pd.get_dummies(test_df, columns=ONE_HOT_COLUMNS, dtype=int)

    target_column = "churn"
    train_features = [column for column in encoded_train.columns if column != target_column]
    test_features = [column for column in encoded_test.columns if column != target_column]
    all_features = sorted(set(train_features) | set(test_features))

    encoded_train = encoded_train.reindex(columns=all_features + [target_column], fill_value=0)
    encoded_test = encoded_test.reindex(columns=all_features + [target_column], fill_value=0)

    return encoded_train, encoded_test


def main() -> None:
    train_df = load_data(TRAIN_FILE)
    test_df = load_data(TEST_FILE)

    validate_same_schema(train_df, test_df)

    transformed_train = transform_common(train_df)
    transformed_test = transform_common(test_df)

    clean_train, clean_test = apply_one_hot_encoding(transformed_train, transformed_test)

    clean_train.to_csv(OUTPUT_TRAIN_FILE, index=False)
    clean_test.to_csv(OUTPUT_TEST_FILE, index=False)

    print("Preprocessing completed successfully.")
    print(f"Saved: {OUTPUT_TRAIN_FILE.name} - shape {clean_train.shape}")
    print(f"Saved: {OUTPUT_TEST_FILE.name} - shape {clean_test.shape}")
    print("Output columns:")
    print(", ".join(clean_train.columns))


if __name__ == "__main__":
    main()
