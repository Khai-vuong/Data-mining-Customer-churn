from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "clean_train.csv"
TEST_FILE = BASE_DIR / "clean_test.csv"


def load_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Cannot find training file: {TRAIN_FILE}")
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Cannot find testing file: {TEST_FILE}")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    return train_df, test_df


def split_features_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if "churn" not in train_df.columns or "churn" not in test_df.columns:
        raise ValueError("Target column 'churn' not found in input data.")

    x_train = train_df.drop(columns="churn")
    y_train = train_df["churn"]

    x_test = test_df.drop(columns="churn")
    y_test = test_df["churn"]
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    return x_train, y_train, x_test, y_test


def main() -> None:
    train_df, test_df = load_clean_data()
    x_train, y_train, x_test, y_test = split_features_target(train_df, test_df)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=15, metric="minkowski", weights="distance")),
        ]
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("KNN Results")
    print(f"Training shape: {x_train.shape}")
    print(f"Testing shape: {x_test.shape}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nKNN Parameters:")
    print("n_neighbors=15, metric='minkowski', weights='distance'")


if __name__ == "__main__":
    main()
