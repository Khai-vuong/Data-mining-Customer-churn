from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "clean_train.csv"
TEST_FILE = BASE_DIR / "clean_test.csv"


def load_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Cannot find training file: {TRAIN_FILE}")
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Cannot find testing file: {TEST_FILE}")

    return pd.read_csv(TRAIN_FILE), pd.read_csv(TEST_FILE)


def main() -> None:
    df_train, df_test = load_clean_data()

    if "churn" not in df_train.columns or "churn" not in df_test.columns:
        raise ValueError("Target column 'churn' not found in data")

    X_train = df_train.drop(columns="churn")
    y_train = df_train["churn"]

    X_test = df_test.drop(columns="churn")
    y_test = df_test["churn"]
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    model = RandomForestClassifier(
        n_estimators=200,
        criterion="gini",
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    feature_importances = pd.Series(
        model.feature_importances_,
        index=X_train.columns,
    ).sort_values(ascending=False)

    print("Random Forest trained successfully")
    print(f"Training shape: {X_train.shape}")
    print(f"Testing shape: {X_test.shape}")
    print(f"Accuracy (100% clean_train -> clean_test): {accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("\nTop 10 Feature Importances:")
    print(feature_importances.head(10).round(4))


if __name__ == "__main__":
    main()
