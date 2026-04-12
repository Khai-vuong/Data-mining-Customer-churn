from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "clean_train.csv"
TEST_FILE = BASE_DIR / "clean_test.csv"


def load_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Cannot find training file: {TRAIN_FILE}")
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Cannot find testing file: {TEST_FILE}")

    return pd.read_csv(TRAIN_FILE), pd.read_csv(TEST_FILE)


def prepare_features() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    df_train, df_test = load_clean_data()

    if "churn" not in df_train.columns or "churn" not in df_test.columns:
        raise ValueError("Target column 'churn' not found in data")

    X_train = df_train.drop(columns="churn")
    y_train = df_train["churn"]

    X_test = df_test.drop(columns="churn")
    y_test = df_test["churn"]
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    return X_train, y_train, X_test, y_test


def build_models() -> dict[str, object]:
    return {
        "GaussianNB": GaussianNB(),
        "KNN": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=10, metric="minkowski", weights="distance")),
            ]
        ),
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        random_state=42,
                        max_iter=1000,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            criterion="gini",
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            criterion="gini",
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=1,
        ),
    }


def evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    results: list[dict[str, float | str]] = []

    for model_name, model in build_models().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        results.append(
            {
                "model": model_name,
                "accuracy": accuracy,
            }
        )

    result_df = pd.DataFrame(results).sort_values(by="accuracy", ascending=False)
    result_df["accuracy"] = result_df["accuracy"].round(4)
    return result_df.reset_index(drop=True)


def main() -> None:
    X_train, y_train, X_test, y_test = prepare_features()
    results = evaluate_models(X_train, y_train, X_test, y_test)

    print("Model comparison completed successfully")
    print(f"Training shape: {X_train.shape}")
    print(f"Testing shape: {X_test.shape}")
    print("\nAccuracy comparison:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
