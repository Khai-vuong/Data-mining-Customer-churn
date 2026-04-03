from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "clean_train.csv"
TEST_FILE = BASE_DIR / "clean_test.csv"
ENABLE_FEATURE_SELECTION = True
FEATURE_SELECTION_MODE = "correlation"
CORRELATION_THRESHOLD = 0.05
SELECTED_FEATURES = [
    "support_calls",
    "total_spend",
    "contract_length_Monthly",
    "contract_length_Annual",
    "contract_length_Quarterly",
    "payment_delay",
    "age",
    "last_interaction",
    "gender_Female",
]



def load_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Cannot find training file: {TRAIN_FILE}")
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Cannot find testing file: {TEST_FILE}")

    return pd.read_csv(TRAIN_FILE), pd.read_csv(TEST_FILE)


def select_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    selected_features: list[str] | None = None,
    enabled: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not enabled or not selected_features:
        return X_train, X_test

    missing_in_train = [col for col in selected_features if col not in X_train.columns]
    missing_in_test = [col for col in selected_features if col not in X_test.columns]

    if missing_in_train:
        raise ValueError(f"Selected features not found in training data: {missing_in_train}")
    if missing_in_test:
        raise ValueError(f"Selected features not found in testing data: {missing_in_test}")

    return X_train[selected_features].copy(), X_test[selected_features].copy()


def select_high_correlation_features(
    df_train: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float,
    target_column: str = "churn",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.Series]:
    correlation_with_target = (
        df_train.corr(numeric_only=True)[target_column]
        .drop(labels=[target_column])
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )

    selected_features = correlation_with_target[
        correlation_with_target.abs() >= threshold
    ].index.tolist()

    if not selected_features:
        raise ValueError(
            f"No features met the correlation threshold of {threshold} for target '{target_column}'"
        )

    return (
        X_train[selected_features].copy(),
        X_test[selected_features].copy(),
        selected_features,
        correlation_with_target,
    )


def k_tunning_values() -> list[int]:
    return [5, 10, 15, 20, 25, 30]


def main() -> None:
    df_train, df_test = load_clean_data()

    if "churn" not in df_train.columns or "churn" not in df_test.columns:
        raise ValueError("Target column 'churn' not found in data")

    X_train = df_train.drop(columns="churn")
    y_train = df_train["churn"]

    X_test = df_test.drop(columns="churn")
    y_test = df_test["churn"]
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    correlation_info = None
    if ENABLE_FEATURE_SELECTION:
        if FEATURE_SELECTION_MODE == "manual":
            X_train, X_test = select_features(
                X_train,
                X_test,
                selected_features=SELECTED_FEATURES,
                enabled=True,
            )
        elif FEATURE_SELECTION_MODE == "correlation":
            X_train, X_test, selected_features, correlation_info = select_high_correlation_features(
                df_train=df_train,
                X_train=X_train,
                X_test=X_test,
                threshold=CORRELATION_THRESHOLD,
            )
        else:
            raise ValueError(
                f"Unsupported FEATURE_SELECTION_MODE: {FEATURE_SELECTION_MODE}"
            )

    k_values = k_tunning_values()
    if not k_values:
        raise ValueError("No k values found")

    print("KNN trained successfully")
    print(f"Training shape: {X_train.shape}")
    print(f"Testing shape: {X_test.shape}")
    print(f"Feature selection enabled: {ENABLE_FEATURE_SELECTION}")
    print(f"Feature selection mode: {FEATURE_SELECTION_MODE}")
    if FEATURE_SELECTION_MODE == "correlation":
        print(f"Correlation threshold: {CORRELATION_THRESHOLD}")
    print(f"Features used: {list(X_train.columns)}")
    if correlation_info is not None:
        print("\nFeature correlations with churn:")
        for feature in X_train.columns:
            print(f"{feature}: {correlation_info[feature]:.4f}")

    best_k = None
    best_accuracy = -1.0

    for k in k_values:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=k, metric="minkowski", weights="distance")),
            ]
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"k={k}: accuracy={accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    print(f"\nBest k: {best_k}")
    print(f"Best accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
