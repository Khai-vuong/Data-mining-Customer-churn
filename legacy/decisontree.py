from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "clean_data.csv"
TEST_FILE = BASE_DIR / "clean_test_data.csv"
TREE_IMAGE_PATH = BASE_DIR / "decision_tree_visualization.png"


def load_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned training and testing data from CSV files."""
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Cannot find training file: {TRAIN_FILE}")
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Cannot find testing file: {TEST_FILE}")

    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)

    return df_train, df_test


def main() -> None:
    df_train, df_test = load_clean_data()

    if "churn" not in df_train.columns or "churn" not in df_test.columns:
        raise ValueError("Target column 'churn' not found in data")

    X = df_train.drop(columns="churn")
    y = df_train["churn"]

    X_test_file = df_test.drop(columns="churn")
    y_test_file = df_test["churn"]
    X_test_file = X_test_file.reindex(columns=X.columns, fill_value=0)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model_params = {
        "random_state": 42,
        "criterion": "entropy",
        "max_depth": 5,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
    }

    model_pp1 = DecisionTreeClassifier(**model_params)
    model_pp1.fit(X_train, y_train)
    y_pred_pp1 = model_pp1.predict(X_valid)

    model_pp2 = DecisionTreeClassifier(**model_params)
    model_pp2.fit(X_train, y_train)
    y_pred_pp2 = model_pp2.predict(X_test_file)

    model_pp3 = DecisionTreeClassifier(**model_params)
    model_pp3.fit(X, y)
    y_pred_pp3 = model_pp3.predict(X_test_file)

    print("Decision Tree trained successfully")
    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape (PP1): {X_valid.shape}")
    print(f"External test shape (PP2, PP3): {X_test_file.shape}")
    print(f"PP1 Accuracy (80% train -> 20% clean_data): {accuracy_score(y_valid, y_pred_pp1):.4f}")
    print(f"PP2 Accuracy (80% train -> clean_test_data): {accuracy_score(y_test_file, y_pred_pp2):.4f}")
    print(f"PP3 Accuracy (100% clean_data -> clean_test_data): {accuracy_score(y_test_file, y_pred_pp3):.4f}")

    # Map feature names to importances
    feature_importances = pd.Series(
        model_pp3.feature_importances_,
            index=X_train.columns
        ).round(3).sort_values(ascending=False)
    
    print("\nFeature Importances:")
    print(feature_importances)
    
    
    
    plt.figure(figsize=(50, 30))
    plot_tree(
        model_pp3,
        feature_names=X_train.columns,
        class_names=["Not Churn", "Churn"],
        filled=True,
        rounded=True,
        fontsize=6,
        max_depth=None,
    )
    plt.title("Decision Tree for Customer Churn", fontsize=16)
    plt.tight_layout()
    plt.savefig(TREE_IMAGE_PATH, dpi=300, bbox_inches="tight")
    print(f"\nDecision tree visualization saved to: {TREE_IMAGE_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
