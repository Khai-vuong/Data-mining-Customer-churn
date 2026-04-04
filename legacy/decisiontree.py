from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree


BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "clean_train.csv"
TEST_FILE = BASE_DIR / "clean_test.csv"
TREE_IMAGE_PATH = BASE_DIR / "decision_tree_visualization.png"


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

    model = DecisionTreeClassifier(
        random_state=42,
        criterion="gini",
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Decision Tree trained successfully")
    print(f"Training shape: {X_train.shape}")
    print(f"Testing shape: {X_test.shape}")
    print(f"Accuracy (100% clean_data --predict--> clean_test_data): {accuracy_score(y_test, y_pred):.4f}")

    feature_importances = pd.Series(
        model.feature_importances_,
        index=X_train.columns,
    ).round(3).sort_values(ascending=False)

    print("\nFeature Importances:")
    print(feature_importances)

    plt.figure(figsize=(50, 30))
    plot_tree(
        model,
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
    plt.show()


if __name__ == "__main__":
    main()
