from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y, )
    
    model = DecisionTreeClassifier( 
        random_state=42, 
        criterion="entropy", 
        max_depth=5, 
        min_samples_split=20, 
        min_samples_leaf=10, 
    ) 
    model.fit(X_train, y_train) 
    # y_pred = model.predict(X_test)
    
    y_pred = model.predict(X_test_file)

    print("Decision Tree trained successfully")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    # print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test_file, y_pred):.4f}")
    # print("\nConfusion matrix:")
    # print(confusion_matrix(y_test, y_pred))
    # print("\nClassification report:")
    # print(classification_report(y_test_file, y_pred))
    
    
    # Map feature names to importances
    feature_importances = pd.Series(
        model.feature_importances_,
            index=X_train.columns
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
