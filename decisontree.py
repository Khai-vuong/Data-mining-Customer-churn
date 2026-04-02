from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "customer_churn_dataset-training-master.csv"
TEST_FILE = BASE_DIR / "customer_churn_dataset-testing-master.csv"
TREE_IMAGE_PATH = BASE_DIR / "decision_tree_visualization.png"


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data: drop CustomerID, normalize columns, one-hot encode."""
    df = df.copy()
    df.drop(columns='CustomerID', inplace=True)
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    df.dropna(inplace=True)
    
    discrete_col = ['age', 'tenure', 'usage_frequency', 'support_calls', 
                    'payment_delay', 'last_interaction', 'churn']
    for col in discrete_col:
        df[col] = df[col].astype(int)
    
    df_encoded = pd.get_dummies(
        df, 
        columns=['gender', 'subscription_type', 'contract_length'], 
        drop_first=False, 
        dtype=int
    )
    return df_encoded


def load_and_preprocess_data() -> tuple:
    """Load and preprocess training and testing data."""
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Cannot find training file: {TRAIN_FILE}")
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Cannot find testing file: {TEST_FILE}")
    
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)
    
    df_train_processed = preprocess_data(df_train)
    df_test_processed = preprocess_data(df_test)
    
    return df_train_processed, df_test_processed


def main() -> None:
    df_train, df_test = load_and_preprocess_data()

    if "churn" not in df_train.columns or "churn" not in df_test.columns:
        raise ValueError("Target column 'churn' not found in data")

    # X_train = df_train.drop(columns="churn")
    # y_train = df_train["churn"]
    
    # X_test = df_test.drop(columns="churn")
    # y_test = df_test["churn"]
    
    

    # model = DecisionTreeClassifier(
    #     random_state=42,
    #     criterion="gini",
    #     max_depth=5,
    #     min_samples_split=20,
    #     min_samples_leaf=10,
    # )
    # model.fit(X_train, y_train)

    # y_pred = model.predict(X_test)
    
    X = df_train.drop(columns="churn") 
    y = df_train["churn"]

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y, )
    
    model = DecisionTreeClassifier( 
        random_state=42, 
        criterion="gini", 
        max_depth=5, 
        min_samples_split=20, 
        min_samples_leaf=10, 
    ) 
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)

    print("Decision Tree trained successfully")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    
    
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
