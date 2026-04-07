from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
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


def build_tree_figure_size(model: DecisionTreeClassifier) -> tuple[float, float, int]:
    depth = model.get_depth()
    leaf_count = model.get_n_leaves()

    width = max(28.0, leaf_count * 1.8)
    height = max(12.0, (depth + 1) * 2.6)
    fontsize = max(7, min(12, int(220 / max(leaf_count, 1))))
    
    width = 22
    height = 8
    fontsize = 10
    return width, height, fontsize


def simplify_tree_labels(plot_annotations: list) -> None:
    for annotation in plot_annotations:
        raw_text = annotation.get_text().strip()
        if not raw_text:
            continue

        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        condition_line = next((line for line in lines if "<=" in line or ">" in line), "")
        class_line = next((line for line in lines if line.startswith("class = ")), "")

        if condition_line:
            label_text = condition_line
        elif class_line:
            label_text = class_line.replace("class = ", "")
        else:
            label_text = ""

        if len(label_text) >= 8:
            label_text = "\n".join(
                textwrap.wrap(
                    label_text,
                    width=8,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
            )

        annotation.set_text(label_text)
        annotation.set_linespacing(1.4)


def main() -> None:
    train_df, test_df = load_clean_data()
    x_train, y_train, x_test, y_test = split_features_target(train_df, test_df)

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Decision Tree Results")
    print(f"Training shape: {x_train.shape}")
    print(f"Testing shape: {x_test.shape}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    feature_importance = pd.Series(
        model.feature_importances_,
        index=x_train.columns,
    ).sort_values(ascending=False)
    print("\nTop 10 Feature Importances:")
    print(feature_importance.head(10).round(4))

    figure_width, figure_height, font_size = build_tree_figure_size(model)

    plt.figure(figsize=(figure_width, figure_height))
    plot_annotations = plot_tree(
        model,
        feature_names=x_train.columns,
        class_names=["No Churn", "Churn"],
        filled=True,
        rounded=True,
        impurity=False,
        proportion=False,
        fontsize=font_size,
    )
    simplify_tree_labels(plot_annotations)
    plt.title("Decision Tree for Customer Churn")
    plt.tight_layout(pad=2.0)
    plt.savefig(TREE_IMAGE_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nDecision tree plot saved to: {TREE_IMAGE_PATH}")


if __name__ == "__main__":
    main()
