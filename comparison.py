from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


CURRENT_DIR = Path(__file__).resolve().parent
sys.path = [path for path in sys.path if Path(path).resolve() != CURRENT_DIR]

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


BASE_DIR = CURRENT_DIR
TRAIN_FILE = BASE_DIR / "clean_train.csv"
TEST_FILE = BASE_DIR / "clean_test.csv"
OUTPUT_FILE = BASE_DIR / "comparison.csv"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: Callable[[], object]
    supports_feature_importance: bool = False


def load_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Cannot find training file: {TRAIN_FILE}")
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Cannot find testing file: {TEST_FILE}")

    return pd.read_csv(TRAIN_FILE), pd.read_csv(TEST_FILE)


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


def build_model_registry() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="Decision Tree",
            builder=lambda: DecisionTreeClassifier(
                criterion="gini",
                max_depth=6,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
            ),
            supports_feature_importance=True,
        ),
        ModelSpec(
            name="Random Forest",
            builder=lambda: RandomForestClassifier(
                n_estimators=200,
                criterion="gini",
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=1,
            ),
            supports_feature_importance=True,
        ),
        ModelSpec(
            name="Logistic Regression",
            builder=lambda: Pipeline(
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
        ),
        ModelSpec(
            name="KNN",
            builder=lambda: Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "knn",
                        KNeighborsClassifier(
                            n_neighbors=15,
                            metric="minkowski",
                            weights="distance",
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            name="Gaussian Naive Bayes",
            builder=lambda: GaussianNB(),
        ),
        ModelSpec(
            name="XGBoost",
            builder=lambda: XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            ),
            supports_feature_importance=True,
        ),
        ModelSpec(
            name="LightGBM",
            builder=lambda: LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                num_leaves=31,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=1,
                verbose=-1,
            ),
            supports_feature_importance=True,
        ),
    ]


def extract_top_features(model: object, feature_names: list[str], top_k: int = 5) -> list[str]:
    if not hasattr(model, "feature_importances_"):
        return [""] * top_k

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(top_k).index.tolist()
    return top_features + [""] * (top_k - len(top_features))


def evaluate_model(
    model_spec: ModelSpec,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    model = model_spec.builder()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    top_features = [""] * 5
    if model_spec.supports_feature_importance:
        estimator = model
        if isinstance(model, Pipeline):
            estimator = model.steps[-1][1]
        top_features = extract_top_features(estimator, x_train.columns.tolist(), top_k=5)

    return {
        "model_name": model_spec.name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "top1": top_features[0],
        "top2": top_features[1],
        "top3": top_features[2],
        "top4": top_features[3],
        "top5": top_features[4],
    }


def main() -> None:
    train_df, test_df = load_clean_data()
    x_train, y_train, x_test, y_test = split_features_target(train_df, test_df)

    model_registry = build_model_registry()
    results = []

    for model_spec in model_registry:
        result = evaluate_model(model_spec, x_train, y_train, x_test, y_test)
        results.append(result)
        print(
            f"{result['model_name']}: "
            f"accuracy={result['accuracy']:.4f}, "
            f"recall={result['recall']:.4f}, "
            f"f1={result['f1']:.4f}"
        )

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\nSaved comparison file to: {OUTPUT_FILE}")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
