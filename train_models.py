"""Train multiple classification models for the Titanic survival dataset and save them for deployment."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
TARGET_COLUMN = "Survived"


def load_dataset() -> pd.DataFrame:
    """Load the public Titanic dataset from a public GitHub mirror."""
    data = pd.read_csv(DATA_URL)
    return data


def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Create useful features and prepare the data for modeling."""
    df = raw_df.copy()

    # Extract title from the passenger name.
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(
        ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"],
        "Rare",
    )
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    # Cabin letter is often informative and can be treated as categorical.
    df["Cabin"] = df["Cabin"].fillna("U").str[0]

    # Create family-based features.
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Fill missing values with appropriate statistics.
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Keep an interpretable subset of columns for modeling.
    feature_columns = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Title",
        "Cabin",
        "FamilySize",
        "IsAlone",
    ]

    prepared = df[feature_columns + [TARGET_COLUMN]].copy()
    prepared["Sex"] = prepared["Sex"].astype(str)
    prepared["Embarked"] = prepared["Embarked"].astype(str)
    prepared["Title"] = prepared["Title"].astype(str)
    prepared["Cabin"] = prepared["Cabin"].astype(str)

    return prepared, feature_columns


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    """Create a preprocessing pipeline for numeric and categorical features."""
    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
    categorical_features = [col for col in feature_columns if col not in numeric_features]

    try:
        categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", categorical_encoder),
                    ]
                ),
                categorical_features,
            ),
        ],
        verbose_feature_names_out=False,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None) -> dict[str, float]:
    """Calculate evaluation metrics for both binary and multiclass settings."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    if y_prob is not None:
        if len(np.unique(y_true)) == 2:
            metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            metrics["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    else:
        metrics["auc"] = float("nan")

    return metrics


def train_and_evaluate() -> None:
    """Train all requested models and save them to disk."""
    raw_data = load_dataset()
    prepared_data, feature_columns = engineer_features(raw_data)

    X = prepared_data[feature_columns]
    y = prepared_data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Save a sample of the holdout set so the Streamlit app can use it as a demo dataset.
    evaluation_frame = X_test.copy()
    evaluation_frame[TARGET_COLUMN] = y_test
    evaluation_frame.to_csv(ROOT / "test_data.csv", index=False)

    # Save metadata so the app knows which columns are expected.
    metadata = {"features": feature_columns, "target": TARGET_COLUMN}
    with (ROOT / "model_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = []

    for name, model in models.items():
        preprocessor = build_preprocessor(feature_columns)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)
        metrics = compute_metrics(y_test.to_numpy(), predictions, probabilities)

        model_path = MODELS_DIR / f"{name}.pkl"
        joblib.dump(pipeline, model_path)

        results.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "auc": metrics["auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "mcc": metrics["mcc"],
            }
        )

        # Save a human-readable confusion matrix and report alongside the models.
        cm = confusion_matrix(y_test, predictions)
        np.save(MODELS_DIR / f"{name}_confusion_matrix.npy", cm)
        report_text = classification_report(y_test, predictions, zero_division=0)
        (MODELS_DIR / f"{name}_report.txt").write_text(report_text, encoding="utf-8")

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(ROOT / "model_comparison.csv", index=False)

    print("Model training finished successfully.")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    train_and_evaluate()
