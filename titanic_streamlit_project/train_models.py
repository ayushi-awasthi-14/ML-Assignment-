"""Train classification models for Titanic dataset."""
from __future__ import annotations

import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from utils import TARGET_COLUMN, engineer_features

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

def build_preprocessor(feature_columns):
    numeric = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
    categorical = [c for c in feature_columns if c not in numeric]
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", enc)]), categorical),
    ], verbose_feature_names_out=False)

def compute_metrics(y_true, y_pred, y_prob):
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_prob is not None:
        m["auc"] = roc_auc_score(y_true, y_prob[:, 1]) if len(np.unique(y_true)) == 2 else roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    else:
        m["auc"] = float("nan")
    return m

def train_and_evaluate():
    raw = pd.read_csv(DATA_URL)
    prepared, feature_columns = engineer_features(raw)
    X, y = prepared[feature_columns], prepared[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    test = X_test.copy()
    test[TARGET_COLUMN] = y_test
    test.to_csv(ROOT / "test_data.csv", index=False)
    
    with (ROOT / "model_metadata.json").open("w", encoding="utf-8") as h:
        json.dump({"features": feature_columns, "target": TARGET_COLUMN}, h, indent=2)
        
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    
    results = []
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", build_preprocessor(feature_columns)), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)
        metrics = compute_metrics(y_test.to_numpy(), pred, prob)
        
        joblib.dump(pipe, MODELS_DIR / f"{name}.pkl")
        results.append({"model": name, **metrics})
        np.save(MODELS_DIR / f"{name}_confusion_matrix.npy", confusion_matrix(y_test, pred))
        (MODELS_DIR / f"{name}_report.txt").write_text(classification_report(y_test, pred, zero_division=0), encoding="utf-8")
        
    df = pd.DataFrame(results)
    df.to_csv(ROOT / "model_comparison.csv", index=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    train_and_evaluate()
