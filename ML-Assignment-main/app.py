"""Streamlit app for Titanic survival classification"""
from __future__ import annotations

import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from utils import engineer_features

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
METADATA_PATH = ROOT / "model_metadata.json"
DEFAULT_DATA_PATH = ROOT / "test_data.csv"
COMPARISON_PATH = ROOT / "model_comparison.csv"

MODEL_OPTIONS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest",
}

REQUIRED_FILES = [
    METADATA_PATH,
    DEFAULT_DATA_PATH,
    MODELS_DIR / "logistic_regression.pkl",
    MODELS_DIR / "decision_tree.pkl",
    MODELS_DIR / "knn.pkl",
    MODELS_DIR / "naive_bayes.pkl",
    MODELS_DIR / "random_forest.pkl",
]

def load_metadata() -> dict:
    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def prepare_uploaded_data(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    if all(col in frame.columns for col in feature_columns):
        return frame
    if "Name" in frame.columns or "Pclass" in frame.columns:
        engineered, _ = engineer_features(frame)
        return engineered
    raise ValueError(
        "Upload test_data.csv from this project, or a raw Titanic CSV with columns like Name, Pclass, Sex."
    )

st.set_page_config(page_title="Titanic Survival Classification", page_icon="🚢", layout="wide")
st.title("🚢 Titanic Survival Classification")
st.markdown("BITS WILP ML Assignment 2 — Upload test CSV, select a model, view metrics.")

missing_files = [str(p) for p in REQUIRED_FILES if not p.exists()]
if missing_files:
    st.error("Required files are missing. Run training first from the project folder.")
    st.code("pip install -r requirements.txt\npython train_models.py")
    st.write("Project folder:", str(ROOT))
    st.write("Missing:")
    for item in missing_files:
        st.write("-", item)
    st.stop()

metadata = load_metadata()
feature_columns = metadata["features"]
target_column = metadata["target"]

with st.sidebar:
    st.header("Controls")
    selected_model_label = st.selectbox("Select a classification model", list(MODEL_OPTIONS.keys()))
    uploaded_file = st.file_uploader("Upload test data CSV", type=["csv"])
    st.caption(f"Project folder: {ROOT}")
    if COMPARISON_PATH.exists():
        st.subheader("All Models (Training Metrics)")
        st.dataframe(pd.read_csv(COMPARISON_PATH), use_container_width=True)

if uploaded_file is not None:
    input_frame = pd.read_csv(uploaded_file)
else:
    input_frame = pd.read_csv(DEFAULT_DATA_PATH)

try:
    prepared_frame = prepare_uploaded_data(input_frame, feature_columns)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

missing_columns = [col for col in feature_columns if col not in prepared_frame.columns]
if missing_columns:
    st.error(f"Missing feature columns: {missing_columns}")
    st.stop()

if target_column not in prepared_frame.columns:
    st.error(f"Missing target column: {target_column}")
    st.stop()

X = prepared_frame[feature_columns]
y_true = prepared_frame[target_column].astype(int)

model_name = MODEL_OPTIONS[selected_model_label]
model_path = MODELS_DIR / f"{model_name}.pkl"
pipeline = joblib.load(model_path)

predictions = pipeline.predict(X)
try:
    probabilities = pipeline.predict_proba(X)
except AttributeError:
    probabilities = None

classes = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(predictions)])))

accuracy = accuracy_score(y_true, predictions)
precision = precision_score(y_true, predictions, average="weighted", zero_division=0)
recall = recall_score(y_true, predictions, average="weighted", zero_division=0)
f1 = f1_score(y_true, predictions, average="weighted", zero_division=0)
mcc = matthews_corrcoef(y_true, predictions)

if probabilities is not None:
    if len(classes) == 2:
        auc_value = roc_auc_score(y_true, probabilities[:, 1])
    else:
        auc_value = roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
else:
    auc_value = float("nan")

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
    "Value": [accuracy, auc_value, precision, recall, f1, mcc],
})

st.success(f"Loaded *{selected_model_label}* on {len(X)} rows.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Predictions")
    st.dataframe(pd.DataFrame({"Actual": y_true, "Prediction": predictions}).head(20), use_container_width=True)

with col2:
    st.subheader("Evaluation Metrics")
    st.dataframe(metrics_df, use_container_width=True)

st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(
    confusion_matrix(y_true, predictions),
    annot=True, fmt="d", cmap="Blues",
    xticklabels=classes, yticklabels=classes, ax=ax,
)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("Classification Report")
st.text(classification_report(y_true, predictions, zero_division=0))
