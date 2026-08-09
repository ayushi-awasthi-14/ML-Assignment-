"""Streamlit app for loading trained Titanic classification models and scoring uploaded data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
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

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
for candidate in [str(ROOT), str(PROJECT_ROOT)]:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from utils import engineer_features

MODELS_DIR = ROOT / "models"
METADATA_PATH = ROOT / "model_metadata.json"
DEFAULT_DATA_PATH = ROOT / "test_data.csv"

MODEL_OPTIONS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest",
}


def load_metadata() -> dict[str, object]:
    """Load saved feature names and target column metadata."""
    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


st.set_page_config(page_title="Titanic Survival Classification", page_icon="🚢", layout="wide")

st.title("🚢 Titanic Survival Classification")
st.markdown(
    "Upload a raw Titanic CSV or use the built-in sample data. The app will engineer the input features before scoring it with your selected model."
)

with st.sidebar:
    st.header("Controls")
    selected_model_label = st.selectbox("Select a classification model", list(MODEL_OPTIONS.keys()))
    uploaded_file = st.file_uploader("Upload Titanic CSV", type=["csv"])
    st.caption("The app accepts a raw Titanic-style CSV with columns such as Pclass, Name, Sex, Age, SibSp, Parch, Fare, Embarked, and Survived.")

metadata = load_metadata()
feature_columns = metadata["features"]
target_column = metadata["target"]

if uploaded_file is not None:
    try:
        input_frame = pd.read_csv(uploaded_file)
    except Exception as exc:  # pragma: no cover - defensive handling
        st.error(f"Could not read the uploaded file: {exc}")
        st.stop()
else:
    input_frame = pd.read_csv(DEFAULT_DATA_PATH)

if target_column not in input_frame.columns:
    st.error(f"The uploaded file must contain a target column named {target_column}.")
    st.stop()

prepared_frame = engineer_features(input_frame, feature_columns)
X = prepared_frame[feature_columns]
y_true = input_frame[target_column].astype(int)

model_name = MODEL_OPTIONS[selected_model_label]
model_path = MODELS_DIR / f"{model_name}.pkl"

if not model_path.exists():
    st.error(f"The selected model file was not found at {model_path}.")
    st.stop()

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

metrics_df = pd.DataFrame(
    {
        "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
        "Value": [accuracy, auc_value, precision, recall, f1, mcc],
    }
)

st.success(f"Loaded the {selected_model_label} model and generated predictions.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Predictions")
    preview = pd.DataFrame({"Actual": y_true, "Prediction": predictions})
    st.dataframe(preview.head(20), use_container_width=True)

with col2:
    st.subheader("Evaluation Metrics")
    st.dataframe(metrics_df, use_container_width=True)

st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(
    confusion_matrix(y_true, predictions),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,
    ax=ax,
)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("Classification Report")
st.text(classification_report(y_true, predictions, zero_division=0))
