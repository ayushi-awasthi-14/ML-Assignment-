# Titanic Survival Classification with Streamlit

**BITS WILP M.Tech (AIML/DSE) - Machine Learning Assignment 2**  
**BITS WILP ID:** 2025AC05631

---

## a. Problem Statement

Build a binary classification model that predicts whether a Titanic passenger survived (Survived = 1) or not (Survived = 0), using demographic and travel attributes. The project covers the end-to-end ML workflow: data preprocessing, training multiple classifiers, evaluating them with standard metrics, and deploying an interactive Streamlit app on Streamlit Community Cloud.

---

## b. Dataset Description

| Item | Details |
|------|---------|
| **Dataset Name** | Titanic Passenger Survival Dataset |
| **Source** | Public GitHub mirror of Kaggle Titanic dataset |
| **URL** | https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv |
| **Instances** | 891 (requirement: >= 500) |
| **Original Features** | 12 columns (PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked) |
| **Engineered Features Used** | 11 (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title, Cabin, FamilySize, IsAlone) |
| **Target** | Survived (0 / 1) |
| **Problem Type** | Binary Classification |

**Preprocessing:** title extraction from Name, FamilySize / IsAlone engineering, median imputation for Age/Fare, mode imputation for Embarked, StandardScaler for numeric features, OneHotEncoder for categorical features.

---

## c. GitHub Repository Link

https://github.com/ayushi-awasthi-14/ML-Assignment-

---

## d. Models Used

### Comparison Table (hold-out test set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.8436 | 0.8733 | 0.8426 | 0.8436 | 0.8421 | 0.6661 |
| Decision Tree | 0.7821 | 0.7551 | 0.7797 | 0.7821 | 0.7797 | 0.5331 |
| kNN | 0.8212 | 0.8572 | 0.8198 | 0.8212 | 0.8195 | 0.6180 |
| Naive Bayes | 0.7654 | 0.7812 | 0.7946 | 0.7654 | 0.7685 | 0.5492 |
| Random Forest (Ensemble) | 0.8045 | 0.8300 | 0.8032 | 0.8045 | 0.8036 | 0.5843 |

### Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Best accuracy and AUC. Fast, interpretable, and well suited to this dataset. |
| Decision Tree | Lowest AUC; tends to overfit tabular patterns; still easy to interpret. |
| kNN | Strong second place after scaling; slower and memory-heavier at inference. |
| Naive Bayes | Moderate results; feature independence assumption is violated (e.g. Fare vs Pclass). |
| Random Forest (Ensemble) | Stable ensemble with good non-linear capacity; slightly below Logistic Regression here. |
| **Overall Winner** | **Logistic Regression** (highest Accuracy 0.8436 and AUC 0.8733) |

---

## e. Live Streamlit App Link

https://usuaqfp8ejc3vwquzqdw8w.streamlit.app/

---

## Project Structure

```
ML-Assignment-/
├── app.py
├── train_models.py
├── utils.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_metadata.json
├── model_comparison.csv
└── models/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
```

---

## How to Run Locally

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

App opens at http://localhost:8501

### Streamlit App Features
- CSV upload for test data
- Model selection dropdown (5 classifiers)
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion matrix and classification report

---

## Streamlit Cloud Deployment

- Repository: `ayushi-awasthi-14/ML-Assignment-`
- Branch: `main`
- Main file path: `app.py`

---
