# Titanic Survival Classification with Streamlit
**BITS ID:** 2025AC05631
# Titanic Survival Classification with Streamlit

**BITS WILP M.Tech (AIML/DSE) - Machine Learning Assignment 2**  
**BITS WILP ID:** 2025AC05631

---

## a. Problem Statement

Build a binary classification model that predicts whether a Titanic passenger survived (Survived = 1) or not (Survived = 0), using demographic and travel attributes. The project covers the complete end-to-end ML workflow: data preprocessing, feature engineering, training five classifiers, evaluating them with six standard metrics, selecting the best model, and deploying an interactive Streamlit app on Streamlit Community Cloud.

---

## b. Dataset Description

| Item | Details |
|------|---------|
| **Dataset Name** | Titanic Passenger Survival Dataset |
| **Source** | Public GitHub mirror of the Kaggle Titanic dataset |
| **URL** | https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv |
| **Instances** | 891 (requirement: >= 500) |
| **Original Features** | 12 columns (PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked) |
| **Engineered Features Used** | 11 (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title, Cabin, FamilySize, IsAlone) |
| **Target** | Survived (0 = did not survive, 1 = survived) |
| **Problem Type** | Binary Classification |
| **Class Balance** | ~62% did not survive, ~38% survived (mild imbalance) |
| **Train/Test Split** | 80/20 stratified, random_state = 42 (712 train / 179 test) |

**Preprocessing pipeline:** title extraction from Name, FamilySize and IsAlone engineering, median imputation for Age/Fare, mode imputation for Embarked, StandardScaler for numeric features, and OneHotEncoder for categorical features. All steps are wrapped in a scikit-learn Pipeline so the exact same transforms are applied at inference time.

---

## c. GitHub Repository Link

https://github.com/ayushi-awasthi-14/ML-Assignment-

---

## d. Models Used

### Comparison Table (evaluated on the 179-row hold-out test set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.8436 | 0.8733 | 0.8426 | 0.8436 | 0.8421 | 0.6661 |
| Decision Tree | 0.7821 | 0.7551 | 0.7797 | 0.7821 | 0.7797 | 0.5331 |
| kNN | 0.8212 | 0.8572 | 0.8198 | 0.8212 | 0.8195 | 0.6180 |
| Naive Bayes | 0.7654 | 0.7812 | 0.7946 | 0.7654 | 0.7685 | 0.5492 |
| Random Forest (Ensemble) | 0.8045 | 0.8300 | 0.8032 | 0.8045 | 0.8036 | 0.5843 |

All six metrics (Accuracy, AUC, Precision, Recall, F1, MCC) are reported for each of the five models. Metrics are computed on the same stratified hold-out split for a fair comparison.

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Best overall model on this dataset: highest Accuracy (0.8436), AUC (0.8733) and MCC (0.6661). Survival on the Titanic is largely driven by a few near-linear signals (Sex, Pclass, Title, Fare), which a linear decision boundary captures very well after scaling. The strong AUC shows well-calibrated ranking of survivors vs non-survivors, and the highest MCC confirms balanced performance across both classes despite the ~62/38 imbalance. It is also the most interpretable and lightest model to deploy. |
| **Decision Tree** | Weakest ranker: lowest AUC (0.7551) and lowest MCC (0.5331). A single unpruned tree overfits the training folds and splits on noise, so it generalises worst on the hold-out set even though train accuracy is high. Its hard, axis-aligned splits give poor probability estimates (hence low AUC). Still valuable for interpretability and feature-importance insight, but not competitive here in raw predictive quality. |
| **kNN** | Second-best model (Accuracy 0.8212, AUC 0.8572). Benefits strongly from StandardScaler in the pipeline, since distance-based methods are scale-sensitive. It captures local neighbourhood structure (e.g. women in 1st class cluster as survivors) but is memory-heavy and slow at inference because every prediction scans the training set. Sensitive to the choice of k (k = 5 used here). |
| **Naive Bayes** | Lowest Accuracy (0.7654) but highest Precision (0.7946), meaning when it predicts "survived" it is often right, yet it misses many true survivors (lower recall). Gaussian NB assumes conditional feature independence, which is violated here (Fare, Pclass and Cabin are correlated), capping its accuracy. Nonetheless it is extremely fast, needs little data, and is a good baseline. |
| **Random Forest (Ensemble)** | Robust and stable (Accuracy 0.8045, AUC 0.8300). Averaging many de-correlated trees fixes the single Decision Tree's overfitting, lifting AUC from 0.7551 to 0.8300 and MCC from 0.5331 to 0.5843. It models non-linear feature interactions well and is the most reliable non-linear option, but on this small, largely linearly-separable dataset it still trails Logistic Regression. |

### Overall Winner for This Dataset

**Winner: Logistic Regression.** It leads on the three most informative metrics for an imbalanced binary problem — Accuracy (0.8436), AUC (0.8733) and MCC (0.6661) — while also being the fastest, most interpretable, and easiest to deploy. The result confirms that Titanic survival is dominated by near-linear relationships (Sex, Pclass, Title), so a well-regularised linear model beats more complex non-linear models on the hold-out set.

---

## e. Live Streamlit App Link

https://usuaqfp8ejc3vwquzqdw8w.streamlit.app/

---

## Project Structure

```
ML-Assignment-/
├── app.py                  # Streamlit web application
├── train_models.py         # Trains all 5 models and exports metrics
├── utils.py                # Shared feature engineering
├── requirements.txt        # Dependencies (scikit-learn pinned to 1.9.0)
├── README.md               # This file
├── test_data.csv           # Hold-out test set (179 rows)
├── model_metadata.json     # Feature list + target
├── model_comparison.csv    # Metrics table export
└── models/                 # 5 trained pipelines (.pkl) + reports + confusion matrices
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
- CSV upload for test data (sidebar)
- Model selection dropdown (5 classifiers)
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion matrix heatmap and full classification report

---

## Streamlit Cloud Deployment

- Repository: `ayushi-awasthi-14/ML-Assignment-`
- Branch: `main`
- Main file path: `app.py`

---

## Author & Submission

- **BITS WILP ID:** 2025AC05631
- **Assignment:** Machine Learning Assignment 2
- **Deadline:** 18-Aug-2026 23:59
---

## a. Problem Statement# Titanic Survival Classification with Streamlit

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

https://gh8danupwx7a27hhdm7mbm.streamlit.app/

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

## Author & Submission

- **BITS WILP ID:** 2025AC05631
- **Assignment:** Machine Learning Assignment 2
- **Deadline:** 18-Aug-2026 23:59

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
