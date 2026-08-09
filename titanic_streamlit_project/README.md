# Titanic Survival Classification with Streamlit

**BITS WILP M.Tech (AIML/DSE) - Machine Learning Assignment 2**

---

## a. Problem Statement

The objective of this project is to build a **binary classification model** that predicts whether a passenger aboard the RMS Titanic survived or not based on their demographic and travel information. The prediction task involves analyzing passenger attributes such as age, sex, ticket class, fare, embarkation port, and family information to determine survival likelihood. This is a classic supervised learning problem with practical applications in understanding historical data patterns and validating machine learning methodologies.

The project demonstrates the complete machine learning pipeline: **data exploration → preprocessing → model training → evaluation → deployment**, culminating in an interactive Streamlit web application deployed on Streamlit Community Cloud.

---

## b. Dataset Description

**Dataset Name:** Titanic Passenger Survival Dataset  
**Source:** Public GitHub Mirror of Kaggle Titanic Dataset  
**URL:** https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

**Dataset Characteristics:**
- **Total Instances:** 891 passengers
- **Original Features:** 12 (PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked, Survived)
- **Engineered Features:** 11 key features
- **Target Variable:** Survived (Binary: 0 = Did not survive, 1 = Survived)
- **Problem Type:** Binary Classification
- **Class Distribution:** Imbalanced (~62% Did Not Survive, ~38% Survived)

**Features Used:**
1. **Pclass** (Passenger Class): 1st, 2nd, or 3rd class ticket
2. **Sex**: Male or Female
3. **Age**: Age in years
4. **SibSp**: Number of siblings/spouses aboard
5. **Parch**: Number of parents/children aboard
6. **Fare**: Ticket fare in pounds sterling
7. **Embarked**: Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)
8. **Title**: Passenger title extracted from name (Mr, Mrs, Miss, etc.)
9. **Cabin**: Cabin letter (first character)
10. **FamilySize**: Total family members aboard (SibSp + Parch + 1)
11. **IsAlone**: Binary indicator for traveling alone

**Data Preprocessing Steps:**
- Extracted passenger title from names
- Created family-size and alone-status features
- Imputed missing Age values with median
- Imputed missing Fare values with median
- Imputed missing Embarked values with mode
- Applied StandardScaler to numerical features
- Applied OneHotEncoder to categorical features

---

## c. GitHub Repository Link

**Repository:** https://github.com/[your-username]/titanic-streamlit-classification

*Note: Push this project to your GitHub account before submission.*

---

## d. Models Used & Evaluation Metrics

### Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|----------|----------|-----|-----------|--------|----------|-----|
| **Logistic Regression** | 0.8436 | 0.8733 | 0.8426 | 0.8436 | 0.8421 | 0.6661 |
| **Decision Tree** | 0.7821 | 0.7551 | 0.7797 | 0.7821 | 0.7797 | 0.5331 |
| **k-Nearest Neighbor (KNN)** | 0.8212 | 0.8572 | 0.8198 | 0.8212 | 0.8195 | 0.6180 |
| **Naive Bayes (Gaussian)** | 0.7654 | 0.7802 | 0.7946 | 0.7654 | 0.7685 | 0.5492 |
| **Random Forest (Ensemble)** | 0.8045 | 0.8300 | 0.8032 | 0.8045 | 0.8036 | 0.5843 |

---

### Model Observations & Performance Analysis

| ML Model | Observation about Model Performance |
|----------|--------------------------------------|
| **Logistic Regression** | **Strongest Performer:** Achieved the highest accuracy (84.36%) and AUC score (0.8733). The linear model generalizes well to the Titanic dataset, demonstrating that passenger survival has strong linear patterns. High precision (84.26%) indicates low false positive rate. The model is highly interpretable and efficient for deployment. |
| **Decision Tree** | **Lower Performance:** Achieved 78.21% accuracy with the lowest AUC (0.7551). Decision trees tend to overfit on structured tabular data like this, capturing noise rather than true patterns. The model lacks robustness but provides excellent interpretability through feature importance analysis. |
| **k-Nearest Neighbor (KNN)** | **Second Best:** Achieved 82.12% accuracy with strong AUC (0.8572). KNN performs well due to proper feature scaling in the preprocessing pipeline. The model captures local patterns effectively but is computationally expensive during inference (O(n) complexity). Memory-intensive for production deployment. |
| **Naive Bayes (Gaussian)** | **Moderate Performance:** Achieved 76.54% accuracy with AUC of 0.7802. The model makes independence assumption between features which is violated in the Titanic dataset (e.g., Fare and Pclass are correlated). Despite this, it remains fast and reliable. Performs adequately for quick prototyping. |
| **Random Forest (Ensemble)** | **Robust & Reliable:** Achieved 80.45% accuracy with AUC of 0.8300. While not the absolute best, it provides excellent balance between accuracy and robustness. Random Forest handles feature interactions well, is resistant to overfitting, and captures non-linear patterns. Ideal for production due to stability and feature importance insights. |

---

### Overall Winner for This Dataset

**🏆 Best Model: Logistic Regression**

**Justification:**
- **Highest Accuracy (84.36%)** and **Best AUC Score (0.8733)** among all models
- **Excellent Precision (84.26%)** with minimal false positives
- **Computational Efficiency:** Fastest inference time, minimal memory footprint
- **Production Ready:** Easily interpretable, explainable predictions for stakeholders
- **Deployability:** Lightweight model ideal for Streamlit Community Cloud constraints
- **Generalization:** High recall (84.36%) ensures good coverage of positive class

**Recommendation:** For this Titanic dataset, **Logistic Regression** is the optimal choice due to its superior predictive performance combined with deployment efficiency and model interpretability.

---

## Project Structure

```
project-folder/
├── app.py                              # Streamlit web application
├── train_models.py                     # Model training & evaluation script
├── requirements.txt                    # Python dependencies
├── README.md                           # This file (project documentation)
├── test_data.csv                       # Test dataset (holdout set)
├── model_metadata.json                 # Feature metadata
├── model_comparison.csv                # Metrics comparison export
└── models/                             # Saved trained models
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── logistic_regression_confusion_matrix.npy
    ├── decision_tree_confusion_matrix.npy
    ├── knn_confusion_matrix.npy
    ├── naive_bayes_confusion_matrix.npy
    └── random_forest_confusion_matrix.npy
```

---

## Installation Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/[your-username]/titanic-streamlit-classification.git
   cd titanic-streamlit-classification
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate          # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run Locally

### Step 1: Train All Models
```bash
python train_models.py
```
This script will:
- Download the Titanic dataset from public GitHub
- Perform feature engineering and preprocessing
- Train 5 classification models
- Save all models to `models/` directory
- Generate `test_data.csv` and `model_metadata.json`
- Display comparison metrics table

### Step 2: Launch Streamlit App
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`

### Usage in the App:
1. **Upload CSV:** Use the sidebar to upload test data (or use default test_data.csv)
2. **Select Model:** Choose from 5 trained models
3. **View Results:** See predictions, metrics, confusion matrix, and classification report

---

## Streamlit Community Cloud Deployment Instructions

### Prerequisites
- GitHub account with the repository pushed
- Streamlit Community Cloud account (free)

### Deployment Steps

1. **Go to Streamlit Cloud:** https://streamlit.io/cloud

2. **Sign In:** Click "Sign in with GitHub" and authorize Streamlit

3. **Create New App:**
   - Click "New app"
   - Select your GitHub repository
   - Choose branch: `main`
   - Set main file path: `app.py`

4. **Deploy:**
   - Click "Deploy"
   - Wait for deployment to complete
   - Share the live link with stakeholders

**Live Streamlit App Link:** [Will be provided after deployment]

---

## Key Features of the Streamlit Application

✅ **Dataset Upload:** Users can upload their own test CSV or use provided test_data.csv  
✅ **Model Selection:** Dropdown menu to select any of 5 trained models  
✅ **Predictions Display:** Table showing actual vs predicted values  
✅ **Evaluation Metrics:** Shows Accuracy, AUC, Precision, Recall, F1, MCC  
✅ **Confusion Matrix:** Interactive heatmap visualization  
✅ **Classification Report:** Detailed precision, recall, F1 per class  
✅ **Error Handling:** Validates uploaded files and missing columns  
✅ **Responsive Layout:** Clean sidebar controls and wide-format display  

---

## Dependencies

See `requirements.txt` for complete list:
- **streamlit** >= 1.35.0
- **scikit-learn** >= 1.4.0
- **pandas** >= 2.2.0
- **numpy** >= 1.26.0
- **matplotlib** >= 3.8.0
- **seaborn** >= 0.13.0
- **joblib** >= 1.4.0

---

## Author & Submission

**BITS WILP ID:** [Your ID]  
**Assignment:** Machine Learning Assignment 2  
**Submission Date:** August 18, 2026  
**Deadline:** 18-Aug-2026 23:59 PM

---

## Acknowledgments

- Dataset sourced from Kaggle Titanic Competition
- Built with Streamlit framework for interactive ML deployment
- Scikit-learn for model implementation and evaluation
