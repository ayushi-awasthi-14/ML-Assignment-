"""Shared feature engineering for Titanic dataset."""
import pandas as pd

TARGET_COLUMN = "Survived"

def engineer_features(raw_df: pd.DataFrame):
    df = raw_df.copy()
    if "Title" not in df.columns and "Name" in df.columns:
        df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        df["Title"] = df["Title"].replace(
            ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"],
            "Rare",
        )
        df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    if "Cabin" in df.columns:
        df["Cabin"] = df["Cabin"].fillna("U").astype(str).str[0]
    else:
        df["Cabin"] = "U"
    if "FamilySize" not in df.columns:
        df["FamilySize"] = df.get("SibSp", 0) + df.get("Parch", 0) + 1
    if "IsAlone" not in df.columns:
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode().iloc[0])
    feature_columns = [
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
        "Embarked", "Title", "Cabin", "FamilySize", "IsAlone",
    ]
    for col in ["Sex", "Embarked", "Title", "Cabin"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df, feature_columns
