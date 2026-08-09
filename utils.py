from __future__ import annotations

from typing import Iterable

import pandas as pd


DEFAULT_FEATURE_COLUMNS = [
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


def engineer_features(raw_df: pd.DataFrame, feature_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Create the engineered Titanic feature set expected by the trained models."""
    df = raw_df.copy()
    feature_columns = list(feature_columns or DEFAULT_FEATURE_COLUMNS)

    # Basic cleaning and type conversion.
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    if "Fare" in df.columns:
        df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")
    if "SibSp" in df.columns:
        df["SibSp"] = pd.to_numeric(df["SibSp"], errors="coerce").fillna(0)
    if "Parch" in df.columns:
        df["Parch"] = pd.to_numeric(df["Parch"], errors="coerce").fillna(0)
    if "Pclass" in df.columns:
        df["Pclass"] = pd.to_numeric(df["Pclass"], errors="coerce")

    # Fill missing numeric values with simple defaults.
    age_median = df["Age"].median() if "Age" in df.columns else 28.0
    fare_median = df["Fare"].median() if "Fare" in df.columns else 32.2
    df["Age"] = df["Age"].fillna(age_median)
    df["Fare"] = df["Fare"].fillna(fare_median)

    if "Embarked" in df.columns:
        embarked_mode = df["Embarked"].mode(dropna=True)
        if not embarked_mode.empty:
            df["Embarked"] = df["Embarked"].fillna(embarked_mode.iloc[0])
        else:
            df["Embarked"] = df["Embarked"].fillna("S")
    else:
        df["Embarked"] = "S"

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].fillna("missing")
    else:
        df["Sex"] = "missing"

    # Extract title from name when available.
    if "Title" not in df.columns:
        if "Name" in df.columns:
            df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        else:
            df["Title"] = "Unknown"

    df["Title"] = df["Title"].fillna("Unknown")
    df["Title"] = df["Title"].replace(
        ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"],
        "Rare",
    )
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    # Cabin letter.
    if "Cabin" in df.columns:
        df["Cabin"] = df["Cabin"].fillna("U").astype(str).str[0].str.upper()
    else:
        df["Cabin"] = "U"

    # Family features.
    if "FamilySize" not in df.columns:
        sibsp = pd.to_numeric(df["SibSp"], errors="coerce").fillna(0) if "SibSp" in df.columns else 0
        parch = pd.to_numeric(df["Parch"], errors="coerce").fillna(0) if "Parch" in df.columns else 0
        df["FamilySize"] = sibsp + parch + 1
    if "IsAlone" not in df.columns:
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Ensure all required columns exist.
    for column in feature_columns:
        if column not in df.columns:
            if column in {"Sex", "Embarked", "Title", "Cabin"}:
                df[column] = "missing"
            elif column in {"Pclass", "SibSp", "Parch", "Age", "Fare", "FamilySize", "IsAlone"}:
                df[column] = 0

    # Final type cleanup.
    for column in ["Sex", "Embarked", "Title", "Cabin"]:
        if column in df.columns:
            df[column] = df[column].astype(str)

    return df[feature_columns]
