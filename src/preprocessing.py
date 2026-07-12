
#This code module works on preprocessing the data by cleaning, normalizing and feature scalingp

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

TARGET_COLUMN = "diagnosis"
BINARY_TARGET_COLUMN = "target"

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_and_binarize(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    for col in CATEGORICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[BINARY_TARGET_COLUMN] = (df[TARGET_COLUMN] > 0).astype(int)
    df = df.drop(columns=[TARGET_COLUMN])
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def get_feature_target_split(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df[BINARY_TARGET_COLUMN]
    return X, y
