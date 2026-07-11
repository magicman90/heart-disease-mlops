"""
preprocessing.py
-----------------
Cleaning, target binarization, and a reusable sklearn Pipeline
(ColumnTransformer) for feature scaling/encoding.

The same `build_preprocessor()` pipeline is fit during training and reused
unchanged at inference time (saved alongside the model), so preprocessing
is guaranteed to be identical in both places.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Numeric (continuous) features -> impute median + scale
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Categorical features (encoded as ints in the raw data but are really
# categories) -> impute most frequent + one-hot encode
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

TARGET_COLUMN = "diagnosis"
BINARY_TARGET_COLUMN = "target"

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_and_binarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Coerces the categorical columns that may contain NaN back to numeric
    - Binarizes the multi-class UCI target (0 = no disease, 1-4 = disease
      of increasing severity) into a binary presence/absence label, as
      required by the assignment ("binary target").
    """
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
