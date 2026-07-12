"""
Unit tests for src/preprocessing.py
"""

import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import (  # noqa: E402
    clean_and_binarize,
    build_preprocessor,
    get_feature_target_split,
    FEATURE_COLUMNS,
    BINARY_TARGET_COLUMN,
)


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame(
        {
            "age": [63, 45, 58],
            "sex": [1, 0, 1],
            "cp": [1, 2, 3],
            "trestbps": [145, 130, 120],
            "chol": [233, 250, 200],
            "fbs": [1, 0, 0],
            "restecg": [2, 0, 1],
            "thalach": [150, 187, 172],
            "exang": [0, 0, 1],
            "oldpeak": [2.3, 3.5, 1.4],
            "slope": [3, 0, 2],
            "ca": ["0.0", "?", "1.0"],
            "thal": ["6.0", "3.0", "?"],
            "diagnosis": [0, 2, 0],
        }
    )


def test_clean_and_binarize_creates_binary_target(sample_raw_df):
    df = clean_and_binarize(sample_raw_df)
    assert BINARY_TARGET_COLUMN in df.columns
    assert "diagnosis" not in df.columns
    assert set(df[BINARY_TARGET_COLUMN].unique()).issubset({0, 1})
    # diagnosis 0 -> 0 ; diagnosis 2 -> 1
    assert df[BINARY_TARGET_COLUMN].tolist() == [0, 1, 0]


def test_clean_and_binarize_handles_missing_markers(sample_raw_df):
    df = clean_and_binarize(sample_raw_df)
    # '?' should become NaN in ca/thal after numeric coercion
    assert df["ca"].isna().sum() == 1
    assert df["thal"].isna().sum() == 1


def test_feature_target_split_shapes(sample_raw_df):
    df = clean_and_binarize(sample_raw_df)
    X, y = get_feature_target_split(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y) == 3


def test_preprocessor_fits_and_transforms(sample_raw_df):
    df = clean_and_binarize(sample_raw_df)
    X, y = get_feature_target_split(df)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(X)
    import numpy as np
    arr = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert not np.isnan(arr).any()
