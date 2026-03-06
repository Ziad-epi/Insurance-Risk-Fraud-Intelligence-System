"""Preprocessing utilities: cleaning, encoding, and train/test split."""

import pandas as pd
from sklearn.model_selection import train_test_split


def clean_data(df):
    """Clean missing values and remove duplicates.

    Numeric columns are filled with median values.
    Categorical columns are filled with the mode or 'Unknown'.
    """
    df = df.copy()
    df = df.drop_duplicates()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            mode_val = df[col].mode()
            fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
    return df


def encode_features(df, categorical_cols=None, drop_first=True):
    """One-hot encode categorical columns.

    If categorical_cols is None, object and category dtypes are encoded.
    """
    df = df.copy()
    if categorical_cols is None:
        categorical_cols = [
            col
            for col in df.columns
            if df[col].dtype == "object" or str(df[col].dtype).startswith("category")
        ]
    if not categorical_cols:
        return df
    return pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)


def train_test_split_data(df, target_col, test_size=0.2, random_state=42, stratify=False):
    """Split a dataset into train and test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    stratify_y = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_y)
