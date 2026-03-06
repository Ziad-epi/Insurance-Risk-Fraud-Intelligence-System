"""Pricing utilities for pure premium estimation."""

import numpy as np


def compute_pure_premium(df, frequency_col="expected_claim_frequency", severity_col="expected_claim_severity", premium_col="premium"):
    """Compute pure premium and expected loss ratio."""
    df = df.copy()
    df["pure_premium"] = df[frequency_col] * df[severity_col]

    if premium_col in df.columns:
        premium = df[premium_col].replace(0, np.nan)
        df["expected_loss_ratio"] = (df["pure_premium"] / premium).fillna(0)
    else:
        df["expected_loss_ratio"] = 0

    return df
