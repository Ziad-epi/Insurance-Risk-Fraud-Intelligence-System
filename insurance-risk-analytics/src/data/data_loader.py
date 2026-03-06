"""Data loading utilities for the insurance analytics project."""

from pathlib import Path
import pandas as pd


def _load_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return pd.read_csv(path)


def load_policy_data(raw_dir="data/raw", filename="policy_master.csv"):
    """Load policy master data."""
    return _load_csv(Path(raw_dir) / filename)


def load_claim_data(raw_dir="data/raw", filename="claim_master.csv"):
    """Load claim master data."""
    return _load_csv(Path(raw_dir) / filename)
