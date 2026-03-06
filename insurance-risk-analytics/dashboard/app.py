"""Streamlit dashboard for insurance risk analytics."""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def load_data():
    """Load processed dataset or generate a synthetic fallback."""
    candidates = [
        Path("data/processed/modeling_dataset.csv"),
        Path("data/processed/portfolio.csv"),
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)

    rng = np.random.default_rng(42)
    size = 500
    df = pd.DataFrame({
        "premium": rng.normal(1200, 250, size).clip(200),
        "number_of_claims": rng.poisson(0.4, size),
        "claim_amount": rng.gamma(2.0, 800, size),
        "expected_claim_frequency": rng.normal(0.4, 0.1, size).clip(0),
        "expected_claim_severity": rng.normal(1600, 400, size).clip(0),
        "fraud_probability": rng.uniform(0, 0.3, size),
        "risk_segment": rng.choice(["Low", "Medium", "High"], size, p=[0.6, 0.3, 0.1]),
        "region": rng.choice(["North", "South", "East", "West"], size),
    })
    df["pure_premium"] = df["expected_claim_frequency"] * df["expected_claim_severity"]
    df["expected_loss_ratio"] = df["pure_premium"] / df["premium"]
    return df


def main():
    st.set_page_config(page_title="Insurance Risk Analytics", layout="wide")
    st.title("Insurance Risk Analytics Dashboard")

    df = load_data()

    st.sidebar.header("Filters")
    if "region" in df.columns:
        region = st.sidebar.multiselect("Region", sorted(df["region"].unique()), default=list(df["region"].unique()))
        df = df[df["region"].isin(region)]

    if "risk_segment" in df.columns:
        segment = st.sidebar.multiselect(
            "Risk Segment",
            sorted(df["risk_segment"].unique()),
            default=list(df["risk_segment"].unique()),
        )
        df = df[df["risk_segment"].isin(segment)]

    st.subheader("Portfolio Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Policies", f"{len(df):,}")
    if "premium" in df.columns:
        col2.metric("Total Premium", f"{df['premium'].sum():,.0f}")
    if "pure_premium" in df.columns:
        col3.metric("Total Pure Premium", f"{df['pure_premium'].sum():,.0f}")
    if "expected_loss_ratio" in df.columns:
        col4.metric("Avg Loss Ratio", f"{df['expected_loss_ratio'].mean():.2%}")

    st.subheader("Premium Analysis")
    if "premium" in df.columns:
        fig_premium = px.histogram(df, x="premium", nbins=30, title="Premium Distribution")
        st.plotly_chart(fig_premium, use_container_width=True)

    st.subheader("Risk Segmentation")
    if "risk_segment" in df.columns:
        seg_counts = df["risk_segment"].value_counts().reset_index()
        seg_counts.columns = ["risk_segment", "count"]
        fig_seg = px.bar(seg_counts, x="risk_segment", y="count", title="Portfolio by Risk Segment")
        st.plotly_chart(fig_seg, use_container_width=True)

    st.subheader("Fraud Analytics")
    if "fraud_probability" in df.columns:
        fig_fraud = px.box(df, y="fraud_probability", title="Fraud Probability Distribution")
        st.plotly_chart(fig_fraud, use_container_width=True)

    st.subheader("Pricing Simulation")
    multiplier = st.slider("Severity Stress Multiplier", 0.8, 1.5, 1.0, 0.05)
    if "expected_claim_frequency" in df.columns and "expected_claim_severity" in df.columns:
        simulated_premium = df["expected_claim_frequency"] * df["expected_claim_severity"] * multiplier
        fig_sim = px.histogram(simulated_premium, nbins=30, title="Simulated Pure Premium")
        st.plotly_chart(fig_sim, use_container_width=True)


if __name__ == "__main__":
    main()
