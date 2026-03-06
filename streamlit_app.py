import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="Insurance Risk Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("Insurance Risk Analytics & Pricing Dashboard")


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load the policy-level master dataset from CSV.
    """
    base_dir = Path(__file__).resolve().parent
    candidate_paths = [
        base_dir / "insurance-risk-analytics" / "data" / "processed" / "modeling_dataset.csv",
        base_dir / "insurance-risk-analytics" / "data" / "raw" / "policies.csv",
        base_dir / "insurance-risk-analytics" / "data" / "raw" / "policy_master.csv",
        base_dir / "policies.csv",
        base_dir / "policy_master.csv",
    ]
    data_path = next((p for p in candidate_paths if p.exists()), None)
    if data_path is None:
        expected = "\n".join(str(p) for p in candidate_paths)
        raise FileNotFoundError(f"Could not find dataset. Tried:\n{expected}")

    df = pd.read_csv(data_path)
    return _add_display_columns(df)


def _add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure dashboard-friendly columns exist, even when data is one-hot encoded.
    """
    df = df.copy()

    if "annual_premium" not in df.columns and "premium" in df.columns:
        df["annual_premium"] = df["premium"]

    if "expected_loss_ratio" not in df.columns:
        if "expected_claim_frequency" in df.columns and "expected_claim_severity" in df.columns:
            expected_loss = df["expected_claim_frequency"] * df["expected_claim_severity"]
            denom = df["annual_premium"] if "annual_premium" in df.columns else 1.0
            df["expected_loss_ratio"] = expected_loss / denom.replace(0, np.nan)

    if "fraud_flag" not in df.columns and "fraud_label" in df.columns:
        df["fraud_flag"] = df["fraud_label"]

    if "vehicle_type" not in df.columns:
        vehicle_cols = [c for c in df.columns if c.startswith("vehicle_type_")]
        if vehicle_cols:
            vehicle_values = df[vehicle_cols]
            max_col = vehicle_values.idxmax(axis=1)
            df["vehicle_type"] = max_col.str.replace("vehicle_type_", "", regex=False)
            df.loc[vehicle_values.max(axis=1) == 0, "vehicle_type"] = "unknown"
        else:
            df["vehicle_type"] = "unknown"

    if "region" not in df.columns:
        df["region"] = "all"

    if "risk_segment" not in df.columns:
        segment_source = None
        for col in ["expected_loss_ratio", "pure_premium", "claim_to_premium_ratio"]:
            if col in df.columns and df[col].nunique(dropna=True) >= 3:
                segment_source = col
                break
        if segment_source:
            df["risk_segment"] = pd.qcut(
                df[segment_source].rank(method="first"),
                q=3,
                labels=["Low", "Medium", "High"],
            )
        else:
            df["risk_segment"] = "Medium"

    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply sidebar filters for interactive exploration.
    """
    st.sidebar.header("Filters")

    regions = sorted(df["region"].dropna().unique().tolist()) if "region" in df.columns else []
    vehicle_types = (
        sorted(df["vehicle_type"].dropna().unique().tolist())
        if "vehicle_type" in df.columns
        else []
    )
    segments = (
        sorted(df["risk_segment"].dropna().unique().tolist())
        if "risk_segment" in df.columns
        else []
    )

    selected_regions = st.sidebar.multiselect("Region", regions, default=regions)
    selected_vehicles = st.sidebar.multiselect(
        "Vehicle Type", vehicle_types, default=vehicle_types
    )

    if "age" in df.columns and not df["age"].dropna().empty:
        age_min = int(df["age"].min())
        age_max = int(df["age"].max())
        selected_age = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))
    else:
        selected_age = None

    selected_segments = st.sidebar.multiselect(
        "Risk Segment", segments, default=segments
    )

    filtered = df.copy()

    if "region" in df.columns:
        filtered = filtered[filtered["region"].isin(selected_regions)]
    if "vehicle_type" in df.columns:
        filtered = filtered[filtered["vehicle_type"].isin(selected_vehicles)]
    if selected_age and "age" in df.columns:
        filtered = filtered[
            (filtered["age"] >= selected_age[0]) & (filtered["age"] <= selected_age[1])
        ]
    if "risk_segment" in df.columns:
        filtered = filtered[filtered["risk_segment"].isin(selected_segments)]

    return filtered


def compute_expected_loss(df: pd.DataFrame) -> pd.Series:
    """
    Expected loss is the actuarial expected cost of claims.
    """
    return df["expected_claim_frequency"] * df["expected_claim_severity"]


def display_metrics(df: pd.DataFrame) -> None:
    """
    Display portfolio KPIs for pricing performance.
    """
    expected_loss = compute_expected_loss(df)
    total_premium = df["annual_premium"].sum()
    total_profit = total_premium - expected_loss.sum()
    portfolio_loss_ratio = expected_loss.sum() / total_premium if total_premium > 0 else np.nan

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Policies", f"{len(df):,}")
    c2.metric("Avg Premium", f"{df['annual_premium'].mean():,.2f}")
    c3.metric("Avg Pure Premium", f"{df['pure_premium'].mean():,.2f}")
    c4.metric("Portfolio Loss Ratio", f"{portfolio_loss_ratio:.2f}")
    c5.metric("Total Expected Profit", f"{total_profit:,.2f}")


def show_pricing_analysis(df: pd.DataFrame) -> None:
    """
    Pricing analytics to visualize cost drivers and profitability risk.
    """
    st.subheader("Pricing Analysis")

    fig1 = px.histogram(
        df,
        x="pure_premium",
        nbins=40,
        title="Distribution of Pure Premium",
        color_discrete_sequence=["#4E79A7"],
    )
    st.plotly_chart(fig1, use_container_width=True)

    expected_loss = compute_expected_loss(df)
    fig2 = px.scatter(
        df,
        x="annual_premium",
        y=expected_loss,
        title="Premium vs Expected Loss",
        labels={"annual_premium": "Annual Premium", "y": "Expected Loss"},
        color_discrete_sequence=["#59A14F"],
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.box(
        df,
        x="vehicle_type",
        y="pure_premium",
        title="Pure Premium by Vehicle Type",
        color_discrete_sequence=["#F28E2B"],
    )
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.bar(
        df.groupby("region", as_index=False)["expected_loss_ratio"].mean(),
        x="region",
        y="expected_loss_ratio",
        title="Average Loss Ratio by Region",
        color_discrete_sequence=["#E15759"],
    )
    st.plotly_chart(fig4, use_container_width=True)


def show_risk_segmentation(df: pd.DataFrame) -> None:
    """
    Risk segmentation insights to guide pricing actions.
    """
    st.subheader("Risk Segmentation")

    seg_counts = df["risk_segment"].value_counts().reset_index()
    seg_counts.columns = ["risk_segment", "count"]
    fig1 = px.pie(
        seg_counts,
        names="risk_segment",
        values="count",
        title="Risk Segment Distribution",
    )
    st.plotly_chart(fig1, use_container_width=True)

    seg_premium = df.groupby("risk_segment", as_index=False)["annual_premium"].mean()
    fig2 = px.bar(
        seg_premium,
        x="risk_segment",
        y="annual_premium",
        title="Average Premium per Segment",
        color_discrete_sequence=["#76B7B2"],
    )
    st.plotly_chart(fig2, use_container_width=True)

    seg_loss = df.groupby("risk_segment", as_index=False)["expected_loss_ratio"].mean()
    fig3 = px.bar(
        seg_loss,
        x="risk_segment",
        y="expected_loss_ratio",
        title="Average Loss Ratio per Segment",
        color_discrete_sequence=["#EDC948"],
    )
    st.plotly_chart(fig3, use_container_width=True)


def show_fraud_analytics(df: pd.DataFrame) -> None:
    """
    Fraud analytics if a fraud flag is available.
    """
    if "fraud_flag" not in df.columns:
        st.info("Fraud analytics not available: 'fraud_flag' column missing.")
        return

    st.subheader("Fraud Analytics")

    fraud_region = df.groupby("region", as_index=False)["fraud_flag"].mean()
    fig1 = px.bar(
        fraud_region,
        x="region",
        y="fraud_flag",
        title="Fraud Rate by Region",
        color_discrete_sequence=["#B07AA1"],
    )
    st.plotly_chart(fig1, use_container_width=True)

    fraud_vehicle = df.groupby("vehicle_type", as_index=False)["fraud_flag"].mean()
    fig2 = px.bar(
        fraud_vehicle,
        x="vehicle_type",
        y="fraud_flag",
        title="Fraud Rate by Vehicle Type",
        color_discrete_sequence=["#FF9DA7"],
    )
    st.plotly_chart(fig2, use_container_width=True)

    if "total_claim_amount_per_policy" in df.columns:
        fig3 = px.histogram(
            df,
            x="total_claim_amount_per_policy",
            color="fraud_flag",
            nbins=40,
            title="Claim Amount Distribution: Fraud vs Non-Fraud",
            barmode="overlay",
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Claim amount distribution not available: missing claim amount column.")


def run_pricing_simulation(df: pd.DataFrame) -> None:
    """
    Pricing simulation to estimate impact of premium adjustments.
    """
    st.subheader("Model Simulation")
    premium_multiplier = st.slider("Premium Adjustment (%)", -20, 20, 0)

    expected_loss = compute_expected_loss(df)
    base_premium = df["annual_premium"].sum()
    base_loss_ratio = expected_loss.sum() / base_premium if base_premium > 0 else np.nan
    base_profit = (df["annual_premium"] - expected_loss).sum()

    multiplier = 1 + premium_multiplier / 100
    simulated_premium = df["annual_premium"] * multiplier
    new_profit = (simulated_premium - expected_loss).sum()
    new_loss_ratio = expected_loss.sum() / simulated_premium.sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Base Profit", f"{base_profit:,.2f}")
    c2.metric("New Profit", f"{new_profit:,.2f}")
    c3.metric("Base Loss Ratio", f"{base_loss_ratio:.2f}")
    c4.metric("New Loss Ratio", f"{new_loss_ratio:.2f}")


def show_data_table(df: pd.DataFrame) -> None:
    """
    Display the filtered data and allow export.
    """
    st.subheader("Portfolio Data")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name="policy_master_filtered.csv",
        mime="text/csv",
    )


def main():
    df = load_data()
    filtered = apply_filters(df)

    if filtered.empty:
        st.warning("No data matches the selected filters.")
        return

    display_metrics(filtered)
    show_pricing_analysis(filtered)
    show_risk_segmentation(filtered)
    show_fraud_analytics(filtered)
    run_pricing_simulation(filtered)
    show_data_table(filtered)


if __name__ == "__main__":
    main()
