import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_data(base_dir: str = "Dataset"):
    """
    Load all source datasets, parse dates, and print basic diagnostics.
    """
    base_path = Path(base_dir)

    print("[LOAD] Reading CSV files...")
    customers = pd.read_csv(base_path / "customers.csv")
    policies = pd.read_csv(
        base_path / "policies.csv",
        parse_dates=["policy_start_date"],
    )
    claims = pd.read_csv(
        base_path / "claims.csv",
        parse_dates=["claim_date"],
    )
    external_risk_factors = pd.read_csv(base_path / "external_risk_factors.csv")

    datasets = {
        "customers": customers,
        "policies": policies,
        "claims": claims,
        "external_risk_factors": external_risk_factors,
    }

    for name, df in datasets.items():
        print(f"[LOAD] {name} shape: {df.shape}")
        print(df.head(5))
        print("[LOAD] Missing values summary:")
        print(df.isna().sum())
        print("-" * 60)

    return customers, policies, claims, external_risk_factors


def validate_integrity(customers, policies, claims, external_risk_factors):
    """
    Validate referential integrity and detect impossible values.
    """
    print("[VALIDATION] Running integrity checks...")

    # Foreign key checks
    missing_customer_fk = ~policies["customer_id"].isin(customers["customer_id"])
    if missing_customer_fk.any():
        print(
            f"[WARN] policies.customer_id missing in customers: {missing_customer_fk.sum()}"
        )

    missing_policy_fk = ~claims["policy_id"].isin(policies["policy_id"])
    if missing_policy_fk.any():
        print(f"[WARN] claims.policy_id missing in policies: {missing_policy_fk.sum()}")

    # Duplicate primary keys
    if customers["customer_id"].duplicated().any():
        print("[WARN] Duplicate customer_id detected in customers.")
    if policies["policy_id"].duplicated().any():
        print("[WARN] Duplicate policy_id detected in policies.")
    if claims["claim_id"].duplicated().any():
        print("[WARN] Duplicate claim_id detected in claims.")
    if external_risk_factors["region"].duplicated().any():
        print("[WARN] Duplicate region detected in external_risk_factors.")

    # Impossible values
    bad_age = (customers["age"] < 18) | (customers["age"] > 100)
    if bad_age.any():
        print(f"[WARN] Invalid ages (<18 or >100): {bad_age.sum()}")

    negative_claim = claims["claim_amount"] < 0
    if negative_claim.any():
        print(f"[WARN] Negative claim_amount detected: {negative_claim.sum()}")

    bad_bonus = (customers["bonus_malus"] < 0.5) | (customers["bonus_malus"] > 1.5)
    if bad_bonus.any():
        print(f"[WARN] bonus_malus outside [0.5, 1.5]: {bad_bonus.sum()}")

    print("[VALIDATION] Integrity checks completed.")


def build_policy_dataset(customers, policies, claims, external_risk_factors):
    """
    Build a policy-level master dataset with claim aggregates.
    """
    print("[BUILD] Creating policy-level master dataset...")

    policy_master = policies.merge(
        customers,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    policy_master = policy_master.merge(
        external_risk_factors,
        on="region",
        how="left",
        validate="many_to_one",
    )

    if claims.empty:
        claims_agg = pd.DataFrame(
            columns=[
                "policy_id",
                "total_claims_per_policy",
                "total_claim_amount_per_policy",
                "average_claim_amount_per_policy",
                "fraud_count_per_policy",
            ]
        )
    else:
        # Insurance logic: aggregate claim frequency and severity per policy
        claims_agg = (
            claims.groupby("policy_id")
            .agg(
                total_claims_per_policy=("claim_id", "size"),
                total_claim_amount_per_policy=("claim_amount", "sum"),
                average_claim_amount_per_policy=("claim_amount", "mean"),
                fraud_count_per_policy=("fraud_label", "sum"),
            )
            .reset_index()
        )

    policy_master = policy_master.merge(
        claims_agg,
        on="policy_id",
        how="left",
    )

    fill_cols = [
        "total_claims_per_policy",
        "total_claim_amount_per_policy",
        "average_claim_amount_per_policy",
        "fraud_count_per_policy",
    ]
    policy_master[fill_cols] = policy_master[fill_cols].fillna(0)

    # Insurance logic: loss ratio and exposure are core pricing/performance metrics
    policy_master["loss_ratio"] = (
        policy_master["total_claim_amount_per_policy"] / policy_master["annual_premium"]
    )
    policy_master["exposure_years"] = policy_master["policy_duration_months"] / 12.0

    return policy_master


def build_claim_dataset(customers, policies, claims, external_risk_factors):
    """
    Build a claim-level master dataset with enriched policy and customer attributes.
    """
    print("[BUILD] Creating claim-level master dataset...")

    claim_master = claims.merge(
        policies,
        on="policy_id",
        how="left",
        validate="many_to_one",
    )

    claim_master = claim_master.merge(
        customers,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    claim_master = claim_master.merge(
        external_risk_factors,
        on="region",
        how="left",
        validate="many_to_one",
    )

    # Insurance logic: claim to premium ratio flags high-cost claims
    claim_master["claim_to_premium_ratio"] = (
        claim_master["claim_amount"] / claim_master["annual_premium"]
    )
    claim_master["delay_flag"] = (claim_master["claim_report_delay_days"] > 15).astype(
        int
    )

    return claim_master


def print_summary(policy_master, claim_master):
    print("[SUMMARY] Final policy_master shape:", policy_master.shape)
    print("[SUMMARY] Final claim_master shape:", claim_master.shape)

    if not policy_master.empty:
        avg_claims_per_policy = policy_master["total_claims_per_policy"].mean()
    else:
        avg_claims_per_policy = 0.0

    if not claim_master.empty:
        fraud_rate = claim_master["fraud_label"].mean()
    else:
        fraud_rate = 0.0

    print(f"[SUMMARY] Average claims per policy: {avg_claims_per_policy:.4f}")
    print(f"[SUMMARY] Fraud rate: {fraud_rate:.4f}")

    if not policy_master.empty:
        print("[SUMMARY] Loss ratio descriptive stats:")
        print(policy_master["loss_ratio"].describe())


def compute_expected_losses(policy_master: pd.DataFrame) -> pd.DataFrame:
    """
    Compute expected losses at the policy level for pricing impact analysis.
    """
    df = policy_master.copy()

    # Business logic: expected loss is the actuarial expected cost of claims.
    df["expected_loss"] = df["expected_claim_frequency"] * df["expected_claim_severity"]

    portfolio_expected_loss = df["expected_loss"].sum()
    portfolio_premium = df["annual_premium"].sum()

    print("\n=== Expected Losses ===")
    print(f"Portfolio expected loss: {portfolio_expected_loss:,.2f}")
    print(f"Portfolio premium: {portfolio_premium:,.2f}")

    return df


def loss_comparison(policy_master: pd.DataFrame) -> None:
    """
    Compare actual vs predicted losses to validate pricing adequacy.
    """
    df = policy_master.copy()

    actual_loss = df["total_claim_amount_per_policy"].astype(float)
    expected_loss = df["expected_loss"].astype(float)

    # Financial metrics for model validation.
    mae = np.mean(np.abs(actual_loss - expected_loss))
    rmse = np.sqrt(np.mean((actual_loss - expected_loss) ** 2))
    corr = np.corrcoef(actual_loss, expected_loss)[0, 1]

    print("\n=== Loss Comparison ===")
    print(f"MAE: {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"Correlation: {corr:.3f}")

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=expected_loss, y=actual_loss, alpha=0.5, color="#4E79A7")
    max_val = max(expected_loss.max(), actual_loss.max())
    plt.plot([0, max_val], [0, max_val], "k--", linewidth=1)
    plt.title("Actual vs Predicted Loss")
    plt.xlabel("Expected Loss")
    plt.ylabel("Actual Loss")
    plt.tight_layout()
    plt.show()


def portfolio_profitability(policy_master: pd.DataFrame) -> None:
    """
    Compute portfolio profitability using expected losses and premiums.
    """
    df = policy_master.copy()

    # Business logic: profit per policy = premium collected - expected loss.
    df["profit"] = df["annual_premium"] - df["expected_loss"]

    total_profit = df["profit"].sum()
    average_profit = df["profit"].mean()
    total_expected_loss = df["expected_loss"].sum()
    total_premium = df["annual_premium"].sum()
    loss_ratio = total_expected_loss / total_premium if total_premium > 0 else np.nan

    print("\n=== Portfolio Profitability ===")
    print(f"Total profit: {total_profit:,.2f}")
    print(f"Average profit per policy: {average_profit:,.2f}")
    print(f"Portfolio loss ratio (expected): {loss_ratio:.3f}")


def pricing_strategy(policy_master: pd.DataFrame) -> None:
    """
    Analyze profitability by risk segment to inform pricing strategy.
    """
    df = policy_master.copy()

    # Business logic: segment-level analysis guides premium adjustments.
    segment_summary = (
        df.groupby("risk_segment")
        .agg(
            avg_premium=("annual_premium", "mean"),
            avg_expected_loss=("expected_loss", "mean"),
            avg_profit=("profit", "mean"),
        )
        .reset_index()
    )

    print("\n=== Pricing Strategy by Risk Segment ===")
    print(segment_summary.to_string(index=False))

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=segment_summary,
        x="risk_segment",
        y="avg_profit",
        color="#59A14F",
    )
    plt.title("Average Profit by Risk Segment")
    plt.xlabel("Risk Segment")
    plt.ylabel("Average Profit")
    plt.tight_layout()
    plt.show()

    # Pricing actions (business guidance):
    # - Increase premiums for Very High Risk if avg_profit is negative.
    # - Offer discounts or retention incentives for Low Risk if margins allow.


def _resolve_fraud_signal(df: pd.DataFrame):
    """
    Resolve fraud signal column (prediction or proxy) from available fields.
    """
    candidates = [
        "fraud_pred",
        "fraud_prediction",
        "fraud_flag_pred",
        "fraud_score",
        "fraud_probability",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    # Fallback: use observed fraud counts to proxy flagged claims when preds are absent.
    if "fraud_count_per_policy" in df.columns:
        return "fraud_count_per_policy"

    return ""


def fraud_business_impact(policy_master: pd.DataFrame, cost_of_fraud_model: float = 200000.0) -> None:
    """
    Estimate financial impact and ROI of the fraud detection model.
    """
    df = policy_master.copy()
    fraud_signal = _resolve_fraud_signal(df)

    if not fraud_signal:
        print("\n[WARN] No fraud prediction column found. Skipping fraud ROI analysis.")
        return

    # Business logic: flagged policies are assumed to have prevented claim payouts.
    if fraud_signal == "fraud_probability":
        flagged = df[fraud_signal] >= 0.5
    elif fraud_signal == "fraud_count_per_policy":
        flagged = df[fraud_signal] > 0
    else:
        flagged = df[fraud_signal].astype(int) == 1

    fraud_savings = df.loc[flagged, "total_claim_amount_per_policy"].sum()
    roi = fraud_savings / cost_of_fraud_model if cost_of_fraud_model > 0 else np.nan

    print("\n=== Fraud Model Business Impact ===")
    print(f"Fraud savings (avoided losses): {fraud_savings:,.2f}")
    print(f"Fraud model cost (annual): {cost_of_fraud_model:,.2f}")
    print(f"ROI: {roi:.2f}")


def pricing_scenario(policy_master: pd.DataFrame) -> None:
    """
    Simulate a pricing adjustment for high-risk customers and compare outcomes.
    """
    df = policy_master.copy()

    base_premium = df["annual_premium"].sum()
    base_expected_loss = df["expected_loss"].sum()
    base_profit = (df["annual_premium"] - df["expected_loss"]).sum()
    base_loss_ratio = base_expected_loss / base_premium if base_premium > 0 else np.nan

    # Business scenario: increase premium by 10% for High Risk and Very High Risk.
    scenario_premium = df["annual_premium"].copy()
    high_risk_mask = df["risk_segment"].isin(["High Risk", "Very High Risk"])
    scenario_premium.loc[high_risk_mask] *= 1.10

    new_premium = scenario_premium.sum()
    new_profit = (scenario_premium - df["expected_loss"]).sum()
    new_loss_ratio = base_expected_loss / new_premium if new_premium > 0 else np.nan

    print("\n=== Pricing Scenario: +10% Premium for High Risk ===")
    print(f"Base premium revenue: {base_premium:,.2f}")
    print(f"New premium revenue: {new_premium:,.2f}")
    print(f"Base profit: {base_profit:,.2f}")
    print(f"New profit: {new_profit:,.2f}")
    print(f"Base loss ratio: {base_loss_ratio:.3f}")
    print(f"New loss ratio: {new_loss_ratio:.3f}")


def pipeline_main(base_dir: str = "Dataset"):
    customers, policies, claims, external_risk_factors = load_data(base_dir)
    validate_integrity(customers, policies, claims, external_risk_factors)
    policy_master = build_policy_dataset(
        customers, policies, claims, external_risk_factors
    )
    claim_master = build_claim_dataset(
        customers, policies, claims, external_risk_factors
    )
    print_summary(policy_master, claim_master)
    return policy_master, claim_master


def main(policy_master: pd.DataFrame):
    policy_master = compute_expected_losses(policy_master)
    loss_comparison(policy_master)
    portfolio_profitability(policy_master)
    pricing_strategy(policy_master)
    fraud_business_impact(policy_master)
    pricing_scenario(policy_master)
    return policy_master


if __name__ == "__main__":
    policy_master, claim_master = pipeline_main()
