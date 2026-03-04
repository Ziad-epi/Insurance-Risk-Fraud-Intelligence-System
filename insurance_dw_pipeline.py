import numpy as np
import pandas as pd
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


def main(base_dir: str = "Dataset"):
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


if __name__ == "__main__":
    policy_master, claim_master = main()
