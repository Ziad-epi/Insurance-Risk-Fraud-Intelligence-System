import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance

SEED = 42
np.random.seed(SEED)

REF_DATE = pd.Timestamp("2026-03-03")

N_REGIONS = 10
N_CUSTOMERS = 20000
N_POLICIES_TARGET = 30000
FRAUD_TARGET = 0.05  # target 4–6%


def _clip(a, lo, hi):
    return np.minimum(np.maximum(a, lo), hi)


def generate_external_risk_factors(n_regions: int) -> pd.DataFrame:
    regions = [f"Region_{i+1}" for i in range(n_regions)]
    accident_rate = np.random.uniform(0.03, 0.12, n_regions)
    fraud_rate = np.random.uniform(0.02, 0.08, n_regions)
    avg_repair = _clip(np.random.normal(1800, 400, n_regions), 800, 3000)
    weather_index = np.random.uniform(0, 1, n_regions)

    return pd.DataFrame(
        {
            "region": regions,
            "accident_rate_region": accident_rate,
            "fraud_rate_region": fraud_rate,
            "average_repair_cost_region": avg_repair,
            "weather_risk_index": weather_index,
        }
    )


def generate_customers(n_customers: int, risk_df: pd.DataFrame) -> pd.DataFrame:
    customer_id = np.arange(1, n_customers + 1)

    age = _clip(np.random.normal(42, 12, n_customers), 18, 85).round().astype(int)

    gender = np.where(np.random.rand(n_customers) < 0.55, "Male", "Female")

    # Non-uniform region distribution
    region_weights = np.array([0.18, 0.14, 0.12, 0.10, 0.10, 0.09, 0.08, 0.07, 0.06, 0.06])
    region_weights = region_weights / region_weights.sum()
    regions = np.random.choice(risk_df["region"].values, size=n_customers, p=region_weights)

    # City density correlated with region (region-specific mix)
    region_to_density = {}
    for i, r in enumerate(risk_df["region"].values):
        # Gradually increase urban share across regions
        urban = 0.25 + 0.04 * i
        suburban = 0.45 - 0.01 * i
        rural = 1.0 - urban - suburban
        probs = np.array([rural, suburban, urban])
        probs = _clip(probs, 0.05, 0.9)
        probs = probs / probs.sum()
        region_to_density[r] = probs

    city_density = np.empty(n_customers, dtype=object)
    for r in risk_df["region"].values:
        mask = regions == r
        probs = region_to_density[r]
        city_density[mask] = np.random.choice(
            ["rural", "suburban", "urban"], size=mask.sum(), p=probs
        )

    # Income level correlated with city density
    income_level = np.empty(n_customers, dtype=object)
    for density in ["rural", "suburban", "urban"]:
        mask = city_density == density
        if density == "rural":
            probs = [0.6, 0.3, 0.1]
        elif density == "suburban":
            probs = [0.35, 0.45, 0.20]
        else:
            probs = [0.20, 0.45, 0.35]
        income_level[mask] = np.random.choice(
            ["Low", "Medium", "High"], size=mask.sum(), p=probs
        )

    # Driving experience
    exp_noise = np.random.normal(0, 2, n_customers)
    driving_exp = age - 18 + exp_noise
    driving_exp = _clip(driving_exp, 0, age - 18).round().astype(int)

    # Bonus-malus correlated with age
    bonus_base = np.ones(n_customers)
    young_adj = np.where(age < 25, 0.005 * (25 - age), 0.0)
    bonus_malus = bonus_base + young_adj + np.random.normal(0, 0.1, n_customers)
    bonus_malus = _clip(bonus_malus, 0.5, 1.5)

    # Marital status by age band
    marital_status = np.empty(n_customers, dtype=object)
    for band, probs in [
        (age < 25, [0.75, 0.23, 0.02, 0.00]),
        ((age >= 25) & (age < 45), [0.30, 0.60, 0.08, 0.02]),
        ((age >= 45) & (age < 65), [0.15, 0.65, 0.15, 0.05]),
        (age >= 65, [0.20, 0.55, 0.15, 0.10]),
    ]:
        labels = ["Single", "Married", "Divorced", "Widowed"]
        marital_status[band] = np.random.choice(labels, size=band.sum(), p=probs)

    number_of_children = np.random.poisson(1, n_customers)

    return pd.DataFrame(
        {
            "customer_id": customer_id,
            "age": age,
            "gender": gender,
            "region": regions,
            "city_density": city_density,
            "income_level": income_level,
            "driving_experience_years": driving_exp,
            "bonus_malus": bonus_malus,
            "marital_status": marital_status,
            "number_of_children": number_of_children,
        }
    )


def _allocate_policy_counts(n_customers: int, target_policies: int) -> np.ndarray:
    # Allocate exactly to target while keeping 1–3 policies per customer
    if target_policies < n_customers or target_policies > 3 * n_customers:
        raise ValueError("target_policies must be between n_customers and 3*n_customers.")

    counts = np.ones(n_customers, dtype=int)
    remaining = target_policies - n_customers

    # Give second policies
    give_second = min(remaining, n_customers)
    if give_second > 0:
        idx = np.random.choice(np.arange(n_customers), size=give_second, replace=False)
        counts[idx] += 1
        remaining -= give_second

    # Give third policies
    if remaining > 0:
        idx2 = np.random.choice(np.where(counts == 2)[0], size=remaining, replace=False)
        counts[idx2] += 1

    return counts


def generate_policies(customers_df: pd.DataFrame, risk_df: pd.DataFrame) -> pd.DataFrame:
    counts = _allocate_policy_counts(len(customers_df), N_POLICIES_TARGET)
    customer_ids = np.repeat(customers_df["customer_id"].values, counts)
    n_policies = len(customer_ids)

    policy_id = np.arange(1, n_policies + 1)

    # Start dates over last 5 years
    days_back = np.random.randint(0, 5 * 365 + 1, n_policies)
    policy_start_date = REF_DATE - pd.to_timedelta(days_back, unit="D")

    policy_duration_months = np.random.choice([12, 24, 36], size=n_policies, p=[0.6, 0.25, 0.15])

    vehicle_types = np.random.choice(
        ["city_car", "sedan", "SUV", "electric", "sports_car"],
        size=n_policies,
        p=[0.35, 0.25, 0.20, 0.10, 0.10],
    )

    vehicle_power = np.empty(n_policies)
    vt_specs = {
        "city_car": (80, 15, 50, 120),
        "sedan": (120, 20, 80, 180),
        "SUV": (160, 25, 110, 230),
        "electric": (140, 20, 90, 200),
        "sports_car": (250, 40, 180, 350),
    }
    for vt, (mu, sd, lo, hi) in vt_specs.items():
        mask = vehicle_types == vt
        vehicle_power[mask] = _clip(np.random.normal(mu, sd, mask.sum()), lo, hi)

    vehicle_age = _clip(np.random.gamma(3, 2, n_policies), 1, 15).round().astype(int)

    annual_mileage = _clip(np.random.normal(13000, 4000, n_policies), 5000, 30000).round().astype(int)

    # Merge customer attributes
    cust = customers_df.set_index("customer_id").loc[customer_ids]

    # Coverage type correlated with income level
    coverage_type = np.empty(n_policies, dtype=object)
    for income in ["Low", "Medium", "High"]:
        mask = cust["income_level"].values == income
        if income == "Low":
            probs = [0.70, 0.25, 0.05]
        elif income == "Medium":
            probs = [0.40, 0.40, 0.20]
        else:
            probs = [0.20, 0.40, 0.40]
        coverage_type[mask] = np.random.choice(
            ["basic", "full", "premium"], size=mask.sum(), p=probs
        )

    # Deductible inversely correlated with coverage
    deductible_amount = np.empty(n_policies)
    for cov, (mu, sd, lo, hi) in {
        "basic": (900, 150, 500, 1500),
        "full": (600, 120, 300, 1200),
        "premium": (400, 100, 200, 800),
    }.items():
        mask = coverage_type == cov
        deductible_amount[mask] = _clip(np.random.normal(mu, sd, mask.sum()), lo, hi).round()

    # Join risk factors for region
    risk = risk_df.set_index("region").loc[cust["region"].values]

    # Annual premium calculation (actuarial risk-based rating)
    base = (
        400
        + 5 * vehicle_power
        + 200 * (coverage_type == "full")
        + 100 * (coverage_type == "premium")
        + 300 * cust["bonus_malus"].values
        + 150 * risk["accident_rate_region"].values
        + np.random.normal(0, 100, n_policies)
    )
    annual_premium = _clip(base, 300, np.inf).round(2)

    return pd.DataFrame(
        {
            "policy_id": policy_id,
            "customer_id": customer_ids,
            "policy_start_date": policy_start_date,
            "policy_duration_months": policy_duration_months,
            "vehicle_type": vehicle_types,
            "vehicle_power": vehicle_power.round(1),
            "vehicle_age": vehicle_age,
            "annual_mileage": annual_mileage,
            "coverage_type": coverage_type,
            "deductible_amount": deductible_amount,
            "annual_premium": annual_premium,
        }
    )


def _sample_by_region(labels, region_array, region_probs):
    out = np.empty(len(region_array), dtype=object)
    for r, probs in region_probs.items():
        mask = region_array == r
        u = np.random.rand(mask.sum())
        cum = np.cumsum(probs)
        out[mask] = np.array(labels)[np.searchsorted(cum, u)]
    return out


def generate_claims(policies_df: pd.DataFrame, customers_df: pd.DataFrame, risk_df: pd.DataFrame) -> pd.DataFrame:
    # Attach customer and region info to policies
    cust = customers_df.set_index("customer_id").loc[policies_df["customer_id"].values]
    risk = risk_df.set_index("region").loc[cust["region"].values]

    # Frequency model (Poisson)
    lam = (
        0.05
        + 0.002 * policies_df["vehicle_power"].values
        + 0.01 * cust["bonus_malus"].values
        + 0.5 * risk["accident_rate_region"].values
        + 0.00002 * policies_df["annual_mileage"].values
    )
    lam = _clip(lam, 0.01, np.inf)
    claims_count = np.random.poisson(lam)

    # Expand policies by claim count
    idx = np.repeat(np.arange(len(policies_df)), claims_count)
    if len(idx) == 0:
        return pd.DataFrame(columns=[
            "claim_id", "policy_id", "claim_date", "accident_type", "weather_condition",
            "injury_involved", "claim_amount", "claim_report_delay_days",
            "police_report_filed", "number_of_previous_claims", "fraud_label"
        ])

    pol = policies_df.iloc[idx].reset_index(drop=True)
    cust_exp = cust.iloc[idx].reset_index(drop=True)
    risk_exp = risk.iloc[idx].reset_index(drop=True)

    # Claim date within policy period
    start = pol["policy_start_date"]
    end = start + pd.to_timedelta(pol["policy_duration_months"] * 30, unit="D")
    end = pd.Series(np.minimum(end.values.astype("datetime64[ns]"), REF_DATE.to_datetime64()))
    days_range = (end - start).dt.days
    days_range = np.maximum(days_range, 1)
    offset_days = np.random.randint(0, days_range + 1)
    claim_date = start + pd.to_timedelta(offset_days, unit="D")

    # Accident type dependent on region (weather index drives weather-related share)
    accident_types = ["rear_end", "side_collision", "single_vehicle", "theft", "weather_related"]
    region_probs = {}
    for r in risk_df["region"].values:
        w = risk_df.loc[risk_df["region"] == r, "weather_risk_index"].values[0]
        base = np.array([0.30, 0.22, 0.20, 0.12, 0.16])
        base[-1] += 0.20 * w
        base[0] -= 0.05 * w
        base[1] -= 0.05 * w
        base = _clip(base, 0.05, 0.7)
        base = base / base.sum()
        region_probs[r] = base

    accident_type = _sample_by_region(
        accident_types, cust_exp["region"].values, region_probs
    )

    # Weather condition correlated with weather risk
    weather_labels = ["clear", "rain", "snow", "storm"]
    weather_probs = {}
    for r in risk_df["region"].values:
        w = risk_df.loc[risk_df["region"] == r, "weather_risk_index"].values[0]
        p_clear = 0.60 - 0.30 * w
        p_rain = 0.25 + 0.20 * w
        p_snow = 0.05 + 0.10 * w
        p_storm = 0.10 + 0.00 * w
        probs = np.array([p_clear, p_rain, p_snow, p_storm])
        probs = _clip(probs, 0.02, 0.9)
        probs = probs / probs.sum()
        weather_probs[r] = probs

    weather_condition = _sample_by_region(
        weather_labels, cust_exp["region"].values, weather_probs
    )

    # Injury involved
    p_injury = _clip(0.05 + 0.001 * pol["vehicle_power"].values, 0, 0.30)
    injury_involved = np.random.rand(len(pol)) < p_injury

    # Claim amount via gamma (cost drivers)
    base_cost = risk_exp["average_repair_cost_region"].values.copy()
    base_cost *= np.where(pol["vehicle_type"].values == "SUV", 1.2, 1.0)
    base_cost *= np.where(pol["vehicle_type"].values == "sports_car", 1.5, 1.0)
    base_cost *= np.where(injury_involved, 2.0, 1.0)

    claim_amount = np.random.gamma(shape=2, scale=base_cost)

    # Report delay
    claim_report_delay_days = np.random.lognormal(mean=2, sigma=1, size=len(pol)).round().astype(int)

    # Police report filed (less likely when delay is high and in lower severity)
    p_police = 0.70 + 0.10 * injury_involved + 0.10 * (claim_amount > 4000) - 0.15 * (claim_report_delay_days > 15)
    p_police = _clip(p_police, 0.2, 0.98)
    police_report_filed = np.random.rand(len(pol)) < p_police

    # Previous claims per policy (cumulative)
    claim_order = pol.groupby("policy_id").cumcount()
    number_of_previous_claims = claim_order.values

    # Fraud probability and calibration
    p_fraud_base = (
        0.02
        + 0.00001 * claim_amount
        + 0.02 * (claim_report_delay_days > 15)
        + 0.03 * (~police_report_filed)
        + risk_exp["fraud_rate_region"].values
    )

    def expected_rate(delta):
        p = _clip(p_fraud_base + delta, 0, 0.4)
        return p.mean()

    # Wider range to ensure we can hit the 4–6% target even with high base risk
    lo, hi = -0.25, 0.08
    for _ in range(25):
        mid = (lo + hi) / 2
        if expected_rate(mid) < FRAUD_TARGET:
            lo = mid
        else:
            hi = mid
    delta = (lo + hi) / 2

    p_fraud = _clip(p_fraud_base + delta, 0, 0.4)
    fraud_label = np.random.rand(len(pol)) < p_fraud

    claims_df = pd.DataFrame(
        {
            "claim_id": np.arange(1, len(pol) + 1),
            "policy_id": pol["policy_id"].values,
            "claim_date": claim_date.values,
            "accident_type": accident_type,
            "weather_condition": weather_condition,
            "injury_involved": injury_involved,
            "claim_amount": claim_amount.round(2),
            "claim_report_delay_days": claim_report_delay_days,
            "police_report_filed": police_report_filed,
            "number_of_previous_claims": number_of_previous_claims,
            "fraud_label": fraud_label.astype(int),
        }
    )

    return claims_df, claims_count


def validate_and_report(customers_df, policies_df, claims_df, claims_count):
    # Basic integrity checks
    assert policies_df["customer_id"].isin(customers_df["customer_id"]).all()
    assert claims_df["policy_id"].isin(policies_df["policy_id"]).all()

    fraud_rate = claims_df["fraud_label"].mean() if len(claims_df) else 0.0
    print(f"Fraud rate: {fraud_rate:.4f}")

    if len(claims_df) > 0:
        print("\nClaim amount summary:")
        print(claims_df["claim_amount"].describe())

        merged = claims_df.merge(
            policies_df[["policy_id", "vehicle_power"]],
            on="policy_id",
            how="left",
        )
        corr = merged["vehicle_power"].corr(merged["claim_amount"])
        print(f"\nCorrelation vehicle_power vs claim_amount: {corr:.4f}")

    avg_claims_per_policy = claims_count.mean()
    print(f"\nAverage claims per policy: {avg_claims_per_policy:.4f}")

    # Bonus-malus vs frequency correlation
    freq_df = policies_df[["policy_id", "customer_id"]].copy()
    freq_df["claims_count"] = claims_count
    bm = customers_df.set_index("customer_id")["bonus_malus"]
    freq_df["bonus_malus"] = bm.loc[freq_df["customer_id"]].values
    bm_corr = freq_df["bonus_malus"].corr(freq_df["claims_count"])
    print(f"\nCorrelation bonus_malus vs frequency: {bm_corr:.4f}")



def _first_existing(df: pd.DataFrame, candidates, label: str, required: bool = True) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"Missing required column for {label}. Tried: {candidates}")
    return ""


def _prepare_policy_master(policy_master: pd.DataFrame) -> pd.DataFrame:
    df = policy_master.copy()

    # Standardize exposure
    if "exposure_years" not in df.columns:
        exposure_col = _first_existing(df, ["exposure_years", "exposure", "policy_exposure"], "exposure")
        df["exposure_years"] = df[exposure_col].astype(float)

    # Standardize total claims per policy
    if "total_claims_per_policy" not in df.columns:
        tc_col = _first_existing(
            df,
            ["total_claims_per_policy", "claims_count", "num_claims", "total_claims"],
            "total_claims_per_policy",
            required=False,
        )
        if tc_col:
            df["total_claims_per_policy"] = df[tc_col].astype(float)

    # Standardize claim frequency
    if "claim_frequency" not in df.columns:
        if "total_claims_per_policy" in df.columns and "exposure_years" in df.columns:
            df["claim_frequency"] = df["total_claims_per_policy"] / df["exposure_years"].replace(0, np.nan)
        else:
            _first_existing(df, ["claim_frequency"], "claim_frequency")

    # Standardize loss ratio
    if "loss_ratio" not in df.columns:
        loss_col = _first_existing(
            df,
            ["total_claim_amount", "total_claims_amount", "incurred_loss", "loss_amount", "total_loss", "claim_cost"],
            "loss_amount",
        )
        prem_col = _first_existing(
            df,
            ["earned_premium", "annual_premium", "premium", "written_premium", "net_premium"],
            "premium",
        )
        df["loss_ratio"] = df[loss_col] / df[prem_col].replace(0, np.nan)

    return df


def _ensure_age_band(policy_master: pd.DataFrame) -> pd.DataFrame:
    df = policy_master.copy()
    if "age_band" in df.columns:
        return df

    age_col = _first_existing(df, ["age", "driver_age", "policyholder_age"], "age")
    bins = [17, 25, 35, 45, 55, 65, 75, 90]
    labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
    df["age_band"] = pd.cut(df[age_col], bins=bins, labels=labels, right=True, include_lowest=True)
    return df


def _set_plot_style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        }
    )


def portfolio_overview(policy_master: pd.DataFrame) -> None:
    """Global portfolio overview for risk monitoring."""
    _set_plot_style()
    df = _prepare_policy_master(policy_master)

    premium_col = _first_existing(
        df,
        ["earned_premium", "annual_premium", "premium", "written_premium", "net_premium"],
        "premium",
    )
    fraud_col = _first_existing(
        df,
        ["fraud_flag", "fraud_label", "is_fraud", "fraud_indicator", "fraudulent"],
        "fraud_flag",
    )

    n_policies = len(df)
    avg_exposure = df["exposure_years"].mean()
    avg_premium = df[premium_col].mean()
    avg_loss_ratio = df["loss_ratio"].mean()
    claim_freq = df["claim_frequency"].mean()
    fraud_rate = df[fraud_col].mean()

    print("\n=== Portfolio Overview ===")
    print(f"Number of policies: {n_policies:,}")
    print(f"Average exposure (years): {avg_exposure:.3f}")
    print(f"Claim frequency (overall): {claim_freq:.4f}")
    print(f"Average premium: {avg_premium:.2f}")
    print(f"Average loss ratio: {avg_loss_ratio:.3f}")
    print(f"Fraud rate: {fraud_rate:.3f}")

    # Distribution of claim frequency
    plt.figure(figsize=(8, 5))
    sns.histplot(df["claim_frequency"].dropna(), bins=30, kde=True, color="#2C7FB8")
    plt.title("Distribution of Claim Frequency")
    plt.xlabel("Claim Frequency")
    plt.ylabel("Policies")
    plt.tight_layout()
    plt.show()

    # Distribution of loss ratio
    plt.figure(figsize=(8, 5))
    sns.histplot(df["loss_ratio"].dropna(), bins=30, kde=True, color="#F28E2B")
    plt.title("Distribution of Loss Ratio")
    plt.xlabel("Loss Ratio")
    plt.ylabel("Policies")
    plt.tight_layout()
    plt.show()

    # Histogram of exposure years
    plt.figure(figsize=(8, 5))
    sns.histplot(df["exposure_years"].dropna(), bins=20, color="#59A14F")
    plt.title("Exposure (Years) Distribution")
    plt.xlabel("Exposure Years")
    plt.ylabel("Policies")
    plt.tight_layout()
    plt.show()

    risk_flag = (avg_loss_ratio > 0.8) or (claim_freq > 0.15)
    # Interpretation (insurance logic)
    print("Interpretation:")
    if risk_flag:
        print("- Portfolio looks moderately risky (elevated loss ratio or frequency).")
    else:
        print("- Portfolio appears stable with manageable frequency and loss ratio.")
    print("- Higher dispersion in loss ratio typically indicates heterogeneous risk segments.")


def frequency_analysis(policy_master: pd.DataFrame) -> None:
    """Frequency diagnostics for rate adequacy."""
    _set_plot_style()
    df = _prepare_policy_master(policy_master)
    df = _ensure_age_band(df)

    vehicle_col = _first_existing(
        df, ["vehicle_type", "vehicle_class", "vehicle_segment"], "vehicle_type"
    )
    region_col = _first_existing(df, ["region", "territory", "geo_region"], "region")
    bm_col = _first_existing(
        df, ["bonus_malus", "bonus_malus_factor", "bonus_malus_score"], "bonus_malus"
    )

    print("\n=== Frequency Analysis ===")

    # Mean frequency by category
    for col, title in [
        ("age_band", "Age Band"),
        (vehicle_col, "Vehicle Type"),
        (region_col, "Region"),
        (bm_col, "Bonus-Malus"),
    ]:
        plt.figure(figsize=(9, 5))
        freq_by = df.groupby(col)["claim_frequency"].mean().sort_values()
        sns.barplot(x=freq_by.index, y=freq_by.values, color="#4E79A7")
        plt.title(f"Mean Claim Frequency by {title}")
        plt.xlabel(title)
        plt.ylabel("Mean Claim Frequency")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.show()

    # Boxplots
    for col, title in [
        ("age_band", "Age Band"),
        (vehicle_col, "Vehicle Type"),
        (region_col, "Region"),
    ]:
        plt.figure(figsize=(9, 5))
        sns.boxplot(x=col, y="claim_frequency", data=df, color="#A0CBE8")
        plt.title(f"Claim Frequency by {title}")
        plt.xlabel(title)
        plt.ylabel("Claim Frequency")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.show()

    # Correlation heatmap for numeric features
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        numeric_df.corr(),
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        square=False,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.show()

    # Overdispersion check
    if "total_claims_per_policy" not in df.columns:
        df["total_claims_per_policy"] = (
            df["claim_frequency"] * df["exposure_years"].replace(0, np.nan)
        )
    mean_claims = df["total_claims_per_policy"].mean()
    var_claims = df["total_claims_per_policy"].var()
    ratio = var_claims / mean_claims if mean_claims > 0 else np.nan
    print(f"Mean claims per policy: {mean_claims:.4f}")
    print(f"Variance of claims per policy: {var_claims:.4f}")
    print(f"Variance/Mean ratio: {ratio:.3f}")

    # Interpretation (insurance logic)
    print("Interpretation:")
    if ratio > 1.5:
        print("- Overdispersion present; Poisson assumption likely fails.")
    else:
        print("- Dispersion is close to Poisson; basic Poisson may be adequate.")
    print("- High frequency segments warrant refined pricing or underwriting.")


def severity_analysis(claim_master: pd.DataFrame) -> None:
    """Severity diagnostics for loss modeling."""
    _set_plot_style()
    df = claim_master.copy()

    amount_col = _first_existing(
        df,
        ["claim_amount", "claim_cost", "loss_amount", "paid_amount", "incurred_amount"],
        "claim_amount",
    )

    claim_amount = df[amount_col].dropna()

    print("\n=== Severity Analysis ===")

    # Histogram (raw)
    plt.figure(figsize=(8, 5))
    sns.histplot(claim_amount, bins=35, color="#E15759")
    plt.title("Claim Amount Distribution (Raw)")
    plt.xlabel("Claim Amount")
    plt.ylabel("Claims")
    plt.tight_layout()
    plt.show()

    # Histogram (log scale)
    plt.figure(figsize=(8, 5))
    sns.histplot(np.log1p(claim_amount), bins=35, color="#76B7B2")
    plt.title("Claim Amount Distribution (Log Scale)")
    plt.xlabel("log(1 + Claim Amount)")
    plt.ylabel("Claims")
    plt.tight_layout()
    plt.show()

    # Q-Q plot vs normal
    plt.figure(figsize=(6, 6))
    stats.probplot(claim_amount, dist="norm", plot=plt)
    plt.title("Q-Q Plot: Normal")
    plt.tight_layout()
    plt.show()

    # Q-Q plot vs gamma
    plt.figure(figsize=(6, 6))
    try:
        shape, loc, scale = stats.gamma.fit(claim_amount, floc=0)
        stats.probplot(claim_amount, dist=stats.gamma, sparams=(shape, loc, scale), plot=plt)
        plt.title("Q-Q Plot: Gamma")
    except Exception:
        plt.title("Q-Q Plot: Gamma (fit failed)")
    plt.tight_layout()
    plt.show()

    mean_val = claim_amount.mean()
    median_val = claim_amount.median()
    std_val = claim_amount.std()
    skew_val = stats.skew(claim_amount)
    kurt_val = stats.kurtosis(claim_amount, fisher=False)

    print(f"Mean: {mean_val:.2f}")
    print(f"Median: {median_val:.2f}")
    print(f"Std: {std_val:.2f}")
    print(f"Skewness: {skew_val:.2f}")
    print(f"Kurtosis: {kurt_val:.2f}")

    # Interpretation (insurance logic)
    print("Interpretation:")
    if skew_val > 1 or kurt_val > 5:
        print("- Severity is heavy-tailed; large losses materially impact risk.")
    else:
        print("- Severity is moderately skewed; standard distributions may fit.")
    print("- Gamma/lognormal often outperform normal for insurance severities.")


def fraud_analysis(claim_master: pd.DataFrame) -> None:
    """Fraud exploration with behavioral signals."""
    _set_plot_style()
    df = claim_master.copy()

    amount_col = _first_existing(
        df,
        ["claim_amount", "claim_cost", "loss_amount", "paid_amount", "incurred_amount"],
        "claim_amount",
    )
    fraud_col = _first_existing(
        df,
        ["fraud_flag", "fraud_label", "is_fraud", "fraud_indicator", "fraudulent"],
        "fraud_flag",
    )
    delay_col = _first_existing(
        df,
        ["delay", "report_delay", "claim_report_delay_days", "days_to_report"],
        "delay",
    )
    region_col = _first_existing(df, ["region", "policy_region", "claim_region"], "region")
    vehicle_col = _first_existing(
        df, ["vehicle_type", "vehicle_class", "vehicle_segment"], "vehicle_type"
    )

    print("\n=== Fraud Analysis ===")

    fraud_rate_region = df.groupby(region_col)[fraud_col].mean().sort_values()
    plt.figure(figsize=(9, 5))
    sns.barplot(x=fraud_rate_region.index, y=fraud_rate_region.values, color="#EDC949")
    plt.title("Fraud Rate by Region")
    plt.xlabel("Region")
    plt.ylabel("Fraud Rate")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.show()

    fraud_rate_vehicle = df.groupby(vehicle_col)[fraud_col].mean().sort_values()
    plt.figure(figsize=(9, 5))
    sns.barplot(x=fraud_rate_vehicle.index, y=fraud_rate_vehicle.values, color="#9C755F")
    plt.title("Fraud Rate by Vehicle Type")
    plt.xlabel("Vehicle Type")
    plt.ylabel("Fraud Rate")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x=fraud_col, y=amount_col, data=df, palette=["#4E79A7", "#E15759"])
    plt.title("Claim Amount by Fraud Flag")
    plt.xlabel("Fraud Flag")
    plt.ylabel("Claim Amount")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df,
        x=delay_col,
        hue=fraud_col,
        bins=30,
        element="step",
        stat="density",
        common_norm=False,
        palette=["#4E79A7", "#E15759"],
    )
    plt.title("Delay Distribution: Fraud vs Non-Fraud")
    plt.xlabel("Report Delay (days)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()

    fraud_amount = df.loc[df[fraud_col] == 1, amount_col]
    non_fraud_amount = df.loc[df[fraud_col] == 0, amount_col]
    delta = fraud_amount.median() - non_fraud_amount.median()

    # Interpretation (insurance logic)
    print("Interpretation:")
    if delta > 0:
        print("- Fraudulent claims tend to have higher amounts (severity signal).")
    else:
        print("- Fraud severity is not higher than non-fraud on average.")
    fraud_delay = df.loc[df[fraud_col] == 1, delay_col]
    non_fraud_delay = df.loc[df[fraud_col] == 0, delay_col]
    if fraud_delay.median() > non_fraud_delay.median():
        print("- Fraudulent claims show longer reporting delays on average.")
    else:
        print("- Reporting delays show limited separation for fraud.")


def economic_risk_analysis(policy_master: pd.DataFrame) -> None:
    """Economic drivers and macro risk sensitivity."""
    _set_plot_style()
    df = _prepare_policy_master(policy_master)

    inflation_col = _first_existing(
        df,
        ["inflation", "inflation_rate", "cpi", "inflation_index"],
        "inflation",
    )
    region_risk_col = _first_existing(
        df,
        ["region_risk_index", "regional_risk_index", "risk_index_region"],
        "region_risk_index",
    )

    print("\n=== Economic Risk Analysis ===")

    plt.figure(figsize=(7, 5))
    sns.regplot(
        x=inflation_col,
        y="loss_ratio",
        data=df,
        scatter_kws={"alpha": 0.4},
        line_kws={"color": "#E15759"},
    )
    plt.title("Inflation vs Loss Ratio")
    plt.xlabel("Inflation")
    plt.ylabel("Loss Ratio")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    sns.regplot(
        x=region_risk_col,
        y="claim_frequency",
        data=df,
        scatter_kws={"alpha": 0.4},
        line_kws={"color": "#4E79A7"},
    )
    plt.title("Region Risk Index vs Claim Frequency")
    plt.xlabel("Region Risk Index")
    plt.ylabel("Claim Frequency")
    plt.tight_layout()
    plt.show()

    # Interpretation (insurance logic)
    print("Interpretation:")
    corr_lr = df[inflation_col].corr(df["loss_ratio"])
    corr_freq = df[region_risk_col].corr(df["claim_frequency"])
    if corr_lr > 0.2:
        print("- Loss ratio rises with inflation, indicating cost pressure.")
    else:
        print("- Inflation sensitivity appears mild at portfolio level.")
    if corr_freq > 0.2:
        print("- Higher regional risk index aligns with higher frequency.")
    else:
        print("- Regional risk index shows limited frequency signal.")


def eda_main(policy_master: pd.DataFrame, claim_master: pd.DataFrame) -> None:
    portfolio_overview(policy_master)
    frequency_analysis(policy_master)
    severity_analysis(claim_master)
    fraud_analysis(claim_master)
    economic_risk_analysis(policy_master)



def prepare_frequency_data(policy_master: pd.DataFrame):
    """Prepare design matrix and offsets for Poisson frequency modeling."""
    df = policy_master.copy()

    # Actuarial: exclude zero/negative exposure (no risk on books)
    df = df[df["exposure_years"] > 0].copy()

    # Target: count of claims per policy (no scaling)
    y_col = _first_existing(
        df,
        ["total_claims_per_policy", "claims_count", "num_claims", "total_claims"],
        "total_claims_per_policy",
    )
    y = df[y_col].astype(float)

    # Core numeric features
    age_col = _first_existing(df, ["age", "driver_age", "policyholder_age"], "age")
    econ_col = _first_existing(
        df,
        ["economic_index", "inflation", "inflation_rate", "cpi", "region_risk_index"],
        "economic_index",
    )

    num_features = df[[age_col, "vehicle_age", "vehicle_power", "bonus_malus", econ_col]].copy()

    # Engineered interaction features (risk mix effects)
    num_features["age_bonus_malus_interaction"] = num_features[age_col] * num_features["bonus_malus"]
    num_features["power_vehicle_age_interaction"] = num_features["vehicle_power"] * num_features["vehicle_age"]
    num_features["bonus_malus_econ_interaction"] = num_features["bonus_malus"] * num_features[econ_col]

    # Categorical features: region and vehicle type
    region_col = _first_existing(df, ["region", "territory", "geo_region"], "region")
    vehicle_col = _first_existing(
        df, ["vehicle_type", "vehicle_class", "vehicle_segment"], "vehicle_type"
    )

    cat_features = df[[region_col, vehicle_col]].copy()
    cat_dummies = pd.get_dummies(cat_features, drop_first=True)

    # Combine features
    X = pd.concat([num_features, cat_dummies], axis=1)

    # Add constant term for GLM intercept
    X = sm.add_constant(X, has_constant="add")

    # Offset: log exposure
    offset = np.log(df["exposure_years"].astype(float))

    # Train/test split (no leakage from target or exposure)
    X_train, X_test, y_train, y_test, offset_train, offset_test = train_test_split(
        X, y, offset, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, offset_train, offset_test


def train_poisson_glm(X_train: pd.DataFrame, y_train: pd.Series, offset_train: pd.Series):
    """Fit Poisson GLM with exposure offset for frequency."""
    # Actuarial: Poisson frequency with log(exposure) as offset
    model = sm.GLM(
        y_train,
        X_train,
        family=sm.families.Poisson(),
        offset=offset_train,
    )
    result = model.fit()
    print(result.summary())
    return result


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, offset_test: pd.Series) -> None:
    """Evaluate predictive performance and dispersion."""
    # Predict expected claim counts
    y_pred = model.predict(X_test, offset=offset_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    poisson_dev = mean_poisson_deviance(y_test, y_pred)

    print("\n=== Model Evaluation ===")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Poisson Deviance: {poisson_dev:.4f}")

    y_train_used = model.model.endog
    mean_claims = np.mean(y_train_used)
    var_claims = np.var(y_train_used, ddof=1)
    ratio = var_claims / mean_claims if mean_claims > 0 else np.nan
    print(f"Variance/Mean ratio (train): {ratio:.3f}")

    # Interpretation (actuarial dispersion check)
    if ratio > 1.5:
        print("- Overdispersion warning: consider Negative Binomial or quasi-Poisson.")
    else:
        print("- Dispersion close to Poisson; Poisson GLM is reasonable.")


def interpret_coefficients(model) -> None:
    """Interpret top drivers using multiplicative effects."""
    params = model.params.copy()
    coef_table = pd.DataFrame(
        {
            "feature": params.index,
            "coef": params.values,
            "exp_coef": np.exp(params.values),
            "abs_coef": np.abs(params.values),
        }
    )
    coef_table = coef_table.sort_values("abs_coef", ascending=False)

    print("\n=== Top 10 Most Impactful Features ===")
    print(coef_table[["feature", "coef", "exp_coef"]].head(10).to_string(index=False))

    # Actuarial interpretation:
    # exp(coef) > 1 increases expected frequency; exp(coef) < 1 decreases it.


def main(policy_master: pd.DataFrame):
    X_train, X_test, y_train, y_test, offset_train, offset_test = prepare_frequency_data(
        policy_master
    )
    model = train_poisson_glm(X_train, y_train, offset_train)
    evaluate_model(model, X_test, y_test, offset_test)
    interpret_coefficients(model)
    return model


def generate_data_main():
    risk_df = generate_external_risk_factors(N_REGIONS)
    customers_df = generate_customers(N_CUSTOMERS, risk_df)
    policies_df = generate_policies(customers_df, risk_df)
    claims_df, claims_count = generate_claims(policies_df, customers_df, risk_df)

    # Export to CSV
    customers_df.to_csv("customers.csv", index=False)
    policies_df.to_csv("policies.csv", index=False)
    claims_df.to_csv("claims.csv", index=False)
    risk_df.to_csv("external_risk_factors.csv", index=False)

    # Validation
    validate_and_report(customers_df, policies_df, claims_df, claims_count)


if __name__ == "__main__":
    generate_data_main()
