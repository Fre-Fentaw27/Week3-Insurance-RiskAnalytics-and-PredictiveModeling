import pandas as pd
import numpy as np
from scipy import stats

# Load dataset
df = pd.read_csv("../data/MachineLearningRating_v3.txt", delimiter="|", low_memory=False)

# Clean and preprocess
df = df.dropna(subset=["TrackingDevice", "Gender", "VehicleType", "TotalPremium", "TotalClaims"])
df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce')
df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce')
df = df.dropna(subset=['TotalPremium', 'TotalClaims'])

# Create binary HasClaim column (Claim Frequency per policy)
df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)

# Create ClaimSeverity (average claim amount given claim occurred)
df['ClaimSeverity'] = df.apply(lambda row: row['TotalClaims'] / 1 if row['HasClaim'] == 1 else np.nan, axis=1)

# Create Margin (profit)
df['Margin'] = df['TotalPremium'] - df['TotalClaims']

def run_chi2_test(feature, kpi):
    """Run Chi-squared test for categorical KPI"""
    contingency_table = pd.crosstab(df[feature], df[kpi])
    chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
    print(f"\nChi-squared test between {feature} and {kpi}:")
    print(f"Chi2 stat = {chi2:.4f}, p-value = {p:.4f}")
    if p < 0.05:
        print("→ Reject null hypothesis: there is a significant association.")
    else:
        print("→ Fail to reject null hypothesis: no significant association.")

def run_ttest(feature, kpi):
    """Run t-test for numerical KPI by feature groups"""
    groups = df[[feature, kpi]].dropna().groupby(feature)
    if len(groups) != 2:
        print(f"Feature {feature} does not have exactly 2 groups, skipping t-test.")
        return
    group_names = list(groups.groups.keys())
    group1 = groups.get_group(group_names[0])[kpi]
    group2 = groups.get_group(group_names[1])[kpi]
    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
    print(f"\nT-test for {kpi} by {feature}:")
    print(f"Group {group_names[0]} mean = {group1.mean():.4f}")
    print(f"Group {group_names[1]} mean = {group2.mean():.4f}")
    print(f"t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
    if p_val < 0.05:
        print("→ Reject null hypothesis: significant difference exists.")
    else:
        print("→ Fail to reject null hypothesis: no significant difference.")

if __name__ == "__main__":
    # Test impact of TrackingDevice on HasClaim (Claim Frequency)
    run_chi2_test("TrackingDevice", "HasClaim")

    # Test impact of TrackingDevice on ClaimSeverity
    run_ttest("TrackingDevice", "ClaimSeverity")

    # Test impact of TrackingDevice on Margin
    run_ttest("TrackingDevice", "Margin")

    # Test impact of Gender on HasClaim (Claim Frequency)
    run_chi2_test("Gender", "HasClaim")

    # Test impact of Gender on ClaimSeverity
    run_ttest("Gender", "ClaimSeverity")

    # Test impact of Gender on Margin
    run_ttest("Gender", "Margin")