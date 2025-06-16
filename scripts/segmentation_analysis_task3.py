import pandas as pd
from scipy import stats

# Load dataset
df = pd.read_csv("../data/MachineLearningRating_v3.txt", delimiter="|", low_memory=False)

# Convert Yes/No to 1/0
df["TrackingDevice"] = df["TrackingDevice"].map({"Yes": 1, "No": 0})

# Drop missing values for essential columns
df.dropna(subset=["TotalPremium", "TotalClaims", "TrackingDevice", "Gender", "VehicleType"], inplace=True)

# Clean numerical columns just in case
df["TotalPremium"] = pd.to_numeric(df["TotalPremium"], errors="coerce")
df["TotalClaims"] = pd.to_numeric(df["TotalClaims"], errors="coerce")

# Create calculated metrics
df["HasClaim"] = df["TotalClaims"] > 0
df["ClaimSeverity"] = df.apply(lambda row: row["TotalClaims"] if row["HasClaim"] else None, axis=1)
df["Margin"] = df["TotalPremium"] - df["TotalClaims"]

# Create groups
group_A = df[df["TrackingDevice"] == 0]
group_B = df[df["TrackingDevice"] == 1]

# Statistical test function
def run_test(name, a_series, b_series):
    print(f"\n🔍 {name.upper()}")
    print(f"{name} A: {a_series.mean():.4f}, B: {b_series.mean():.4f}")
    t_stat, p_val = stats.ttest_ind(a_series.dropna().astype(float), b_series.dropna().astype(float), equal_var=False)
    print(f"T-test: t = {t_stat:.4f}, p = {p_val:.4f}")

# Claim Frequency
run_test("Claim Frequency", group_A["HasClaim"].astype(int), group_B["HasClaim"].astype(int))

# Claim Severity
run_test("Claim Severity",
         group_A[group_A["HasClaim"]]["ClaimSeverity"],
         group_B[group_B["HasClaim"]]["ClaimSeverity"])

# Margin
run_test("Margin", group_A["Margin"], group_B["Margin"])

# Optional: Control check
print("\n✅ Control Check: Gender Distribution")
print(group_A["Gender"].value_counts(normalize=True))
print(group_B["Gender"].value_counts(normalize=True))

print("\n✅ Control Check: VehicleType Distribution")
print(group_A["VehicleType"].value_counts(normalize=True))
print(group_B["VehicleType"].value_counts(normalize=True))