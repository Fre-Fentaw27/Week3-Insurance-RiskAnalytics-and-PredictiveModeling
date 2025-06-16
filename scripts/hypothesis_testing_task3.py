import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway

def load_and_prepare_data(filepath):
    """Load and prepare the insurance data for analysis."""
    df = pd.read_csv(filepath, delimiter="|", low_memory=False)
    
    # Select and clean required columns
    required_cols = ["TotalPremium", "TotalClaims", "Province", "PostalCode", "Gender"]
    df = df.dropna(subset=required_cols)
    df = df.rename(columns={"PostalCode": "ZipCode"})
    
    # Calculate metrics
    df["ClaimFrequency"] = (df["TotalClaims"] > 0).astype(int)
    df["ClaimSeverity"] = np.where(df["ClaimFrequency"] == 1, df["TotalClaims"], np.nan)
    df["Margin"] = df["TotalPremium"] - df["TotalClaims"]
    
    return df

def test_province_risk(df):
    """Test for risk differences across provinces."""
    print("\n🔍 Hypothesis 1: No risk differences across Provinces (Claim Frequency)")
    province_groups = df.groupby("Province")["ClaimFrequency"].apply(list)
    
    if len(province_groups) >= 2:
        f_stat, p_value = f_oneway(*province_groups)
        print(f"ANOVA result: F-stat = {f_stat:.4f}, p = {p_value:.4f}")
        print("→ Reject H₀" if p_value < 0.05 else "→ Fail to reject H₀")
        
        # Show top 3 highest and lowest claim frequency provinces
        province_stats = df.groupby("Province")["ClaimFrequency"].mean().sort_values(ascending=False)
        print("\nTop 3 Highest Risk Provinces:")
        print(province_stats.head(3))
        print("\nTop 3 Lowest Risk Provinces:")
        print(province_stats.tail(3))
    else:
        print("Insufficient province groups for ANOVA")

def test_zipcode_risk(df, min_samples=30):
    """Test for risk differences between zip codes."""
    print("\n🔍 Hypothesis 2: No risk differences between ZipCodes (Claim Frequency)")
    zip_groups = [group for _, group in df.groupby("ZipCode") if len(group) >= min_samples]
    
    if len(zip_groups) >= 2:
        f_stat, p_value = f_oneway(*[g["ClaimFrequency"] for g in zip_groups])
        print(f"ANOVA result: F-stat = {f_stat:.4f}, p = {p_value:.4f}")
        print("→ Reject H₀" if p_value < 0.05 else "→ Fail to reject H₀")
    else:
        print(f"Insufficient zip codes with ≥{min_samples} samples")

def test_zipcode_margin(df, min_samples=30):
    """Test for margin differences between zip codes."""
    print("\n🔍 Hypothesis 3: No margin difference between ZipCodes")
    zip_groups = [group for _, group in df.groupby("ZipCode") if len(group) >= min_samples]
    
    if len(zip_groups) >= 2:
        f_stat, p_value = f_oneway(*[g["Margin"] for g in zip_groups])
        print(f"ANOVA result: F-stat = {f_stat:.4f}, p = {p_value:.4f}")
        print("→ Reject H₀" if p_value < 0.05 else "→ Fail to reject H₀")
    else:
        print(f"Insufficient zip codes with ≥{min_samples} samples")

def test_gender_risk(df):
    """Test for risk differences between genders."""
    print("\n🔍 Hypothesis 4: No risk difference between Women and Men (Claim Frequency)")
    
    # Clean gender categories
    df["Gender"] = df["Gender"].str.strip().str.title()
    valid_genders = ["Male", "Female"]
    gender_df = df[df["Gender"].isin(valid_genders)]
    
    if len(gender_df["Gender"].unique()) == 2:
        female = gender_df[gender_df["Gender"] == "Female"]["ClaimFrequency"]
        male = gender_df[gender_df["Gender"] == "Male"]["ClaimFrequency"]
        
        t_stat, p_value = ttest_ind(female, male, equal_var=False)
        print(f"T-test result: t = {t_stat:.4f}, p = {p_value:.4f}")
        print("→ Reject H₀" if p_value < 0.05 else "→ Fail to reject H₀")
        
        # Show gender proportions
        print(f"\nFemale claim rate: {female.mean():.2%}")
        print(f"Male claim rate: {male.mean():.2%}")
    else:
        print("Insufficient gender data for comparison")

def main():
    # Load and prepare data
    data_path = "../data/MachineLearningRating_v3.txt"
    df = load_and_prepare_data(data_path)
    
    # Run all hypothesis tests
    test_province_risk(df)
    test_zipcode_risk(df)
    test_zipcode_margin(df)
    test_gender_risk(df)

if __name__ == "__main__":
    main()