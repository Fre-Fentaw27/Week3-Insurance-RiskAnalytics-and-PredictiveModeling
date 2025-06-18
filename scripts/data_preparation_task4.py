import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

def load_data_correctly(filepath):
    """Load the data with proper delimiter handling"""
    # First read the first line to check the format
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
    
    # Check if the first line contains pipes
    if '|' in first_line:
        print("Detected pipe-delimited format")
        # Read with pipe delimiter and proper header handling
        data = pd.read_csv(filepath, delimiter='|', header=0)
    else:
        # Try other delimiters as fallback
        print("Trying other delimiters")
        for delimiter in [',', '\t', ';']:
            try:
                data = pd.read_csv(filepath, delimiter=delimiter, header=0)
                print(f"Success with delimiter: {repr(delimiter)}")
                break
            except:
                continue
    
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    return data

def identify_target_columns(data):
    """Identify the correct target columns"""
    # Look for claim amount column (case insensitive)
    claim_col = next((col for col in data.columns 
                     if 'claim' in col.lower() and 'total' in col.lower()), None)
    
    # Look for premium column
    premium_col = next((col for col in data.columns 
                       if 'premium' in col.lower() and 'calculated' in col.lower()), None)
    
    print("\nIdentified columns:")
    print(f"Claim amount column: {claim_col}")
    print(f"Premium column: {premium_col}")
    
    if not claim_col:
        print("\nAvailable columns:")
        print(data.columns.tolist())
        raise ValueError("Could not identify claims column. Please check your data.")
    
    return claim_col, premium_col

def prepare_features(data, claim_col, premium_col):
    """Prepare features and preprocessor"""
    # Convert claim amount to numeric
    data[claim_col] = pd.to_numeric(data[claim_col], errors='coerce')
    
    # Identify numeric and categorical features
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target columns from features
    for col in [claim_col, premium_col]:
        if col and col in numeric_features:
            numeric_features.remove(col)
        elif col and col in categorical_features:
            categorical_features.remove(col)
    
    print("\nSelected numeric features:", numeric_features)
    print("Selected categorical features:", categorical_features)
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    
    return preprocessor

def split_data(data, claim_col, test_size=0.2, random_state=42):
    """Split data into training and test sets"""
    # Convert claim column to numeric if needed
    if not pd.api.types.is_numeric_dtype(data[claim_col]):
        data[claim_col] = pd.to_numeric(data[claim_col], errors='coerce')
    
    # For claim severity model (only rows with claims > 0)
    severity_data = data[data[claim_col] > 0].copy()
    X_sev = severity_data.drop(columns=[claim_col])
    y_sev = severity_data[claim_col]
    
    # For claim probability model (all data)
    X_prob = data.drop(columns=[claim_col])
    y_prob = (data[claim_col] > 0).astype(int)
    
    # Split data
    X_sev_train, X_sev_test, y_sev_train, y_sev_test = train_test_split(
        X_sev, y_sev, test_size=test_size, random_state=random_state)
    
    X_prob_train, X_prob_test, y_prob_train, y_prob_test = train_test_split(
        X_prob, y_prob, test_size=test_size, random_state=random_state)
    
    return {
        'severity': (X_sev_train, X_sev_test, y_sev_train, y_sev_test),
        'probability': (X_prob_train, X_prob_test, y_prob_train, y_prob_test)
    }

if __name__ == "__main__":
    # Load data
    data_path = '../data/MachineLearningRating_v3.txt'
    print(f"Loading data from: {data_path}")
    data = load_data_correctly(data_path)
    
    # Show sample of data
    print("\nFirst 5 rows of data:")
    print(data.head())
    
    # Identify target columns
    claim_col, premium_col = identify_target_columns(data)
    
    # Prepare features and preprocessor
    preprocessor = prepare_features(data, claim_col, premium_col)
    
    # Split data
    splits = split_data(data, claim_col)
    
    # Save outputs
    joblib.dump(preprocessor, 'preprocessor.joblib')
    joblib.dump(splits, 'data_splits.joblib')
    
    print("\nData preparation complete!")
    print(f"Shape of severity training data: {splits['severity'][0].shape}")
    print(f"Shape of probability training data: {splits['probability'][0].shape}")