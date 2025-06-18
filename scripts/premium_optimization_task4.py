import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from scipy.sparse import issparse

# Configure output
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8')

class DataCleaner(BaseEstimator, TransformerMixin):
    """Custom transformer to clean data before preprocessing"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if issparse(X):
            raise ValueError("DataCleaner cannot handle sparse matrices. Ensure input is a DataFrame.")
            
        X_clean = X.copy()
        # Numeric columns
        num_cols = X_clean.select_dtypes(include=['number']).columns
        for col in num_cols:
            X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
        # Categorical columns
        cat_cols = X_clean.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            X_clean[col] = X_clean[col].astype(str).fillna('missing')
        return X_clean

class SparseToDenseTransformer(BaseEstimator, TransformerMixin):
    """Convert sparse matrices to dense arrays for models that need them"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if issparse(X):
            return X.toarray()
        return X

def verify_files():
    """Verify all required files exist"""
    required_files = [
        'severity_model.joblib',
        'probability_model.joblib',
        'preprocessor.joblib',
        'data_splits.joblib'
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"- {f}")
        print("\nPlease run these scripts first:")
        print("1. data_preparation_task4.py")
        print("2. claim_probability_task4.py")
        print("3. severity_model_task4.py")
        return False
    return True

def load_models():
    """Load all required models with proper pipeline handling"""
    print("\nLoading models and data...")
    try:
        # Load models with their full pipelines
        severity_model = joblib.load('severity_model.joblib')
        prob_model = joblib.load('probability_model.joblib')
        
        # Load preprocessor separately
        preprocessor = joblib.load('preprocessor.joblib')
        
        # Load data splits
        splits = joblib.load('data_splits.joblib')
        
        print("✔ All models loaded successfully")
        return severity_model, prob_model, preprocessor, splits
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None, None, None

def calculate_premiums(prob_model, severity_model, preprocessor, X):
    """Calculate risk-based premiums with proper data handling"""
    try:
        # Create a complete pipeline for premium calculation
        premium_pipeline = Pipeline([
            ('cleaner', DataCleaner()),
            ('preprocessor', preprocessor),
            ('sparse_to_dense', SparseToDenseTransformer())
        ])
        
        # Fit and transform the data
        X_processed = premium_pipeline.fit_transform(X)
        
        # Get predictions - handle sparse outputs if needed
        if hasattr(prob_model, 'predict_proba'):
            prob_claim = prob_model.predict_proba(X_processed)[:, 1]
        else:
            prob_claim = prob_model.predict(X_processed)
            
        severity = severity_model.predict(X_processed)
        
        # Premium = (Prob * Severity) * (1 + 0.2 + 0.1) [20% expenses + 10% profit]
        premiums = prob_claim * severity * 1.3
        
        print(f"Calculated premiums for {len(premiums)} policies")
        print(f"Sample premiums (first 5): {premiums[:5]}")
        print(f"Premium statistics: Mean={np.mean(premiums):.2f}, Min={np.min(premiums):.2f}, Max={np.max(premiums):.2f}")
        return premiums
    except Exception as e:
        print(f"Error calculating premiums: {str(e)}")
        return None

def evaluate_results(y_true, y_pred, title=""):
    """Evaluate and visualize results"""
    if y_true is None or y_pred is None:
        print("Cannot evaluate - missing data")
        return None, None
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{title} Evaluation:")
    print(f"- RMSE: {rmse:.2f}")
    print(f"- R2: {r2:.2f}")
    
    # Create visualizations
    plt.figure(figsize=(12, 6))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Premium')
    plt.ylabel('Predicted Premium')
    plt.title('Actual vs Predicted')
    
    # Residual plot
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.title('Residual Distribution')
    
    plt.tight_layout()
    plot_filename = f'premium_evaluation_{title.lower().replace(" ", "_")}.png'
    plt.savefig(plot_filename)
    print(f"✔ Saved evaluation plot: {plot_filename}")
    plt.close()
    
    return rmse, r2

def main():
    print("\n" + "="*50)
    print("PREMIUM OPTIMIZATION SYSTEM")
    print("="*50)
    
    if not verify_files():
        return
    
    severity_model, prob_model, preprocessor, splits = load_models()
    if None in [severity_model, prob_model, preprocessor, splits]:
        return
    
    X_train, X_test, _, _ = splits['probability']
    
    # 1. Calculate risk-based premiums
    print("\n" + "-"*50)
    print("CALCULATING RISK-BASED PREMIUMS")
    print("-"*50)
    
    train_premiums = calculate_premiums(prob_model, severity_model, preprocessor, X_train)
    test_premiums = calculate_premiums(prob_model, severity_model, preprocessor, X_test)
    
    if train_premiums is None or test_premiums is None:
        print("Error calculating premiums - exiting")
        return
    
    # Save premium calculations
    premium_results = {
        'train_premiums': train_premiums,
        'test_premiums': test_premiums,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    joblib.dump(premium_results, 'premium_calculations.joblib', compress=3)
    print("\n✔ Saved premium calculations to 'premium_calculations.joblib'")
    
    # 2. Evaluate against actual premiums if available
    if 'CalculatedPremiumPerTerm' in X_train.columns:
        print("\n" + "-"*50)
        print("EVALUATING AGAINST ACTUAL PREMIUMS")
        print("-"*50)
        
        y_train = X_train['CalculatedPremiumPerTerm']
        y_test = X_test['CalculatedPremiumPerTerm']
        
        evaluate_results(y_train, train_premiums, "Training Set")
        evaluate_results(y_test, test_premiums, "Test Set")
    
    # 3. Generate summary report
    print("\n" + "-"*50)
    print("SUMMARY REPORT")
    print("-"*50)
    
    print("\nRisk-Based Premium Statistics:")
    stats_df = pd.DataFrame({
        'Training Set': [np.mean(train_premiums), np.std(train_premiums), np.min(train_premiums), np.max(train_premiums)],
        'Test Set': [np.mean(test_premiums), np.std(test_premiums), np.min(test_premiums), np.max(test_premiums)]
    }, index=['Mean', 'Std Dev', 'Min', 'Max'])
    print(stats_df)
    
    if 'CalculatedPremiumPerTerm' in X_train.columns:
        print("\nComparison to Actual Premiums:")
        print(f"Average difference: {np.mean(y_test - test_premiums):.2f}")
        print(f"Median difference: {np.median(y_test - test_premiums):.2f}")
        print(f"Correlation: {np.corrcoef(y_test, test_premiums)[0,1]:.2f}")
    
    print("\n" + "="*50)
    print("PREMIUM OPTIMIZATION COMPLETED SUCCESSFULLY")
    print("="*50)

if __name__ == "__main__":
    main()