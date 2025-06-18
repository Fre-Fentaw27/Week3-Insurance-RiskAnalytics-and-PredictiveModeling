import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import shap
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

def clean_data(X):
    """Ensure all categorical columns are strings and numeric columns are numbers"""
    X_clean = X.copy()
    
    # Identify numeric and categorical columns
    numeric_cols = X_clean.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_clean.select_dtypes(include=['object', 'category']).columns
    
    # Convert numeric columns - coerce errors to NaN
    for col in numeric_cols:
        X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
    
    # Convert categorical columns to strings
    for col in categorical_cols:
        X_clean[col] = X_clean[col].astype(str)
    
    return X_clean

def get_preprocessor(X_train):
    """Create a robust preprocessor that handles mixed types"""
    # Identify numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

def train_severity_models(X_train, X_test, y_train, y_test):
    """Train and evaluate severity models"""
    # Clean the data first
    X_train_clean = clean_data(X_train)
    X_test_clean = clean_data(X_test)
    
    # Get dynamic preprocessor
    preprocessor = get_preprocessor(X_train_clean)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # Create full pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Train model
        pipeline.fit(X_train_clean, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test_clean)
        
        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'pipeline': pipeline,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"{name} - RMSE: {rmse:.2f}, R2: {r2:.2f}")
    
    return results

def analyze_feature_importance(pipeline, X_test):
    """Perform SHAP analysis on the trained model"""
    try:
        # Get preprocessed data
        X_test_clean = clean_data(X_test)
        preprocessor = pipeline.named_steps['preprocessor']
        model = pipeline.named_steps['regressor']
        
        # Transform the test data
        X_test_processed = preprocessor.transform(X_test_clean)
        
        # Convert to dense array if sparse
        if hasattr(X_test_processed, 'toarray'):
            X_test_processed = X_test_processed.toarray()
        
        # Ensure numeric dtype
        X_test_processed = np.array(X_test_processed, dtype=np.float32)
        
        # Get feature names
        numeric_features = X_test_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X_test_clean.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # For one-hot encoded features
        if 'cat' in preprocessor.named_transformers_:
            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
            categorical_features = ohe.get_feature_names_out(categorical_features)
        
        all_features = numeric_features + categorical_features.tolist()
        
        # SHAP analysis - use TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(model)
        
        # Sample data if too large (SHAP can be memory intensive)
        if X_test_processed.shape[0] > 100:
            sample_idx = np.random.choice(X_test_processed.shape[0], 100, replace=False)
            X_sample = X_test_processed[sample_idx]
        else:
            X_sample = X_test_processed
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Summary plot
        plt.figure()
        shap.summary_plot(shap_values, X_sample, feature_names=all_features, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig('shap_summary.png')
        plt.close()
        
        return shap_values
    
    except Exception as e:
        print(f"SHAP analysis failed: {str(e)}")
        print("Attempting alternative feature importance method...")
        
        # Fallback to model's native feature importance
        if hasattr(model, 'feature_importances_'):
            # Get feature importances
            importances = model.feature_importances_
            
            # Get feature names
            numeric_features = X_test_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = X_test_clean.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if 'cat' in preprocessor.named_transformers_:
                ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
                categorical_features = ohe.get_feature_names_out(categorical_features)
            
            # Ensure we have the same number of features as importances
            all_features = numeric_features + categorical_features.tolist()
            if len(all_features) != len(importances):
                print(f"Mismatch in feature counts: {len(all_features)} features but {len(importances)} importances")
                # Use generic feature names if mismatch
                all_features = [f"feature_{i}" for i in range(len(importances))]
            
            # Create importance DataFrame
            feature_importance = pd.DataFrame({
                'feature': all_features,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Plot top features
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', 
                        data=feature_importance.head(20))
            plt.title('Feature Importance (Fallback Method)')
            plt.tight_layout()
            plt.savefig('feature_importance_fallback.png')
            plt.close()
            
            print("\nTop 20 features by importance:")
            print(feature_importance.head(20))
            
            return feature_importance
        else:
            print("No feature importance method available for this model")
            return None

if __name__ == "__main__":
    # Load data splits
    splits = joblib.load('data_splits.joblib')
    X_sev_train, X_sev_test, y_sev_train, y_sev_test = splits['severity']
    
    # Train models
    print("Training severity models...")
    results = train_severity_models(X_sev_train, X_sev_test, y_sev_train, y_sev_test)
    
    # Identify best model
    best_model_name = min(results, key=lambda k: results[k]['rmse'])
    best_model = results[best_model_name]['pipeline']
    
    print(f"\nBest model: {best_model_name} with RMSE: {results[best_model_name]['rmse']:.2f}")
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    shap_values = analyze_feature_importance(best_model, X_sev_test)
    
    # Save best model
    joblib.dump(best_model, 'severity_model.joblib')
    print("\nBest model saved to 'severity_model.joblib'")