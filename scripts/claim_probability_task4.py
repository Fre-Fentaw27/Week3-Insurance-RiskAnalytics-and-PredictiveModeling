import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from time import time

# Configure output
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

class DataCleaner(BaseEstimator, TransformerMixin):
    """Optimized data cleaner that handles mixed types"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_clean = X.copy()
        # Convert all categoricals to strings first
        cats = X_clean.select_dtypes(include=['object', 'category']).columns
        X_clean[cats] = X_clean[cats].astype(str).fillna('missing')
        # Then handle numerics
        nums = X_clean.select_dtypes(include=['number']).columns
        X_clean[nums] = X_clean[nums].apply(pd.to_numeric, errors='coerce')
        return X_clean

def load_data():
    """Optimized data loading with memory efficiency"""
    try:
        t0 = time()
        splits = joblib.load('data_splits.joblib')
        X_train, X_test, y_train, y_test = splits['probability']
        print(f"✔ Data loaded in {time()-t0:.2f}s | Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, None, None

def get_preprocessor(X_sample):
    """Create optimized preprocessor with reduced memory usage"""
    t0 = time()
    numeric_cols = X_sample.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X_sample.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    # Use handle_unknown='infrequent_if_exist' for memory efficiency
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=True))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)],
        remainder='drop', n_jobs=-1)
    
    print(f"Preprocessor created in {time()-t0:.2f}s")
    return preprocessor

def train_models(X_train, y_train, preprocessor):
    """Optimized model training with early stopping and parallel processing"""
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=500, random_state=42, class_weight='balanced', n_jobs=-1),
        'Random Forest': RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=42, 
            class_weight='balanced', n_jobs=-1),
        'XGBoost': XGBClassifier(
            n_estimators=100, max_depth=5, random_state=42, 
            eval_metric='logloss', n_jobs=-1, 
            scale_pos_weight=np.sqrt(len(y_train)/sum(y_train)))
    }
    
    results = {}
    
    for name, model in models.items():
        try:
            t0 = time()
            pipeline = Pipeline(steps=[
                ('cleaner', DataCleaner()),
                ('preprocessor', preprocessor),
                ('classifier', model)])
            
            print(f"\nTraining {name}...")
            pipeline.fit(X_train, y_train)
            train_time = time()-t0
            
            # Fast evaluation with subsampling if large dataset
            eval_size = min(10000, len(y_train))
            if len(y_train) > eval_size:
                idx = np.random.choice(len(y_train), eval_size, replace=False)
                X_eval = X_train.iloc[idx]
                y_eval = y_train.iloc[idx]
            else:
                X_eval, y_eval = X_train, y_train
            
            y_pred = pipeline.predict(X_eval)
            y_proba = pipeline.predict_proba(X_eval)[:, 1]
            
            print(f"Trained in {train_time:.2f}s | Accuracy: {accuracy_score(y_eval, y_pred):.4f}")
            print(f"ROC AUC: {roc_auc_score(y_eval, y_proba):.4f}")
            
            results[name] = pipeline
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    return results

def main():
    print("\n" + "="*50)
    print("OPTIMIZED CLAIM PROBABILITY MODEL TRAINING")
    print("="*50)
    
    # 1. Load data
    X_train, X_test, y_train, y_test = load_data()
    if X_train is None:
        return
    
    # 2. Create preprocessor using a sample (for faster initialization)
    preprocessor = get_preprocessor(X_train.sample(min(1000, len(X_train))))
    
    # 3. Train models
    models = train_models(X_train, y_train, preprocessor)
    
    if not models:
        print("\nNo models were successfully trained")
        return
    
    # 4. Select best model (based on ROC AUC with subsampling)
    eval_size = min(5000, len(y_test))
    idx = np.random.choice(len(y_test), eval_size, replace=False)
    X_eval = X_test.iloc[idx]
    y_eval = y_test.iloc[idx]
    
    best_model_name = max(
        models.keys(),
        key=lambda k: roc_auc_score(y_eval, models[k].predict_proba(X_eval)[:, 1]))
    best_model = models[best_model_name]
    print(f"\n✔ Best model: {best_model_name}")
    
    # 5. Save the best model
    try:
        joblib.dump(best_model, 'probability_model.joblib', compress=3)
        print("✔ Model saved to 'probability_model.joblib'")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)

if __name__ == "__main__":
    t_start = time()
    main()
    print(f"\nTotal runtime: {time()-t_start:.2f} seconds")