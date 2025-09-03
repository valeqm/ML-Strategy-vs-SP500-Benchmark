"""
train_models.py
- Loads features.parquet
- Defines X / y sets: TO_PREDICT (target_5d_up), numeric columns, dummies, etc.
- Splits data by time (train/val/test) to avoid leakage
- Trains Decision Tree, Random Forest, XGBoost classifiers
- Performs hyperparameter tuning using RandomizedSearchCV
- Saves models, scaler, and selected feature columns
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

os.makedirs("models", exist_ok=True)

def load_data(path="data/processed/features.parquet"):
    """Load feature dataset and drop rows without target."""
    df = pd.read_parquet(path)
    df = df.dropna(subset=['target_5d_up'])  # drop last rows without target
    return df

def prepare_Xy(df):
    """Prepare X (features) and y (target) sets."""
    exclude = ['date','ticker','target_5d_up','future_5d_ret']
    numeric_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [float, int]]
    X = df[numeric_cols].fillna(method='ffill').fillna(0)
    y = df['target_5d_up'].astype(int)
    meta = df[['date','ticker']]
    return X, y, meta, numeric_cols

def time_split_index(meta, train_ratio=0.7, val_ratio=0.15):
    """Split data by date to create train, validation, and test sets."""
    dates = pd.to_datetime(meta['date']).sort_values().unique()
    n = len(dates)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]
    return train_dates, val_dates, test_dates

def fit_and_save(X_train, y_train, X_val, y_val, feature_cols):
    """Fit models, perform hyperparameter tuning, and save models and scaler."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Class weight ratio for XGBoost
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    # ================= Decision Tree =================
    dt = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    dt_params = {'max_depth':[3,5,7,9,12], 'min_samples_split':[2,5,10,20]}
    dt_search = RandomizedSearchCV(
        dt, dt_params, n_iter=8, cv=3,
        scoring='average_precision', n_jobs=-1, random_state=42
    )
    dt_search.fit(X_train_s, y_train)
    joblib.dump(dt_search.best_estimator_, "models/decision_tree.joblib")
    print("DT best:", dt_search.best_params_)

    # ================= Random Forest =================
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")
    rf_params = {'n_estimators':[50, 100, 150], 'max_depth':[5,8,12,None], 'min_samples_split':[2,5,10]}
    rf_search = RandomizedSearchCV(
        rf, rf_params, n_iter=12, cv=3,
        scoring='average_precision', n_jobs=-1, random_state=42
    )
    rf_search.fit(X_train_s, y_train)
    joblib.dump(rf_search.best_estimator_, "models/random_forest.joblib")
    print("RF best:", rf_search.best_params_)

    # ================= XGBoost =================
    xgb = XGBClassifier(
        use_label_encoder=False, eval_metric='logloss',
        n_jobs=-1, random_state=42,
        scale_pos_weight=pos_weight
    )
    xgb_params = {
        'n_estimators':[100,200],
        'max_depth':[3,5,7],
        'learning_rate':[0.01,0.05,0.1],
        'subsample':[0.6,0.8,1.0]
    }
    xgb_search = RandomizedSearchCV(
        xgb, xgb_params, n_iter=12, cv=3,
        scoring='average_precision', n_jobs=-1, random_state=42
    )
    xgb_search.fit(X_train_s, y_train)
    joblib.dump(xgb_search.best_estimator_, "models/xgboost.joblib")
    print("XGB best:", xgb_search.best_params_)

    # ================= Save scaler & feature columns =================
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(feature_cols, "models/feature_cols.joblib")

    # ================= Validation scores =================
    val_scores = {}
    models = {
        'DT': dt_search.best_estimator_,
        'RF': rf_search.best_estimator_,
        'XGB': xgb_search.best_estimator_
    }

    for name, model in models.items():
        preds = model.predict_proba(X_val_s)[:,1]
        auc = roc_auc_score(y_val, preds)
        ap = average_precision_score(y_val, preds)
        val_scores[name] = ap
        print(f"{name} Val AUC: {auc:.4f} | AvgPrecision: {ap:.4f}")

    # ================= Select best model =================
    best_name = max(val_scores, key=val_scores.get)
    best_model = models[best_name]
    joblib.dump(best_model, "models/best_model.joblib")
    print(f"Best model: {best_name} saved to models/best_model.joblib")

if __name__ == "__main__":
    df = load_data()
    X, y, meta, numeric_cols = prepare_Xy(df)
    train_dates, val_dates, test_dates = time_split_index(meta)
    train_idx = meta['date'].isin(train_dates)
    val_idx = meta['date'].isin(val_dates)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    fit_and_save(X_train, y_train, X_val, y_val, numeric_cols)



