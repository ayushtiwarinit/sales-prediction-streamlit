import pandas as pd
import pickle
import numpy as np
import os
import json
import optuna
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import clean_and_merge_data

# Paths
sales_path = "data/raw/sales_data.csv"
transactions_path = "data/raw/transactions.csv"
clean_path = "data/processed/clean_data.csv"
junk_path = "data/junk/junk_data.csv"

data = clean_and_merge_data(sales_path, transactions_path, clean_path, junk_path)

X = data[['store_num', 'item_num', 'day', 'month', 'year', 'weekday', 'promo', 'price_scaled', 'transaction_count_scaled']]
y = data['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- HYPERPARAMETER TUNING SECTION ---

print("Starting Hyperparameter Tuning...")

# 1. Random Forest Tuning using RandomizedSearchCV
print("Tuning Random Forest...")
rf_params = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), rf_params, n_iter=3, cv=3, scoring='r2', n_jobs=-1)
rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_

# 2. XGBoost Tuning using Optuna
print("Tuning XGBoost with Optuna...")
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0)
    }
    model = XGBRegressor(**param, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return r2_score(y_test, preds)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5) # n_trials=5 for speed, increase for better results
xgb_best = XGBRegressor(**study.best_params, random_state=42)
xgb_best.fit(X_train, y_train)

# --- EVALUATION ---

models = {
    "Linear": LinearRegression(),
    "Random_Forest": rf_best,
    "XGBoost": xgb_best
}

metrics = {}
best_model_name = None
best_score = -999

for name, model in models.items():
    if name == "Linear": model.fit(X_train, y_train) # Linear doesn't need tuning
    
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    metrics[name] = {"RMSE": float(round(rmse, 2)), "MAE": float(round(mae, 2)), "R2": float(round(r2, 4))}
    pickle.dump(model, open(f"models/{name.lower()}.pkl", "wb"))

    if r2 > best_score:
        best_score = r2
        best_model_name = name

with open("outputs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

with open("outputs/best_model.txt", "w") as f:
    f.write(best_model_name)

print(f"Training and Tuning Complete! Best Model: {best_model_name}")