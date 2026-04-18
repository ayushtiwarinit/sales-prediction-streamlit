import pandas as pd
import pickle
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocess import clean_and_merge_data

# Paths
sales_path = "data/raw/sales_data.csv"
transactions_path = "data/raw/transactions.csv"
clean_path = "data/processed/clean_data.csv"
junk_path = "data/junk/junk_data.csv"

print("Cleaning and merging data...")
data = clean_and_merge_data(sales_path, transactions_path, clean_path, junk_path)

# Prepare Features 
X = data[['store_num', 'item_num', 'day', 'month', 'year', 'weekday', 'promo', 'price_scaled', 'transaction_count_scaled']]
y = data['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear": LinearRegression(),
    "Random_Forest": RandomForestRegressor(n_estimators=50, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=50, random_state=42)
}

metrics = {}
best_model_name = None
best_score = -999

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("Training models...")
for name, model in models.items():
    model.fit(X_train, y_train)
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

print(f"Training complete! Best Model: {best_model_name}")