import streamlit as st
import pandas as pd
import pickle
import json
import plotly.express as px
from utils.forecast import generate_future_data

st.set_page_config(page_title="Advanced Sales Prediction Dashboard", layout="wide")

@st.cache_data
def load_data():
    data = pd.read_csv("data/processed/clean_data.csv")
    data['date'] = pd.to_datetime(data['date'])
    return data

data = load_data()

# Load all 3 models
models = {
    "Linear": pickle.load(open("models/linear.pkl", "rb")),
    "Random_Forest": pickle.load(open("models/random_forest.pkl", "rb")),
    "XGBoost": pickle.load(open("models/xgboost.pkl", "rb"))
}

with open("outputs/metrics.json") as f:
    metrics = json.load(f)

with open("outputs/best_model.txt") as f:
    best_model_name = f.read()

# --- Sidebar Setup ---
st.sidebar.title("Configuration")
st.sidebar.markdown(f"🏆 **Best Model Detected:** `{best_model_name}`")
st.sidebar.markdown("---")

forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 7, 90, 30)

store_options = ["Overall (All Stores)"] + sorted(data['store_id'].unique().tolist())
item_options = ["Overall (All Items)"] + sorted(data['item_id'].unique().tolist())

store_choice = st.sidebar.selectbox("Select Store", store_options)
item_choice = st.sidebar.selectbox("Select Product (Item)", item_options)

# --- Dashboard Main Panel ---
st.title("📊 Integrated Sales Forecasting")

# 1. Visualization of ALL Model Accuracy Graphs
# 1. Visualization of ALL Model Accuracy Graphs
with st.expander("Model Evaluation & Accuracy Metrics", expanded=True):
    # Create the dataframe from the metrics JSON
    metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'Model'})
    
    # NEW: Calculate Accuracy Percentage from R2 Score
    metrics_df['Accuracy (%)'] = (metrics_df['R2'] * 100).round(2).astype(str) + "%"
    
    # Reorder columns for better readability
    metrics_df = metrics_df[['Model', 'Accuracy (%)', 'R2', 'RMSE', 'MAE']]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Accuracy Scoreboard")
        # Display the table with the new Accuracy Percentage column
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        
    with col2:
        # Show the bar chart comparison
        fig_acc = px.bar(
            metrics_df, 
            x='Model', 
            y=['R2', 'RMSE', 'MAE'], 
            barmode='group', 
            title="Metric Comparison across Models"
        )
        st.plotly_chart(fig_acc, use_container_width=True)

st.markdown("---")
st.header("🔮 Granular Sales Predictions")

store_val = data['store_id'].mode()[0] if store_choice == "Overall (All Stores)" else store_choice
item_val = data['item_id'].mode()[0] if item_choice == "Overall (All Items)" else item_choice

# Calculate base features for forecasting
subset = data[(data['store_id'] == store_val) & (data['item_id'] == item_val)]
if subset.empty:
    subset = data

avg_price = subset['price_scaled'].mean()
avg_trans = subset['transaction_count_scaled'].mean()

# Predict Logic - Generate predictions for ALL models at once
dates_df, future_features = generate_future_data(forecast_days, store_val, item_val, avg_price, avg_trans)

for model_name, model_obj in models.items():
    dates_df[model_name] = model_obj.predict(future_features)

# 2. Monthly / Daily / Weekly Breakdowns (Using Best Model for KPIs)
st.markdown(f"##### Quick Stats (Based on {best_model_name} predictions)")
colA, colB, colC = st.columns(3)

daily_avg = dates_df[best_model_name].mean()
colA.metric("Avg Daily Sales", f"{daily_avg:.2f}")

if forecast_days >= 7:
    weekly_avg = dates_df.set_index('date').resample('W')[best_model_name].sum().mean()
    colB.metric("Avg Weekly Sales", f"{weekly_avg:.2f}")
else:
    colB.metric("Avg Weekly Sales", "N/A (< 7 days)")

if forecast_days >= 30:
    monthly_avg = dates_df.set_index('date').resample('ME')[best_model_name].sum().mean()
    colC.metric("Avg Monthly Sales", f"{monthly_avg:.2f}")
else:
    colC.metric("Avg Monthly Sales", "N/A (< 30 days)")

# 3. Multi-Model Trend Plot Chart
st.subheader(f"Future Trend Comparison: {store_choice} | {item_choice}")

# Plot all three model columns on the same graph
fig_forecast = px.line(
    dates_df, 
    x='date', 
    y=['Linear', 'Random_Forest', 'XGBoost'], 
    markers=True, 
    title=f"Forecast Comparison for next {forecast_days} days",
    labels={'value': 'Predicted Sales', 'variable': 'Machine Learning Model'}
)
st.plotly_chart(fig_forecast, use_container_width=True)