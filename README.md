# 📊 Advanced Sales Forecasting & Data Engineering Pipeline

An end-to-end Machine Learning regression project designed to predict continuous numerical sales values. This system features a robust data engineering pipeline, comprehensive Exploratory Data Analysis (EDA), and automated hyperparameter tuning to ensure peak model performance.

Built for the Hackathon Use Case: **Machine Learning Regression Model for Predictive Analytics**.

---

## 🎯 Project Objective
The goal is to provide businesses with a reliable tool for forecasting sales at a granular level (Store and Item specific). The project emphasizes **Data Quality** by implementing a strict reconciliation process to handle faulty records, ensuring that the predictive models are trained only on high-quality data.

---

## 🚀 Complete Workflow & Tasks Performed

### 1. Data Ingestion & Reconciliation (Step 1 & 2)
* **Multi-Source Ingestion:** Merged `sales_data.csv` and `transactions.csv` to combine historical sales with customer transaction volumes.
* **Junk Data Pipeline:** Implemented an automated "Quarantine" system. Any row containing missing values, negative prices, or zero sales is isolated into `data/junk/junk_data.csv` with a `junk_reason` label for auditing.
* **Data Integrity:** Ensured that the final training set is clean, deduplicated, and consistent.

### 2. Exploratory Data Analysis - EDA (Step 3)
* **Skewness Analysis:** Identified that `transaction_count` is highly right-skewed (1.46), influencing the choice of non-linear models.
* **Outlier Detection:** Used the **Interquartile Range (IQR)** method to mathematically identify over 5,500 sales outliers.
* **Correlation Mapping:** Generated heatmaps to visualize the relationship between promotions, price, and volume.

### 3. Feature Engineering & Scaling (Step 4)
* **Temporal Features:** Extracted Year, Month, Day, and Weekday to capture seasonality.
* **Numerical Extraction:** Processed categorical IDs (Store/Item) into numerical formats for model compatibility.
* **Standardization:** Utilized `StandardScaler` to normalize features, preventing the model from being biased by different units of measurement.

### 4. Hyperparameter Tuning (Step 5)
Moved beyond default parameters to find the optimal mathematical configuration:
* **RandomizedSearchCV:** Used for Random Forest to efficiently explore the parameter space.
* **Optuna (Bayesian Optimization):** Used for XGBoost to intelligently "search" for the best learning rate and tree depth based on previous trial results.

### 5. Deployment & Visualization (Step 6)
An interactive **Streamlit Dashboard** provides:
* **Accuracy Scoreboard:** Real-time comparison of R², RMSE, and MAE across all models.
* **Multi-Model Forecast:** Interactive Plotly charts comparing Linear, Random Forest, and XGBoost predictions simultaneously.
* **Granular Filters:** Specific forecasting for any Store or Product ID.

---

## 🛠️ Tech Stack
* **Core:** Python 3.13
* **ML Frameworks:** Scikit-Learn, XGBoost, Optuna
* **Data Science:** Pandas, NumPy, Scipy
* **Visualization:** Matplotlib, Seaborn, Plotly
* **Web App:** Streamlit

---

## 📂 Project Structure

```text
SALES-PREDICTION-STREAMLIT/
├── data/
│   ├── raw/                # Original Datasets
│   ├── processed/          # Cleaned & Scaled Data
│   └── junk/               # Quarantined faulty data
├── models/
│   ├── train_models.py     # ML Pipeline + Tuning (Optuna)
│   └── *.pkl               # Optimized Model Artifacts
├── outputs/
│   ├── eda/                # Distribution & Correlation Plots
│   ├── metrics.json        # Accuracy Scores
│   └── best_model.txt      # Top performing model ID
├── utils/
│   ├── preprocess.py       # Data Cleaning & Reconciliation
│   └── forecast.py         # Future date generation logic
├── app.py                  # Streamlit Dashboard UI
├── eda.py                  # Statistical Analysis Script
└── README.md               # Project Documentation
