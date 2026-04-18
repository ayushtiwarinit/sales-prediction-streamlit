import matplotlib
matplotlib.use('Agg')  # This line prevents the Tcl/Tkinter error

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_eda():
    print("Starting Exploratory Data Analysis (EDA)...")
    
    # 1. Load the cleaned data
    data_path = "data/processed/clean_data.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run train_models.py first.")
        return

    df = pd.read_csv(data_path)
    
    # Create output folder for images
    output_dir = "outputs/eda"
    os.makedirs(output_dir, exist_ok=True)
    
    # We will analyze these core numerical columns
    num_cols = ['sales', 'price', 'transaction_count']
    
    # --- STEP A: IDENTIFY SKEWNESS ---
    print("\n--- 📊 SKEWNESS ANALYSIS ---")
    skewness = df[num_cols].skew()
    print(skewness.to_string())
    print("* Note: Skewness > 1 (Right-skewed) or < -1 (Left-skewed) indicates extreme values pushing the average.")

    # --- STEP B: IDENTIFY OUTLIERS (IQR Method) ---
    print("\n--- 🚨 OUTLIER DETECTION ---")
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"{col.capitalize()}: {len(outliers)} outliers detected out of {len(df)} total records.")

    # --- STEP C: DISTRIBUTION PLOTS ---
    print("\nGenerating Distribution Plots...")
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(num_cols, 1):
        plt.subplot(1, 3, i)
        sns.histplot(df[col], kde=True, bins=30, color='royalblue')
        plt.title(f'Distribution of {col.capitalize()}')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/distribution_plots.png", dpi=300)
    plt.close()

    # --- STEP D: OUTLIER BOXPLOTS ---
    print("Generating Outlier Boxplots...")
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(num_cols, 1):
        plt.subplot(1, 3, i)
        sns.boxplot(x=df[col], color='darkorange')
        plt.title(f'Boxplot of {col.capitalize()}')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/outlier_boxplots.png", dpi=300)
    plt.close()

    # --- STEP E: CORRELATION HEATMAP ---
    print("Generating Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    
    # Select features for the correlation matrix
    corr_cols = ['sales', 'price', 'transaction_count', 'day', 'month', 'year', 'weekday', 'promo']
    corr_matrix = df[corr_cols].corr()
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300)
    plt.close()

    print(f"\n✅ EDA Complete! All insights printed above. High-res plots saved to the '{output_dir}/' folder.")

if __name__ == "__main__":
    run_eda()