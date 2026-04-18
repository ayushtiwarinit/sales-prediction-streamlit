import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def clean_and_merge_data(sales_path, transactions_path, clean_path, junk_path):
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    os.makedirs(os.path.dirname(junk_path), exist_ok=True)

    # 1. Load Data
    sales = pd.read_csv(sales_path)
    if 'Unnamed: 0' in sales.columns:
        sales = sales.drop(columns=['Unnamed: 0'])
    transactions = pd.read_csv(transactions_path)

    # 2. Clean Sales Data
    missing_sales = sales[sales.isnull().any(axis=1)].copy()
    missing_sales['junk_reason'] = 'Missing Values (Sales)'
    
    invalid_sales = sales[sales['sales'] <= 0].copy()
    invalid_sales['junk_reason'] = 'Negative/Zero Sales'

    # Filter out the bad sales data
    sales_clean = sales.dropna().drop_duplicates()
    sales_clean = sales_clean[sales_clean['sales'] > 0]

    # 3. Clean Transactions Data
    missing_trans = transactions[transactions.isnull().any(axis=1)].copy()
    missing_trans['junk_reason'] = 'Missing Values (Transactions)'

    invalid_trans = transactions[transactions['price'] <= 0].copy()
    invalid_trans['junk_reason'] = 'Negative/Zero Price (Transactions)'
    
    # Filter out the bad transactions data
    trans_clean = transactions.dropna().drop_duplicates()
    trans_clean = trans_clean[trans_clean['price'] > 0]

    # 4. Compile all junk data and save
    junk_data = pd.concat([missing_sales, invalid_sales, missing_trans, invalid_trans], ignore_index=True)
    
    if not junk_data.empty:
        junk_data.to_csv(junk_path, index=False)

    # 5. Process Transactions (Aggregate by Date, Store, and Item)
    trans_agg = trans_clean.groupby(['transaction_date', 'store_id', 'items']).size().reset_index(name='transaction_count')
    trans_agg.rename(columns={'transaction_date': 'date', 'items': 'item_id'}, inplace=True)

    # 6. Merge Clean Data
    data = pd.merge(sales_clean, trans_agg, on=['date', 'store_id', 'item_id'], how='left')
    data['transaction_count'] = data['transaction_count'].fillna(0)

    # 7. Feature Engineering
    data['date'] = pd.to_datetime(data['date'])
    data['day'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    data['weekday'] = data['date'].dt.dayofweek
    
    data['store_num'] = data['store_id'].str.extract(r'(\d+)').astype(int)
    data['item_num'] = data['item_id'].str.extract(r'(\d+)').astype(int)

    scaler = StandardScaler()
    data[['price_scaled', 'transaction_count_scaled']] = scaler.fit_transform(data[['price', 'transaction_count']])

    # Save Cleaned Data
    data.to_csv(clean_path, index=False)
    return data