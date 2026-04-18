import pandas as pd

def generate_future_data(days, store_id, item_id, base_price, base_transactions):
    future_dates = pd.date_range(start=pd.Timestamp.today(), periods=days)
    
    future_df = pd.DataFrame({'date': future_dates})
    future_df['day'] = future_df['date'].dt.day
    future_df['month'] = future_df['date'].dt.month
    future_df['year'] = future_df['date'].dt.year
    future_df['weekday'] = future_df['date'].dt.dayofweek
    
    # Store categorical for reference
    future_df['store_id'] = store_id
    future_df['item_id'] = item_id
    
    # Parse numbers to feed to the ML model
    future_df['store_num'] = int(''.join(filter(str.isdigit, store_id))) if any(char.isdigit() for char in store_id) else 1
    future_df['item_num'] = int(''.join(filter(str.isdigit, item_id))) if any(char.isdigit() for char in item_id) else 1
    
    # Use moving averages for future unknown variables
    future_df['promo'] = 0 
    future_df['price_scaled'] = base_price
    future_df['transaction_count_scaled'] = base_transactions
    
    features = future_df[['store_num', 'item_num', 'day', 'month', 'year', 'weekday', 'promo', 'price_scaled', 'transaction_count_scaled']]
    return future_df, features