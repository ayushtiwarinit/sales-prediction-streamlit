import pandas as pd

# Load existing transactions
trans_path = "data/raw/transactions.csv"
trans = pd.read_csv(trans_path)

# Create intentional junk rows
junk_rows = pd.DataFrame({
    'transaction_id': ['TXN-ERR1', 'TXN-ERR2', 'TXN-ERR3'],
    'transaction_date': ['2022-01-01', None, '2022-01-03'], # Missing date
    'items': ['item_1', 'item_2', None], # Missing item
    'price': [-50.00, 76.72, 0.00], # Negative and Zero prices
    'store_id': ['store_2', 'store_1', 'store_3'],
    'phone_no': ['123-456-7890', '098-765-4321', None]
})

# Append and save back to the raw file
trans = pd.concat([trans, junk_rows], ignore_index=True)
trans.to_csv(trans_path, index=False)

print("Faulty data successfully injected into transactions.csv!")