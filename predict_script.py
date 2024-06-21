import pickle
import pandas as pd
import requests

def fetch_historical_data(api_key, limit=1):
    url = f"https://min-api.cryptocompare.com/data/v2/histohour"
    params = {
        'fsym': 'BTC',
        'tsym': 'USD',
        'limit': limit,
        'api_key': api_key
    }
    response = requests.get(url, params=params)
    data = response.json()['Data']['Data']
    return pd.DataFrame(data)

api_key = '17011d57142c4a08089d7a61548de4f7cd9bb98bb186872eea1889a03c47ca73'
df = fetch_historical_data(api_key)

df = df.drop(columns=['conversionType', 'conversionSymbol'])

# Convert time column to datetime
df['time'] = pd.to_datetime(df['time'], unit='s')
# Set time as index
df.set_index('time', inplace=True)
df = df[:-2:-3]

# Load the trained model from the pickle file
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Make predictions
prediction = model.predict(df.index[0])

# Save the predictions
print(f'{float(prediction.iloc[0]):.2f}')

