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

def send_discord_message(message, webhook_url):
    payload = {
        "content": message
    }
    response = requests.post(webhook_url, json=payload)
    return response.status_code == 204  # Discord returns 204 No Content for successful requests

api_key = '17011d57142c4a08089d7a61548de4f7cd9bb98bb186872eea1889a03c47ca73'
discord_webhook_url = 'https://discord.com/api/webhooks/1253709695202361405/L9ZRfiTEwbOKEgpt2_Qzk0FO9TEwbmQGD_Oy--yBKc9ADhhBim1NOmgJQIJgnJQFLRez'

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

# Format the prediction message
message = f"""
---------------------------------------------------------------------------------

Retraining model...

Model retrained successfully !

Predicting...

BTC/USD Prediction for the next hour's close: {float(prediction.iloc[0]):.2f}

---------------------------------------------------------------------------------
"""

# Send the message to Discord
if send_discord_message(message, discord_webhook_url):
    print("Message sent successfully")
else:
    print("Failed to send message")

