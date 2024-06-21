#17011d57142c4a08089d7a61548de4f7cd9bb98bb186872eea1889a03c47ca73

import requests
import datetime
import pandas as pd
import mlflow
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings

warnings.filterwarnings("ignore")

def fetch_historical_data(api_key, limit=2000):
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

df = df.drop(df.index[-1])

# Define the split ratio
train_size = int(len(df) * 0.8)

# Split the data
train, test = df[:train_size], df[train_size:]

# Perform Augmented Dickey-Fuller test
result = adfuller(df['close'])

# If p-value > 0.05, apply differencing
if result[1] > 0.05:
    df['close_diff'] = df['close'].diff().dropna()
    
    # Perform ADF test on differenced data
    result_diff = adfuller(df['close_diff'].dropna())

    if result_diff[1] < 0.05:
        d = 1
    else:
        d = 2
else:
    d = 0

# Based on the plots and differencing
p = 1
q = 1

# Fit ARIMA model
model = ARIMA(df['close'], order=(p, d, q))
model_fit = model.fit()

# Predicting the test set
start_index = test.index[0]
end_index = test.index[-1]
predictions = model_fit.predict(start=start_index, end=end_index, typ='levels')

# Calculate R-squared
r2 = r2_score(test['close'], predictions)
print(f'R² Score: {r2}')

with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model_fit, file)

print("Model saved successfully as trained_model.pkl")