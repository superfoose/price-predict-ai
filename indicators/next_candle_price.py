# import yfinance as yf
# import pandas as pd
# import pandas_ta as ta
# import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
# from sklearn.linear_model import LinearRegression
#
# # Download historical data
# data = yf.download("MSFT", period="5d", interval="1m")
# print(data)
# data = data[:-10]
# print(data)
# # Calculate RSI
# data["RSI"] = ta.rsi(data["Close"], 14)
# data = data[14:]
#
# print(data["RSI"])
#
# # Prepare data for prediction
# rsi_data = data["RSI"].values.reshape(-1, 1)
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_rsi_data = scaler.fit_transform(rsi_data)
#
# # Split data into training and testing sets
# train_size = int(0.8 * len(scaled_rsi_data))
# train_data = scaled_rsi_data[:train_size]
# test_data = scaled_rsi_data[train_size:]
#
# # Train a linear regression model
# model = LinearRegression()
# model.fit(train_data, train_data)
#
# # Make predictions
# predicted_rsi = model.predict(test_data)
# predicted_rsi = scaler.inverse_transform(predicted_rsi)
#
# # Print the predicted RSI value
# print("Predicted RSI:", predicted_rsi[-1])
# print(predicted_rsi[-1], data["RSI"].iloc[-1])
import yfinance as yf
import pandas as pd
import talib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas_ta as ta
# Download historical data for MSFT
# Prepare data
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from alpaca.data import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.enums import DataFeed
import alpaca_trade_api as tradeapi
import alpaca


API_KEY = 'PKO4TXBQQQ687VW9V4NR'
API_SECRET = 'eGYPbodJOZTBSttduXBCWBDe12AZr3nH3k0y7tjZ'
BASE_URL = 'https://paper-api.alpaca.markets'  # Use 'https://api.alpaca.markets' for live trading
# Create an instance of the Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, base_url=BASE_URL, api_version='v2')

import requests


stocks_list = []


def get_data(symbol):
    def current_date_components():
        current_date = datetime.now()
        tommorow = current_date + timedelta(days=1)
        year = tommorow.year
        month = tommorow.month
        day = tommorow.day
        return day, month, year

    # Example usage:
    day, month, year = current_date_components()

    def yesterday_date_components():
        current_date = datetime.now()
        yesterday = current_date - timedelta(days=100)
        year = yesterday.year
        month = yesterday.month
        day = yesterday.day
        return day, month, year

    # Example usage:
    day1, month1, year1 = yesterday_date_components()
    # no keys required.
    crypto_client = CryptoHistoricalDataClient()

    # keys required
    stock_client = StockHistoricalDataClient(API_KEY, API_SECRET)

    candles = StockBarsRequest(symbol_or_symbols=symbol, timeframe=alpaca.data.TimeFrame(5, alpaca.data.TimeFrameUnit.Minute),
                               start=datetime(year1, month1, day1),
                               # start=datetime(2023, 10, 19),
                               feed=DataFeed.IEX,
                               end=datetime(year, month, day))
    # end = datetime(2023, 10, 20))

    bars = stock_client.get_stock_bars(candles)
    df = bars.df
    pd.set_option('display.max_rows', None)
    return df

# data = get_data('ai')

def pre_candledata(data):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # data = data[:-104]
    data["avg"] = (data['close'] + data['open'] + data['high'] + data['low']) / 4
    data = data[14:]  # Keep only rows with RSI values

    # Split data into training and testing sets
    train_size = int(0.8 * len(data))
    train_data = data["avg"].values[:train_size].reshape(-1, 1)
    test_data = data["avg"].values[train_size:].reshape(-1, 1)

    # Scale data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_test_data = scaler.transform(test_data)

    # Create the RNN model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(None, 1)))
    model.add(LSTM(32))
    model.add(Dense(1))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(scaled_train_data, scaled_train_data, epochs=10, batch_size=32, verbose=0)

    # Make predictions and inverse transform to original scale
    predicted_avg = model.predict(scaled_test_data)
    predicted_avg_original = scaler.inverse_transform(predicted_avg)

    # Print the predicted RSI values
    # print(data['RSI'])
    #
    # print(data["avg"])
    # print(predicted_rsi_original[-1])
    # print(predicted_rsi[-1] * 100)
    # print((predicted_rsi[-1] * 100)[0])
    pred_avg = (predicted_avg_original[-1])
    return pred_avg

# avg = pre_candledata(data)
# print((data['close'] + data['open'] + data['high'] + data['low']) / 4)
# print(avg)

