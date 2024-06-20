# import yfinance as yf
# import pandas as pd
# import pandas_ta as ta
# import numpy as np
from sklearn.preprocessing import MinMaxScaler
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
import os
import subprocess

import alpaca
# from Openai.noprint import get_companies
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas_ta as ta
import time
import threading

import openai
import alpaca_trade_api as tradeapi
import talib
# from indicators.advanced_macd import advanced_macd
from alpaca.common import exceptions as alpaca_except
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from alpaca.data import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.enums import DataFeed
# import datetime
from indicators.next_price import pre_rsi
from indicators.next_candle_price import pre_candledata
from indicators.long_predict import future_price
import json
from indicators.sma_ml import ml_sma

API_KEY = 'PKW149ETKNRVY0VCPNF1'
API_SECRET = 'NVjPCqpB0XD0x9H0sOc0Gv1d54DFhjbr36XJWyxx'

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

    candles = StockBarsRequest(symbol_or_symbols=symbol, timeframe=alpaca.data.TimeFrame(3, alpaca.data.TimeFrameUnit.Minute),
                               start=datetime(year1, month1, day1),
                               # start=datetime(2023, 10, 19),
                               feed=DataFeed.IEX,
                               end=datetime(year, month, day))
    # end = datetime(2023, 10, 20))

    bars = stock_client.get_stock_bars(candles)
    df = bars.df
    pd.set_option('display.max_rows', None)
    return df


df = get_data('tsla')
df = df[:-15]  # Remove the last 10 rows
print(df)

def pre_open(data):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # data = data[:-4]  # Remove the last 10 rows
    data["RSI"] = data["close"]  # Calculate RSI
    data = data[14:]  # Keep only rows with RSI values

    # Split data into training and testing sets
    train_size = int(0.8 * len(data))
    train_data = data["RSI"].values[:train_size].reshape(-1, 1)
    test_data = data["RSI"].values[train_size:].reshape(-1, 1)

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
    predicted_rsi = model.predict(scaled_test_data)
    predicted_rsi_original = scaler.inverse_transform(predicted_rsi)

    # Print the predicted RSI values
    # print(data['RSI'])
    #
    # print(data['RSI'])
    # print(predicted_rsi_original[-1])
    # print(predicted_rsi[-1] * 100)
    # print((predicted_rsi[-1] * 100)[0])
    pred_rsi = (predicted_rsi_original[-1])[0]
    return pred_rsi



# fut_rsi = pre_open(df)
# print(fut_rsi)
# rsi = ta.rsi(data["Close"], 14)
# print(fut_rsi, rsi.iloc[-1])
# # pct = (fut_rsi - rsi.iloc[-1] / rsi.iloc[-1]) * 100
# pct = ((38.955902 - 39.067835491249205) / 39.067835491249205) * 100
# print(pct)
# print(predicted_rsi)
# print(data["RSI"].to_list())

