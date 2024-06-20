import os
import alpaca
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf
from datetime import datetime, timedelta
from alpaca.common import exceptions as alpaca_except
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from alpaca.data import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.enums import DataFeed
# from indicators.next_price import pre_rsi
# from indicators.next_candle_price import pre_candledata
import pandas_ta as ta




from indicators.ml_ind import ml_ind

API_KEY = 'PKPNWFQMQ83GWQEEHWZ8'
API_SECRET = 'SobdX9fNMLFaVtNeO2WdJ0VkOKCnzaFPqxomxHE4'

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


# df = get_data("wit")
# print(df)
# df = df[:-15]  # Remove the last 10 rows


def pre_low(data):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # data = data[:-4]  # Remove the last 10 rows
    # data["RSI"] = ta.rsi(data["close"], 14)  # Calculate RSI
    # data = data[14:]  # Keep only rows with RSI values
    data["RSI"] = data['open'] - data["low"]
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

# df["RSI"] = df['open'] - df["low"]
# You can now use the predicted_candles for analysis or visualization
# print(df["RSI"].tail(10))
# print(pre_low(df))