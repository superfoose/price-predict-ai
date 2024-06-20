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






from indicators.ml_ind import ml_ind

API_KEY = 'PKO4TXBQQQ687VW9V4NR'
API_SECRET = 'eGYPbodJOZTBSttduXBCWBDe12AZr3nH3k0y7tjZ'

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


# df = get_data("ai")
# print(df)
#
# df = df[:-15]  # Remove the last 10 rows


def future_price(df):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Preprocessing: Extract relevant columns, normalize data
    data = df['close'].values.reshape(-1, 1)  # Considering 'Close' prices for example
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Split data into sequences for input and output
    def create_sequences(data, seq_length):
        x, y = [], []
        for i in range(len(data) - seq_length):
            x.append(data[i : i + seq_length])
            y.append(data[i + seq_length])
        return np.array(x), np.array(y)

    sequence_length = 7  # Number of past candles to consider
    x, y = create_sequences(scaled_data, sequence_length)

    # Splitting data into train and test sets
    train_size = int(len(x) * 0.8)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Creating and training the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_test, y_test), verbose=0)

    # Predicting the next 15 candles
    predicted_candles = []
    current_sequence = x_test[-1]  # Assuming we start prediction from the last sequence in the test set
    for i in range(sequence_length):
        pred = model.predict(current_sequence.reshape(1, sequence_length, 1))
        predicted_candles.append(pred[0, 0])
        current_sequence = np.append(current_sequence[1:], pred[0])

    # Inverse transforming the predicted values
    predicted_candles = np.array(predicted_candles).reshape(-1, 1)
    predicted_candles = scaler.inverse_transform(predicted_candles)
    return predicted_candles.flatten().tolist()
# You can now use the predicted_candles for analysis or visualization
# print(df["close"])
# print(future_price(df))