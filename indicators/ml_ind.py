# import talib
import yfinance as yf
import pandas_ta as ta
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from datetime import datetime, timedelta
from alpaca.common import exceptions as alpaca_except
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from alpaca.data import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.enums import DataFeed
API_KEY = 'PKRT2KUEGX3HC0DVP9UU'
API_SECRET = 'j4NuAkvb2bl2n5UUD9McPjsBUZcQybL5XHIUbt5E'

def get_data(symbol):
    def current_date_components():
        current_date = datetime.now()
        tommorow = current_date + timedelta(days=10)
        year = tommorow.year
        month = tommorow.month
        day = tommorow.day
        return day, month, year

    # Example usage:
    day, month, year = current_date_components()

    def yesterday_date_components():
        current_date = datetime.now()
        yesterday = current_date - timedelta(days=10)
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

    candles = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute,
                               start=datetime(year1, month1, day1),
                               # start=datetime(2023, 10, 19),
                               feed=DataFeed.IEX,
                               end=datetime(year, month, day))
    # end = datetime(2023, 10, 20))

    bars = stock_client.get_stock_bars(candles)
    df = bars.df
    pd.set_option('display.max_rows', None)
    return df


# df = get_data("ETNB")
#



def ml_ind(df):
    # Calculate technical indicators using pandas_ta
    df.ta.sma(length=10, append=True)  # Example: Simple Moving Average (SMA)
    df.ta.rsi(length=14, append=True)  # Example: Relative Strength Index (RSI)
    # print(df)
    df.ta.sma(length=14, append=True, close=ta.rsi(df["close"]))  # Example: Relative Strength Index (RSI)
    df.ta.stoch(append=True)
    df.ta.macd(append=True)
    df.ta.rsi(length=5, append=True)
    df.ta.rsi(length=7, append=True)

    def generate_labels(df):
        df['Trend'] = 0  # Initialize the Trend column
        # df.loc[(df['Close'] > df['SMA_10']) & (df["RSI_14"] > 50) & (df["RSI_14"] < 71), 'Trend'] = 1  # Bullish trend
        df.loc[((df["RSI_7"] > 49) & (df["RSI_7"] < 70)), 'Trend'] = 1  # Bullish trend

        # df.loc[df['Close'] <= df['SMA_10'], 'Trend'] = -1  # Bearish trend
        df.loc[(df["RSI_7"] < 49) | (df["RSI_7"] > 70), 'Trend'] = -1  # Bearish trend

    generate_labels(df)

    # Prepare the feature matrix (X) and target labels (y)
    features = ['RSI_14', "SMA_14", "STOCHk_14_3_3", "STOCHd_14_3_3", "SMA_10", "MACD_12_26_9", "RSI_5"]  # Add more indicators if needed

    X = df[features]
    y = df['Trend']

    # Copy the original DataFrame to create a test set
    df_test = df.copy()

    split_point = len(X) // 3
    # print(split_point, len(X) - split_point)
    X_train, X_test, y_train, y_test = X.head(len(X) - split_point), X.tail(split_point), y.head(len(X) - split_point), y.tail(split_point)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # print(X_train, X_test)
    # Create and train a decision tree classifier
    clf = DecisionTreeClassifier(max_features=4, min_samples_leaf=100, max_depth=4, random_state=1024,
                                 max_leaf_nodes=100)
    clf.fit(X_train, y_train)
    # print('Training Accuracy : ',
    #       metrics.accuracy_score(y_train,
    #                              clf.predict(X_train)) * 100)
    # print('Validation Accuracy : ',
          # metrics.accuracy_score(y_test,
          #                        clf.predict(X_test)) * 100)    # Make predictions on the test data
    y_pred = clf.predict(X_test)
    pd.set_option('display.max_rows', None)
    # Create a new DataFrame containing the test data and predictions
    # print(X_test)
    test_results = X_test.copy()
    test_results['Actual Trend'] = y_test
    test_results['Predicted Trend'] = y_pred
    return test_results

# pd.set_option('display.max_rows', None)
# #
# print(df, len(df))
#
# print(ml_ind(df))
#
# sma = ta.sma(df["close"], length=4)
# deriative = sma - sma.shift(1)
# top_der = sorted(deriative.tail(200), reverse=True)
# avg = top_der[4] / 10
# print(f"der {deriative} {avg}")