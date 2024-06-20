import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression


# Define the input parameters
signal_length = 11
sma_signal = True

lin_reg = True
linreg_length = 11

# Create a pandas DataFrame with your OHLCV data
# Replace the example data with your own data
data_b = yf.download("TSLA", period='1d', interval='1m')
# Simulated input data, replace this with your actual data
# For example, you can read OHLC data from a CSV file into a pandas DataFrame
# Replace this sample data with your actual data
data = {
    'open': data_b['Open'],
    'high': data_b['High'],
    'low': data_b['Low'],
    'close': data_b['Close']
}
df = pd.DataFrame(data)
# Initialize empty DataFrames for bopen, bhigh, blow, and bclose
bopen_df = pd.DataFrame()
bhigh_df = pd.DataFrame()
blow_df = pd.DataFrame()
bclose_df = pd.DataFrame()
# Calculate bopen, bhigh, blow, and bclose based on lin_reg flag
if lin_reg:
    lr = LinearRegression()

    # Calculate bopen and store in the DataFrame
    lr.fit(np.arange(len(df)).reshape(-1, 1), df['open'].values.reshape(-1, 1))
    bopen_df['bopen'] = lr.predict(np.arange(len(df)).reshape(-1, 1)).flatten() - 0.2

    # Calculate bhigh and store in the DataFrame
    lr.fit(np.arange(len(df)).reshape(-1, 1), df['high'].values.reshape(-1, 1))
    bhigh_df['bhigh'] = lr.predict(np.arange(len(df)).reshape(-1, 1)).flatten() - 0.15

    # Calculate blow and store in the DataFrame
    lr.fit(np.arange(len(df)).reshape(-1, 1), df['low'].values.reshape(-1, 1))
    blow_df['blow'] = lr.predict(np.arange(len(df)).reshape(-1, 1)).flatten() - 0.18

    # Calculate bclose and store in the DataFrame
    lr.fit(np.arange(len(df)).reshape(-1, 1), df['close'].values.reshape(-1, 1))
    bclose_df['bclose'] = lr.predict(np.arange(len(df)).reshape(-1, 1)).flatten()- 0.19
else:
    # If lin_reg is False, store the original values
    bopen_df['bopen'] = df['open']
    bhigh_df['bhigh'] = df['high']
    blow_df['blow'] = df['low']
    bclose_df['bclose'] = df['close']

# Concatenate the DataFrames to create a new DataFrame with bopen, bhigh, blow, and bclose
result_df = pd.concat([bopen_df, bhigh_df, blow_df, bclose_df], axis=1)

# Calculate the 'r' condition
result_df['r'] = result_df['bopen'] < result_df['bclose']

# Calculate the 'signal' based on sma_signal flag
if sma_signal:
    result_df['signal'] = result_df['bclose'].rolling(window=signal_length, min_periods=1).mean()
else:
    result_df['signal'] = result_df['bclose'].ewm(span=signal_length, adjust=False).mean()

# Display the resulting DataFrame
print(result_df)






