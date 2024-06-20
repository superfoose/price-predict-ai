import numpy as np
import pandas as pd
import yfinance as yf

# Python equivalent variables for Pine Script inputs
signal_length = 11  # You can change this to any integer between 1 and 200
sma_signal = True  # You can change this to either True or False

# Python equivalent variables for Pine Script inputs
lin_reg = True  # You can change this to either True or False
linreg_length = 11  # You can change this to any integer between 1 and 200

# Sample OHLC price data (you would use your own data)
df = yf.download("AAPL", period='1d', interval='1m')
print(df)
open_prices = df['Open']
high_prices = df['High']
low_prices = df['Low']
close_prices = df['Close']

# Calculate bopen, bhigh, blow, bclose based on lin_reg
if lin_reg:
    from scipy.stats import linregress
    linreg = linregress(np.arange(len(open_prices)), open_prices)
    bopen = linreg.slope * np.arange(len(open_prices)) + linreg.intercept
    linreg = linregress(np.arange(len(high_prices)), high_prices)
    bhigh = linreg.slope * np.arange(len(high_prices)) + linreg.intercept
    linreg = linregress(np.arange(len(low_prices)), low_prices)
    blow = linreg.slope * np.arange(len(low_prices)) + linreg.intercept
    linreg = linregress(np.arange(len(close_prices)), close_prices)
    bclose = linreg.slope * np.arange(len(close_prices)) + linreg.intercept
else:
    bopen = open_prices
    bhigh = high_prices
    blow = low_prices
    bclose = close_prices

# Calculate r (open < close)
r = bopen < bclose

# Calculate signal based on sma_signal
if sma_signal:
    # Calculate Simple Moving Average (SMA)
    bclose_series = pd.Series(bclose)
    signal = bclose_series.rolling(window=signal_length).mean()
else:
    # Calculate Exponential Moving Average (EMA)
    bclose_series = pd.Series(bclose)
    signal = bclose_series.ewm(span=signal_length, adjust=False).mean()


data = {
    'bopen': bopen ,
    'bhigh': bhigh ,
    'blow': blow ,
    'bclose': bclose,
    'bullish' : r
}
pd.set_option('display.max_rows', False)
df = pd.DataFrame(data)

# Print the DataFrame
print(df)