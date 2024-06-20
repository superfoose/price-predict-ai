import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf

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
print(df, 'sddsfd')

signal_length = 11  # You can change this value
sma_signal = True  # You can change this value

lin_reg = False  # You can change this value
linreg_length = 11  # You can change this value

bopen = df['open'] if lin_reg else df['open'].rolling(linreg_length).mean()
bhigh = df['high'] if lin_reg else df['high'].rolling(linreg_length).mean()
blow = df['low'] if lin_reg else df['low'].rolling(linreg_length).mean()
bclose = df['close'] if lin_reg else df['close'].rolling(linreg_length).mean()
data_reg = {
    'open': bopen,
    'high': bhigh,
    'low': blow,
    'close': bclose
}
dfreg = pd.DataFrame(data_reg)
print(dfreg, 'REG')

r = bopen < bclose

signal = df['close'].rolling(signal_length).mean() if sma_signal else df['close'].ewm(span=signal_length, adjust=False).mean()

# Create a DataFrame to hold the candlestick data
ohlc = dfreg.copy()
ohlc.index = df.index

# Plot candlestick chart
mpf.plot(ohlc, type='candle', style='default', title="LinReg Candles", ylabel='Price', volume=False)

# Plot signals as points
for i in range(len(df)):
    if r[i]:
        plt.plot(i, bopen[i], 'go')
        plt.plot(i, bhigh[i], 'go')
        plt.plot(i, blow[i], 'go')
        plt.plot(i, bclose[i], 'go')


    else:
        plt.plot(i, bopen[i], 'ro')
        plt.plot(i, bhigh[i], 'ro')
        plt.plot(i, blow[i], 'ro')
        plt.plot(i, bclose[i], 'ro')


# Plot the signal line
plt.plot(signal, color='white', label='Signal')

plt.legend()
plt.show()