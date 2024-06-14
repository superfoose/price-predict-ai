import subprocess
import asyncio
import alpaca
# from Openai.noprint import get_companies
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas as pd
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
# from indicators.rsi_open import pre_open
from indicators.sma_ml import ml_sma
from indicators.pre_low import pre_low

###########  Gabbay trading keys   #######################
API_KEYg = "PKN9S8GOGNR3UNGANJT5"
API_SECRETg = "N1NsetJpmBAhSNZB7CViBOgIB9bQCAvBhfHsfodC"


# from indicators.loss_ai import ml_indicator
from indicators.ml_ind import ml_ind

API_KEY = 'PKLJ044DDL0ODIQ93YRD'
API_SECRET = 'Vb7sat3km82i4p6MC56QcPzHzeCUQIgNRj2dA9LD'
BASE_URL = 'https://paper-api.alpaca.markets'  # Use 'https://api.alpaca.markets' for live trading
# Create an instance of the Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, base_url=BASE_URL, api_version='v2')
apig = tradeapi.REST(API_KEYg, API_SECRETg, base_url=BASE_URL, api_version='v2')
openai.api_key = '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'

import requests


stocks_list = []
def get_price(symbol):
    pos = apig.get_position(symbol)
    price = pos.current_price
    return price

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


def buy(symbol, notional, extended=False, api=api):
    if not extended:
        lock = threading.Lock()

        lock.acquire()
        try:
            # Submit a market order to buy the specified symbol and quantity
            api.submit_order(
                symbol=symbol,
                notional=round(notional, 2),
                side='buy',
                type='market',
                time_in_force='day'
            )
            print(f"Bought {notional} shares of {symbol}")
            time.sleep(0.5)


        except Exception as e:

            price = yf.Ticker(symbol).info["currentPrice"]
            qty = (notional / price) // 1
            print(qty, " share of ", symbol, " Fixed bug")
            api.submit_order(
                symbol=symbol,
                qty= qty,
                side='buy',
                type='market',
                time_in_force='day'
            )
            print(f"Bought {notional} shares of {symbol}")
            time.sleep(0.5)

        except Exception as e:
            print("####################BUYING ERROR: ", e, " ######################")


        lock.release()
    else:
        prebuy(symbol, notional)


def sell(symbol, qty, extended=False, api=api):
    if not extended:
        lock = threading.Lock()
        lock.acquire()
        try:
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='day',
            )
            print(f"Sold {qty} shares of {symbol}")
            time.sleep(0.5)
        except Exception as e:
            print(f"Error selling {qty} shares of {symbol}: {e}")
        lock.release()
    else:
        presell(symbol, qty)


def prebuy(symbol, notional,api=api):
    print("Prebuy order")
    lock = threading.Lock()

    lock.acquire()
    price = get_price(symbol)#yf.Ticker(symbol).history(period="1m", interval="1m", prepost=True)["Close"].iloc[-1]
    print(price)
    print("Will order ", int(((notional / float(price))) // 1))
    try:
        # Submit a market order to buy the specified symbol and quantity
        api.submit_order(
            symbol=symbol,
            qty=int(((notional / float(price))) // 1),
            side='buy',
            type='limit',
            time_in_force='day',
            extended_hours=True,
            limit_price=(float(price) // 1 + 20) // 1
        )
        print(f"Bought {int(((notional / float(price))) // 1)} shares of {symbol}")
        time.sleep(0.5)

    except Exception:
        try:
            api.submit_order(
                symbol=symbol,
                qty=int(((notional / price)) // 1),
                side='buy',
                type='limit',
                time_in_force='day',
                extended_hours=True,
                limit_price=price + 20
            )
            print(f"Bought {notional} shares of {symbol}")
            time.sleep(0.5)

        except Exception as e:
            print(f"Error buying {int(((notional / float(price))) // 1)} shares of {symbol}: {e}")

    lock.release()


def presell(symbol, notional,api=api):
    print("Pre sell order")
    lock = threading.Lock()

    lock.acquire()
    try:
        # Submit a market order to buy the specified symbol and quantity
        api.submit_order(
            symbol=symbol,
            qty=notional,
            side='sell',
            type='limit',
            time_in_force='day',
            extended_hours=True,
            limit_price=0.01
        )
        print(f"Sold {notional} shares of {symbol}")
        time.sleep(0.5)

    except Exception:
        try:
            api.submit_order(
                symbol=symbol,
                qty=notional,
                side='sell',
                type='limit',
                time_in_force='day',
                extended_hours=True,
                limit_price=0.01
            )
            print(f"Bought {notional} shares of {symbol}2")
            time.sleep(0.5)

        except Exception as e:
            print(f"Error buying {notional} shares of {symbol}: {e}")

    lock.release()


def buy_calc(symbol, df, len_pos, positions, cash, account, extended=False):
    try:
        equity = float(account.equity)
        try:
            position_without = api.get_position(symbol)
            position = float(position_without.qty)
            print("position is ", position)
        except:
            position = 0
            print("Position is 0")

        try:
            qty_invest = equity  # / len_pos

        except ZeroDivisionError:
            qty_invest = equity

        if position == 0:
            print(len_pos)
            if len_pos == 0:
                print("You have equity ", equity)
                buy(symbol, equity, extended)
            else:
                print("You have equity ", equity)
                len_pos += 1
                for pos in positions:
                    print('Stock ', pos.symbol, " has ", pos.qty)
                    sell(pos.symbol, ((float(pos.qty) / len_pos)) // 1, extended)
                buy(symbol, qty_invest / len_pos, extended)
    except IndexError as ie:
        equity = float(account.equity)/df["close"]



def sell_calc(symbol, df, len_pos, positions, cash, account, reason, extended=False):
    equity = float(account.equity)
    try:
        position_without = api.get_position(symbol)
        position = float(position_without.qty)
    except:
        position = 0

    if position != 0:
        if len_pos == 1:
            sell(symbol, position, extended)
            print(f"Sold stock because of {reason}")
        else:
            sell(symbol, position, extended)
            print(f"Sold stock because of {reason}")
            print("You have equity ", equity)
            for pos in positions:
                if pos.symbol != symbol:
                    # len_pos -= 1
                    print("len pos: ", len_pos)
                    buy(pos.symbol, ((equity / len_pos / len_pos - 1)), extended)


def premarket(df, symbol, path):
    if len(df) >= 39:
        strat(symbol, df, path)
    else:
        print("Symbol: " + symbol)
        print(df)


def get_pre_market_price(stock, lock, num):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    url = f"https://www.marketwatch.com/investing/stock/{stock}?mod=search_symbol"
    lock.acquire()
    # num = num / 3
    response = requests.get(url)

    soup = BeautifulSoup(response.content, 'html.parser')

    pre_label_quote = soup.find('bg-quote', {'class': 'value'})
    # print(pre_label_quote)
    pre_label = pre_label_quote.get_text()
    pre_label = pre_label.replace("$", "")
    pre_label = pre_label.replace(f"{stock} : ", "")
    # print(pre_label)
    pre_market = float(pre_label)
    lock.release()
    return pre_market




def create_candles(unsorted, pre_candles):

    print("Unsorted ", unsorted)
    o = unsorted[0]
    c = unsorted[-1]
    h = max(unsorted)
    l = min(unsorted)
    now = datetime.now()
    minute = now.minute
    hour = now.hour

    pre_candle = {

        "Open": float(o),
        "High": float(h),
        "Low": float(l),
        "Close": float(c),
        "Adj Close": float(c),
        "Volume": 0,
        "Time": f"{hour}:{minute}"

    }
    pre_candles.append(pre_candle)
    print("Completed ", pd.DataFrame(pre_candles))
    return pd.DataFrame(pre_candles)


def get_symbol(company_name):
    try:
        yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
        params = {"q": company_name, "quotes_count": 1, "country": "United States"}

        res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
        data = res.json()

        company_code = data['quotes'][0]['symbol']
    except Exception:
        company_code = "Not a company"
    return company_code


def is_dataframe(variable):
    return isinstance(variable, pd.DataFrame)


def run_script_and_read_csv(path, symbol):
    # Run the specified script in a separate process
    subprocess.run(["python", path, symbol], check=True)

    data = []
    with open(f"lcdir/lc{symbol}.csv", "r"):
        df = pd.read_csv(f"lcdir/lc{symbol}.csv")

    return df[["signal", "prediction"]]


def StochRSI(series, period=14, smoothK=3, smoothD=3):
    def RSI(series, period=14):
        delta = series.diff().dropna()
        ups = delta * 0
        downs = ups.copy()
        ups[delta > 0] = delta[delta > 0]
        downs[delta < 0] = -delta[delta < 0]
        ups[ups.index[period - 1]] = np.mean(ups[:period])  # first value is sum of avg gains
        ups = ups.drop(ups.index[:(period - 1)])
        downs[downs.index[period - 1]] = np.mean(downs[:period])  # first value is sum of avg losses
        downs = downs.drop(downs.index[:(period - 1)])
        rs = ups.ewm(com=period - 1, min_periods=0, adjust=False, ignore_na=False).mean() / \
             downs.ewm(com=period - 1, min_periods=0, adjust=False, ignore_na=False).mean()
        return 100 - 100 / (1 + rs)

    # Calculate RSI
    rsi = RSI(series, period)

    # Calculate StochRSI
    stochrsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    stochrsi_K = stochrsi.rolling(smoothK).mean()
    stochrsi_D = stochrsi_K.rolling(smoothD).mean()

    return stochrsi * 100, stochrsi_K * 100, stochrsi_D * 100


def strat(symbol, df):
    print("DATAFRAME: ", symbol, df.tail(2))
    df.columns = df.columns.str.lower()

    try:
        lcdf = ml_ind(df)
        lcdf = lcdf.rolling(10).mean()
        mldf = ml_sma(df).rolling(10).mean()
        print("LCDF: ", lcdf["Predicted Trend"].iloc[-1])
        # mldf = mldf

    except Exception as e:
        print("AI error for ", symbol, e)
        lcdf = None
        mldf = None



    rsi = ta.rsi(df['close'], 7).tail(3)
    low = df['low'].iloc[-1]


    fut_rsi = pre_rsi(data=df)
    fut_avg = pre_candledata(data=df)
    current_avg = (df["close"].iloc[-1] + df["open"].iloc[-1] + df["high"].iloc[-1] + df["low"].iloc[-1]) / 4
    long_rsi = pre_low(df)

    long_pred = future_price(df)
    smp = long_pred[-1]
    last = df['close'].iloc[-1]
    last_rsi = long_rsi

    open_future = pre_candledata(df)



    pct_rsi = ((fut_rsi - rsi.iloc[-1]) / rsi.iloc[-1]) * 100
    pct_avg = ((fut_avg - current_avg) / current_avg) * 100
    pct_long = ((smp - last) / last) * 100
    pct_open = ((open_future - df['open'].iloc[-1]) / df['open'].iloc[-1]) * 100
    pct_lrsi = ((last_rsi - low) / low) * 100

    #
    print(f" {symbol} future rsi {rsi.tail(3)}  predicted: {fut_rsi} rsi going up {fut_rsi > rsi.iloc[-1]} ")

    print(f" {symbol} future avg {current_avg} {fut_avg} avg going up {fut_avg > current_avg} ")
    # # print(f" {symbol} future price {fp} {last_close} price going up {is_trend} ")
    # print(f" {symbol} topwick {top_wick} bottom wick {bottom_wick} {type(top_wick)} ")
    print("Finsihed strat")

    if lcdf is not None:
        return {"symbol": symbol, "rsi": pct_rsi, "avg": pct_avg[0], "lcdf": lcdf["Predicted Trend"].iloc[-1],
            "long": pct_long, "sma_ml": mldf["ML_pred"].iloc[-1], "fut_open": pct_open[0], "long_rsi": pct_lrsi}
    else:
        return {"symbol": symbol, "rsi": pct_rsi, "avg": pct_avg[0], "lcdf": -99,
                "long": pct_long, "sma_ml": -99, "fut_open": pct_open[0], "long_rsi": pct_lrsi}


def run1(stocks, lock, arg, path):
    print("Running")
    current_minute = time.localtime().tm_min
    time.sleep(1)

    url = "https://www.marketwatch.com/investing/stock/mdb?mod=search_symbol"
    response = requests.get(url)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    try:

        hours_label = soup.find('span', {'class': 'volume__label'}).text
    except AttributeError:
        hours_label = "None"
    #
    # pre_df1 = yf.Ticker(stocks[arg]).history(period="2d", interval='1m', prepost=True)
    # pre_df2 = yf.Ticker(stocks[arg + 1]).history(period="2d", interval='1m', prepost=True)
    # pre_df3 = yf.Ticker(stocks[arg + 2]).history(period="2d", interval='1m', prepost=True)
    #



    while True:
        # unsorted = []
        pre_candles = []
        #
        # unsorted2 = []
        pre_candles2 = []
        pre_candles3 = []

        unsorted = []

        unsorted2 = []
        unsorted3 = []



        while True:#hours_label == "None":
            current_time = time.localtime()
            minute = current_time.tm_min


            if minute % 5 == 0 and current_minute != minute: #5MINUTES
                print("Run!")
                start_time = time.time()
                import pandas as pd
                if arg == 0:
                    first_row = {"symbol": "AAAAAAA", "rsi": -19999, "avg": -9999999, "lcdf": -99999999,
                                 "long": -99999999, "sma_ml": -99999999, "fut_open": -99999999, "long_rsi": -9999999}
                    data_df = pd.DataFrame(first_row, index=[0])
                    data_df.to_csv("stock_data.csv", index=False, mode='w')


                lock = multiprocessing.Lock()
                print(stocks[arg])
                # lock.acquire()

                df1 = get_data(stocks[arg])
                df2 = get_data(stocks[arg + 1])
                df3 = get_data(stocks[arg + 2])

                # lock.release()

                stock_data1 = strat(stocks[arg], df1)
                stock_data2 = strat(stocks[arg+1], df2)
                stock_data3 = strat(stocks[arg+2], df3)
                stock_data_list = [stock_data1, stock_data2, stock_data3]
                stock_data_df = pd.DataFrame(stock_data_list)

                stock_data_df.to_csv("stock_data.csv", mode='a', index=False, header=False)

                if arg == 0:

                    stock_data_df = pd.read_csv("stock_data.csv")

                    lcdf_df = stock_data_df.sort_values(by=['lcdf'], ascending=[False], ignore_index=True)
                    rsi_df = stock_data_df.sort_values(by=['rsi'], ascending=[False], ignore_index=True)
                    avg_df = stock_data_df.sort_values(by=['avg'], ascending=[False], ignore_index=True)
                    long_df = stock_data_df.sort_values(by=['long'], ascending=[False], ignore_index=True)
                    sma_df = stock_data_df.sort_values(by=['sma_ml'], ascending=[False], ignore_index=True)
                    open_df = stock_data_df.sort_values(by=['fut_open'], ascending=[False], ignore_index=True)
                    lrsi_df = stock_data_df.sort_values(by=['long_rsi'], ascending=[False], ignore_index=True)




                    print(lcdf_df, rsi_df, avg_df)

                    import pandas as pd
                    stocks_to_buy = []

                    # Assuming df is your DataFrame
                    for stock in stocks:
                        try:
                            stock_row_rsi = rsi_df.loc[rsi_df['symbol'] == stock]
                            rsi_index = stock_row_rsi.index[0]

                            stock_row_lcdf = lcdf_df.loc[lcdf_df['symbol'] == stock]
                            lcdf_index = stock_row_lcdf.index[0]

                            stock_row_avg = avg_df.loc[avg_df['symbol'] == stock]
                            avg_index = stock_row_avg.index[0]

                            stock_row_long = long_df.loc[long_df['symbol'] == stock]
                            long_index = stock_row_long.index[0]

                            stock_row_open = open_df.loc[open_df['symbol'] == stock]
                            open_index = stock_row_open.index[0]

                            stock_row_sma = sma_df.loc[sma_df['symbol'] == stock]
                            sma_index = stock_row_sma.index[0]

                            stock_row_lrsi = lrsi_df.loc[lrsi_df['symbol'] == stock]
                            lrsi_index = stock_row_lrsi.index[0]

                            total_index = avg_index + rsi_index + lcdf_index + long_index + open_index + sma_index + lrsi_index
                            stocks_to_buy.append({"symbol": stock, "total": total_index})
                        except Exception:
                            print(Exception)
                            total_index = 1000000000000000
                            stocks_to_buy.append({"symbol": stock, "total": total_index})



                    print("Stock List: ", stocks_to_buy)
                    tbdf = pd.DataFrame(stocks_to_buy)
                    sorted_tbdf = tbdf.sort_values(by=['total'], ascending=[True])
                    print("Sorted index's: ", sorted_tbdf)
                    stocktb1 = sorted_tbdf.iloc[0]["symbol"]
                    stocktb2 = sorted_tbdf.iloc[1]["symbol"]
                    stocktb3 = sorted_tbdf.iloc[2]["symbol"]

                    print("################################## Stocks to buy: ", stocktb1, stocktb2, stocktb3)

                    account = api.get_account()
                    equity = float(account.equity)
                    # positions = []
                    try:
                        positions = api.list_positions()
                    except Exception:
                        positions = []

                    if len(positions) > 0:
                        positon1 = positions[0].symbol
                        positon1qty = positions[0].qty
                        positon2 = positions[1].symbol
                        positon2qty = positions[1].qty
                        positon3 = positions[2].symbol
                        positon3qty = positions[2].qty

                        print("previous stocks:", positon1, positon2, positon3)

                        if positon1 != stocktb1:
                            print(f"Sell {positon1} {positon1qty}  and buy {stocktb1}")
                            sell(positon1, positon1qty)
                            buy(stocktb1, (equity / 100)*30)


                        if positon2 != stocktb2:
                            print(f"Sell {positon2} {positon2qty}  and buy {stocktb2}")
                            sell(positon2, positon2qty)
                            buy(stocktb2, (equity / 100) * 30)

                        if positon3 != stocktb3:
                            print(f"Sell {positon3} {positon3qty}  and buy {stocktb3}")
                            sell(positon3, positon3qty)
                            buy(stocktb3, (equity / 100) * 30)
                    else:
                        print("No stocks owned")
                        buy(stocktb1, (equity / 100)* 30)
                        buy(stocktb2, (equity / 100) * 30)
                        buy(stocktb3, (equity / 100) * 30)






                url = "https://www.marketwatch.com/investing/stock/mdb?mod=search_symbol"
                response = requests.get(url)

                # Parse the HTML content using BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')

                try:

                    hours_label = soup.find('span', {'class': 'volume__label'}).text
                except AttributeError:
                    hours_label = "None"



                end_time = time.time()


                print(f"Total time is {end_time - start_time}")


                current_minute = minute

            # else:
            #     print(minute % 3 == 0, minute)




import multiprocessing
import threading
lock = multiprocessing.Lock()
if __name__ == '__main__':
    e = True
    print("start")



    stocks = ['SE', 'ADVM', 'BMY', 'ACGL', 'CSCO', 'MMM', 'ARW', 'UNM', 'GME', 'BAH',
              'AAPL', 'WMT', 'IBM', 'BAC', 'SDRL', 'GOOS', 'FFIE', 'DXC']

    # ['BLCO', 'AAPL', 'FANG', 'CHK', 'WMT', 'TEVA', 'DINO', 'RVMD', 'WIT', 'QDEL', 'KMB', 'PANW', 'ADI', 'JPM', 'RXRX',
    #  'AMZN', 'HDB', 'NI', 'JNJ', 'COF']

    print(stocks[5])
    hi = 1

    print("Is main")
    process1 = multiprocessing.Process(target=run1, args=(stocks, lock, 0, "runlcdir/runlc1.py"))
    process2 = multiprocessing.Process(target=run1, args=(stocks, lock, 3, "runlcdir/runlc2.py"))
    process3 = multiprocessing.Process(target=run1, args=(stocks, lock, 6, "runlcdir/runlc3.py"))
    process4 = multiprocessing.Process(target=run1, args=(stocks, lock, 9, "runlcdir/runlc4.py"))
    process5 = multiprocessing.Process(target=run1, args=(stocks, lock, 12, "runlcdir/runlc4.py"))
    process6 = multiprocessing.Process(target=run1, args=(stocks, lock, 15, "runlcdir/runlc4.py"))
    # process7 = multiprocessing.Process(target=run1, args=(stocks, lock, 18, "runlcdir/runlc4.py"))


    # sprocess1 = multiprocessing.Process(target=secondrun, args=(stocks, lock, 0, "runlcdir/runlc1.py"))
    # sprocess2 = multiprocessing.Process(target=secondrun, args=(stocks, lock, 3, "runlcdir/runlc2.py"))
    # sprocess3 = multiprocessing.Process(target=secondrun, args=(stocks, lock, 6, "runlcdir/runlc3.py"))
    # sprocess4 = multiprocessing.Process(target=secondrun, args=(stocks, lock, 9, "runlcdir/runlc4.py"))

    process1.start()
    process2.start()
    process3.start()
    process4.start()
    process5.start()
    process6.start()
    # process7.start()

    # sprocess1.start()
    # sprocess2.start()
    # sprocess3.start()
    # sprocess4.start()

    process1.join()
    process2.join()
    process3.join()
    process4.join()
    process5.join()
    process6.join()
    # process7.join()

    # sprocess1.join()
    # sprocess2.join()
    # sprocess3.join()
    # sprocess4.join()

    import yfinance as yf

    # Define the stock symbol and date range
    stock_symbol = "AAPL"  # Replace with the stock symbol of your choice
    start_date = "2020-01-01"
    end_date = "2021-12-31"

    # Fetch historical stock data using yfinance
