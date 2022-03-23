# import modules to address environment variables
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

import os
import pandas_datareader.data as web
import pandas_datareader as pdr
from datetime import datetime
import datetime as dt
import requests
import time


def get_stock_daily(symbol, outputsize, datatype):
    url = "https://alpha-vantage.p.rapidapi.com/query"
    querystring = {
        "function": "TIME_SERIES_DAILY",
        "symbol": {symbol},
        "outputsize": {outputsize},
        "datatype": {datatype},
    }
    headers = {
        "x-rapidapi-host": os.getenv("x-rapidapi-host"),
        "x-rapidapi-key": os.getenv("x-rapidapi-key"),
    }
    return requests.request("GET", url, headers=headers, params=querystring)


def get_df_stock_daily(symbol, start_date="2012-01-01", end_date="2021-12-31"):
    return web.DataReader(
        symbol,
        "av-daily",
        start=datetime.strptime(start_date, "%Y-%m-%d").date(),
        end=datetime.strptime(end_date, "%Y-%m-%d").date(),
        api_key=os.getenv("x-rapidapi-key"),
    )


def get_all_focused_stocks(start_date="2012-01-01", end_date="2021-12-31"):
    symbols = ["AAPL", "BTCUSD", "FB", "GOOG", "MSFT", "TSLA"]

    df_aapl = get_df_stock_daily("AAPL", start_date, end_date)
    time.sleep(15)
    df_btcusd = get_df_stock_daily("BTCUSD", start_date, end_date)
    time.sleep(15)
    df_fb = get_df_stock_daily("FB", start_date, end_date)
    time.sleep(15)
    df_goog = get_df_stock_daily("GOOG", start_date, end_date)
    time.sleep(15)
    df_msft = get_df_stock_daily("MSFT", start_date, end_date)
    time.sleep(15)
    df_tsla = get_df_stock_daily("TSLA", start_date, end_date)

    return df_aapl, df_btcusd, df_fb, df_goog, df_msft, df_tsla
