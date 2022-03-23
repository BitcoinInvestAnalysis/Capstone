# import modules to address environment variables
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

import os
import pandas_datareader.data as web
import pandas_datareader as pdr
from datetime import datetime
import datetime as dt
import requests


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
