# import modules to address environment variables
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
import os

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
