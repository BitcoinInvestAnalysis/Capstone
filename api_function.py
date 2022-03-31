import pandas_datareader.data as web
import datetime as dt
import time


def get_df_stock_daily(symbol, start_date="2015-01-01", end_date=dt.date.today()):
    return web.DataReader(symbol, "yahoo", start=start_date, end=end_date)


def get_all_focused_stocks(start_date="2015-01-01", end_date=dt.date.today()):
    symbols = ["AAPL", "BTC-USD", "FB", "GOOG", "MSFT", "TSLA"]

    df_aapl = get_df_stock_daily("AAPL", start_date, end_date)
    time.sleep(5)
    df_btcusd = get_df_stock_daily("BTC-USD", start_date, end_date)
    time.sleep(5)
    df_fb = get_df_stock_daily("FB", start_date, end_date)
    time.sleep(5)
    df_goog = get_df_stock_daily("GOOG", start_date, end_date)
    time.sleep(5)
    df_msft = get_df_stock_daily("MSFT", start_date, end_date)
    time.sleep(5)
    df_tsla = get_df_stock_daily("TSLA", start_date, end_date)

    return df_aapl, df_btcusd, df_fb, df_goog, df_msft, df_tsla
