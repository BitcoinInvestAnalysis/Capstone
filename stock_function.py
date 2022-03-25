import pandas as pd
import numpy as np


def add_log_returns_column(df, start_date, end_date):
    df.columns = [col.replace(" ", "_").lower() for col in df.columns]

    df.index = pd.to_datetime(df.index)
    mask = (df.index >= start_date) & (df.index <= end_date)
    df = df.loc[mask].reset_index()
    df["log_returns"] = np.log(df.close / df.close.shift())
    return df


def calc_volatility(df, start_date, end_date, trading_day):
    if "log_returns" in df.columns:
        return df["log_returns"].std() * trading_day**0.5
    else:
        df = add_log_returns_column(df, start_date, end_date)
        return df["log_returns"].std() * trading_day**0.5


def revenue(df, start, end, invest):
    df.columns = [col.replace(" ", "_").lower() for col in df.columns]
    df.index = pd.to_datetime(df.index)

    if (invest / (df.open.loc[df.index == start])[0]) <= (
        df.volume.loc[df.index == start]
    )[0]:
        profit = (df.close.loc[df.index == end])[0] / (df.close.loc[df.index == start])[
            0
        ]
        profit *= invest / (df.open.loc[df.index == start])[0]
        return profit
    else:
        print("That many Stocks are not available")


def revenue_ratio(df, start, end):
    df.columns = [col.replace(" ", "_").lower() for col in df.columns]

    profit = (df.close.loc[df.index == end])[0] / (df.close.loc[df.index == start])[0]
    return profit
