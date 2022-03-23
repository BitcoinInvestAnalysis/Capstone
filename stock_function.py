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
