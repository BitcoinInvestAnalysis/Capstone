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


def on_balance_volume(df):
    """Add On-Balance Volume (OBV) and 
    OBV Exponential Moving Average (EMA) for given dataframe.
    
    #     :param df: pandas.DataFrame
    #     :return: pandas.DataFrame
    #     """

    OBV = []
    OBV.append(0)

    #Loop through the data set (close price) from the second row (index 1) to the end of the data set
    for i in range(1, len(df.Close)):
        if df.Close[i] > df.Close[i-1]:
            OBV.append(OBV[-1] + df.Volume[i])
        elif df.Close[i] < df.Close[i-1]:
            OBV.append(OBV[-1] - df.Volume[i])
        else:
            OBV.append(OBV[-1])
    
    #Store the OBV and OBV Exponential Moving Average (EMA) into new columns
    df['OBV'] = OBV
    df['OBV_EMA'] = df['OBV'].ewm(span=20).mean()
    
    return df


def obv_indicator(df, col1, col2):
    """Add signal when to buy and sell depend on OBV and 
    OBV Exponential Moving Average (EMA)

    #     :param df: pandas.DataFrame
    #     :param col1: pandas.Series
    #     :param col2: pandas.Series
    #     :return: pandas.DataFrame
    #     """

    sigPriceBuy, sigPriceSell = [], []
    flag = -1
    #Loop through the length of the data set
    for i in range(0, len(df)):
        # If OBV > OBV_EMA Then Buy --> col1 => 'OBV' and col2 => 'OBV_EMA'
        if df[col1][i] > df[col2][i] and flag != 1:
            sigPriceBuy.append(df['Close'][i])
            sigPriceSell.append(np.nan)
            flag = 1
        # IF OBV < OBV_EMA Then Sell
        elif df[col1][i] < df[col2][i] and flag != 0:
            sigPriceSell.append(df['Close'][i])
            sigPriceBuy.append(np.nan)
            flag = 0
        else:
            sigPriceSell.append(np.nan)
            sigPriceBuy.append(np.nan)

    df['Buy_Signal_Price'] = sigPriceBuy
    df['Sell_Signal_Price'] = sigPriceSell      
    
    return df



def revenue_random(df):
    df = df.resample("B").mean().dropna()
    df = df.reset_index()
    df['week'] = np.nan
    df['month'] = np.nan
    df['quartal'] = np.nan
    df['halfyear'] = np.nan
    df['year'] = np.nan
    df['total'] = np.nan
    for day in df.index[:-5]:
        df.week[day] = (df.close[day+5] / df.close[day]) * 100
    for day in df.index[:-22]:
        df.month[day] = (df.close[day+22] / df.close[day]) * 100
    for day in df.index[:-63]:
        df.quartal[day] = (df.close[day+63] / df.close[day]) * 100
    for day in df.index[:-126]:
        df.halfyear[day] = (df.close[day+126] / df.close[day]) * 100   
    for day in df.index[:-252]:
        df.year[day] = (df.close[day+252] / df.close[day]) * 100
    for day in df.index:
        df.total[day] = (df.close[df.index[-1]] / df.close[day]) * 100
    df = df.set_index('timestamp')
    return df