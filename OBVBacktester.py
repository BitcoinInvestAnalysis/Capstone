import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import datetime as dt

plt.style.use("seaborn")


class OBVBacktester():

    def __init__(self, symbol, start, end):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.results = None
        self.get_data()

    def __repr__(self):
        return "OBVBacktester(symbol = {}, start = {}, end = {})".format(self.symbol, self.start, self.end)
        
    def get_data(self):
        ''' Retrieves and prepares the data.
            Add On-Balance Volume (OBV) and 
            OBV Exponential Moving Average (EMA) for given data
        '''
        company = ["AAPL", "BTC-USD", "FB", "GOOG", "MSFT", "TSLA"]
        start = dt.date(2015, 1, 1)
        end = dt.date.today()
        raw = web.DataReader(company, "yahoo", start, end)
        raw.columns = pd.MultiIndex.from_tuples(raw.columns)
        raw = raw.resample("B").mean().dropna()
        raw = raw[['Close', 'Volume']]
        raw = raw.swaplevel(axis = 1).sort_index(axis = 1)

        raw = raw[self.symbol]
        raw = raw.loc[self.start:self.end]
        self.data = raw
        raw["returns"] = np.log(raw.Close / raw.Close.shift(1))
        OBV = []
        OBV.append(0)
        for i in range(1, len(self.data.Close)):
            if self.data.Close[i] > self.data.Close[i-1]:
                OBV.append(OBV[-1] + self.data.Volume[i])
            elif self.data.Close[i] < self.data.Close[i-1]:
                OBV.append(OBV[-1] - self.data.Volume[i])
            else:
                OBV.append(OBV[-1])
        
        self.data['OBV'] = OBV
        self.data['OBV_EMA'] = self.data['OBV'].ewm(span=20).mean()
                
    def test_strategy(self):
        ''' Backtests the trading strategy.
        '''                
        data = self.data.copy().dropna()
        data["position"] = np.where(data["OBV"] > data["OBV_EMA"], 1, -1)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        # absolute performance of the strategy
        perf = data["cstrategy"].iloc[-1]
        # out-/underperformance of strategy
        outperf = perf - data["creturns"].iloc[-1]
        return round(perf, 6), round(outperf, 6)

    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to buy and hold.
        '''
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else:
            title = f"{self.symbol} Buy and Hold versus Trading strategy"
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
            