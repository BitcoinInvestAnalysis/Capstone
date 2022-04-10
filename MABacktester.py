import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_datareader.data as web
import datetime as dt

from scipy.optimize import brute

pio.templates.default = 'presentation'


class MABacktester():
        
    def __init__(self, symbol, MA_S, MA_L, start, end):
        self.symbol = symbol
        self.MA_S = MA_S
        self.MA_L = MA_L
        self.start = start
        self.end = end
        self.results = None 
        self.get_data()
        
    def __repr__(self):
        return "MABacktester(symbol = {}, MA_S = {}, MA_L = {}, start = {}, end = {})".format(self.symbol, self.MA_S, self.MA_L, self.start, self.end)
        
    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        company = ["AAPL", "BTC-USD", "FB", "GOOG", "MSFT", "TSLA"]
        start = dt.date(2015, 1, 1)
        end = dt.date.today()
        raw = web.DataReader(company, "yahoo", start, end)
        raw.columns = pd.MultiIndex.from_tuples(raw.columns)
        raw = raw.resample("B").mean().dropna()
        raw = raw['Close']
        raw.columns = [['Close'] * len(raw.columns), raw.columns]
        raw = raw.droplevel(level=0, axis=1)

        raw = raw[self.symbol].to_frame().dropna()
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={self.symbol: "Close"}, inplace=True)
        raw["returns"] = np.log(raw / raw.shift(1))
        raw["MA_S"] = raw.Close.rolling(self.MA_S).mean()
        raw["MA_L"] = raw.Close.rolling(self.MA_L).mean()
        self.data = raw
        
    def set_parameters(self, MA_S = None, MA_L = None):
        ''' Updates MA parameters and resp. time series.
        '''
        if MA_S is not None:
            self.MA_S = MA_S
            self.data["MA_S"] = self.data.Close.rolling(self.MA_S).mean()
        if MA_L is not None:
            self.MA_L = MA_L
            self.data["MA_L"] = self.data.Close.rolling(self.MA_L).mean()
            
    def test_strategy(self):
        ''' Backtests the trading strategy.
        '''
        data = self.data.copy().dropna()
        data["position"] = np.where(data["MA_S"] > data["MA_L"], 1, -1)
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
            fig = px.line(self.results, x=self.results.index, y=["creturns", "cstrategy"], 
            title = f"{self.symbol} Buy and Hold versus Trading strategy with MA_S = {self.MA_S} and MA_L = {self.MA_L}", 
            labels={"value":"Cumulative Return %"})
            fig.show()
            
    def plot_buys_and_sells(self):
        ''' Plots buy and sell signals
        '''
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=self.results.index, y=self.results.Close, name="Close Price", opacity=0.35),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=self.results.index, y=self.results["MA_S"], name="MA_S"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=self.results.index, y=self.results["MA_L"], name="MA_L"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=self.results.index, y=self.results['position'], name="Position"), 
            secondary_y=True
        )
        # Add figure title
        fig.update_layout(
            title_text=f"{self.symbol} Buy and Sell Signals"
        )
        # Set x-axis title
        fig.update_xaxes(title_text="Date")
        # Set y-axes titles
        fig.update_yaxes(title_text="Close Price", secondary_y=False)
        fig.update_yaxes(title_text="Buy and Sell Position", secondary_y=True)
        fig.show()
        
    def update_and_run(self, MA):
        ''' Updates MA parameters and returns the negative absolute performance (for minimization algorithm).

        Parameters
        ==========
        MA: tuple
            MA parameter tuple
        '''
        self.set_parameters(int(MA[0]), int(MA[1]))
        return -self.test_strategy()[0]
    
    def optimize_parameters(self, MA1_range, MA2_range):
        ''' Finds global maximum given the MA parameter ranges.

        Parameters
        ==========
        MA1_range, MA2_range: tuple
            tuples of the form (start, end, step size)
        '''
        opt = brute(self.update_and_run, (MA1_range, MA2_range), finish=None)
        return opt, -self.update_and_run(opt)

