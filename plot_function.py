import plotly.graph_objects as go


def plot_candlestick(df):
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.timestamp, open=df.open, high=df.high, low=df.low, close=df.close
            )
        ]
    )
    fig.show()


def plot_ohlc(df):
    fig = go.Figure(
        data=[
            go.Ohlc(
                x=df.index, open=df.open, high=df.high, low=df.low, close=df.close
            )
        ]
    )
    fig.show()
