import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = 'presentation'


def plot_candlestick(df):
    df.columns = [col.replace(" ", "_").lower() for col in df.columns]
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index, open=df.open, high=df.high, low=df.low, close=df.close
            )
        ]
    )
    fig.show()


def plot_ohlc(df):
    df.columns = [col.replace(" ", "_").lower() for col in df.columns]
    fig = go.Figure(
        data=[
            go.Ohlc(
                x=df.index, open=df.open, high=df.high, low=df.low, close=df.close
            )
        ]
    )
    fig.show()
