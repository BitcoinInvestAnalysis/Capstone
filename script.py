# import modules to address environment variables
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
import os

import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


# Specifing figure layout
# %matplotlib inline
plt.style.use("fivethirtyeight")
sns.set_color_codes("bright")
plot_kwds = {"alpha": 0.25, "s": 80, "linewidths": 0}
plt.rcParams["figure.figsize"] = (15, 10)


def get_data(query):
    # create a connection to the PostgreSQL server
    conn = psycopg2.connect(
        host=os.getenv("host"),
        port=os.getenv("port"),
        database=os.getenv("database"),
        options="-c search_path="
        + os.getenv("schema"),  # this special looking parameter is for the schema
        user=os.getenv("user"),
        password=os.getenv("password"),
    )
    return pd.read_sql_query(query, conn)


engine = create_engine(
    f'postgresql+psycopg2://{os.getenv("user")}:{os.getenv("password")}@{os.getenv("host")}:{os.getenv("port")}/{os.getenv("database")}',
    connect_args={"options": "-csearch_path={}".format(os.getenv("schema"))},
)


# Write records stored in a dataframe to SQL database
def write_data(engine, dataframe, table_name):
    if engine != None:
        try:
            dataframe.to_sql(
                name=table_name,  # Name of SQL table
                con=engine,  # Engine or connection
                if_exists="replace",  # Drop the table before inserting new values
                index=False,  # Write DataFrame index as a column
                chunksize=5000,  # Specify the number of rows in each batch to be written at a time
                method="multi",
            )  # Pass multiple values in a single INSERT clause
            print(f"The {table_name} table was imported successfully.")
        # Error handling
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            engine = None


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


# defining a utility function for testing the clustering algorithms
def plot_clusters(data, algorithm, args, kwds):
    # cluster the data while taking the time the process needs
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()

    # defining the colors for visual representation
    palette = sns.color_palette("bright", np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]

    # plotting the data, removing the axis and adding title and time
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title("Clusters found by {}".format(str(algorithm.__name__)), fontsize=24)
    plt.text(
        -0.5, 0.7, "Clustering took {:.2f} s".format(end_time - start_time), fontsize=14
    )
