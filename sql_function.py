# import modules to address environment variables
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
import os

import pandas as pd
import psycopg2
from sqlalchemy import create_engine


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
