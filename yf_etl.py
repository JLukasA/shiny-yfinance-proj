import pandas as pd
import yfinance as yf
import sqlite3
from sqlalchemy import create_engine
import pandas as pd
pd.options.mode.chained_assignment = None


""" Fetches financial data from Yahoo using one or more tickers, puts it in a dictionary of pandas DataFrames and returns the dictionary. """


def create_yf_df(tickers: list[str]) -> dict:
    from datetime import datetime
    import datetime

    end_date = datetime.datetime.now() - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=365)
    dfs = {}

    for ticker in tickers:
        if not check_ticker_validity(ticker):
            print(f"Data for ticker '{ticker}' is not available. Skipping.")
            continue
        stock = yf.Ticker(ticker)
        stock_data = stock.history(start=start_date, end=end_date, interval="1d")
        stock_data = stock_data.reset_index()
        stock_data = stock_data.drop(['Dividends', 'Stock Splits', 'High', 'Low', ], axis=1)
        stock_data['Returns'] = stock_data['Close'] - stock_data['Close'].shift(1)
        stock_data = stock_data.dropna()
        stock_data = stock_data.rename(columns={'Date': 'timestamp', 'Open': 'open', 'Close': 'close', 'Volume': 'volume', 'Returns': 'returns'})
        dfs[ticker] = stock_data

    return dfs


def check_ticker_validity(ticker: str) -> bool:
    info = yf.Ticker(ticker).history(period='7d', interval='1d')
    return len(info) > 0


def yf_df_to_db(engine, tickers: list[str], dfs: dict):

    for ticker in tickers:
        dfs[ticker].to_sql('yf_stock_data', engine, index=False, if_exists='append')


def check_data_validity(df: pd.DataFrame) -> bool:

    # check if empty
    if df.empty:
        print("No data downloaded.")
        return False

    # check primary key
    if pd.Series(df['date']).is_unique:
        pass
    else:
        raise Exception("Primary key is not unique.")

    # check for null values
    if df.isnull().values.any():
        raise Exception("Null values found in data.")

    return True


def run_yf_etl(tickers: list[str]):
    DATABASE_LOCATION = 'sqlite:///yf_stock_data.sqlite'

    # fetch data using yfinance
    data = create_yf_df(tickers)

    # validate fetched data
    for ticker in data:
        if check_data_validity(data[ticker]):
            print(f"{ticker} data valid.")

    # load data to SQL database
    engine = create_engine(DATABASE_LOCATION)
    conn = sqlite3.connect('yf_stock_data.sqlite')
    print("Database connection initiated.")
    cursor = conn.cursor()
    for ticker in data:

        sql_query = f""""
        CREATE TABLE IF NOT EXISTS {ticker}(
            timestamp VARCHAR(200),
            open DOUBLE(9, 2),
            close DOUBLE(9, 2),
            volume INT(12),
            returns DOUBLE(9, 2),
            CONSTRAINT primary_key_constraint PRIMARY KEY (timestamp)
        )
        """
        cursor.execute(sql_query)
        try:
            data[ticker].to_sql(ticker, engine, index=False, if_exists='append')
        except:
            print("Failed uploading data to database.")

    conn.close()
    print("Database connection concluded.")
