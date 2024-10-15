
""" Fetches financial data from Yahoo using one or more tickers, puts it in a pandas DataFrame and returns it. """


def create_yf_df(ticker):
    from datetime import datetime
    import datetime
    import pandas as pd
    import sqlite3
    import sqlalchemy
    import yfinance as yf

    end_date = datetime.datetime.now() - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=365)

    dfs = {}

    for tick in ticker:
        stock = yf.Ticker(tick)
        stock_data = stock.history(start=start_date, end=end_date, interval="1h")
        stock_data = stock_data.reset_index()
        stock_data = stock_data.drop(['Dividends', 'Stock Splits', 'High', 'Low', ], axis=1)
        stock_data['Returns'] = stock_data['Close'] - stock_data['Close'].shift(1)
        stock_data = stock_data.dropna()
        dfs[tick] = stock_data

    return dfs
