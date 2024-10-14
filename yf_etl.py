
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
    df = pd.DataFrame()

    for tick in ticker:
        stock = yf.Ticker(tick)
        stock_data = stock.history(start=start_date, end=end_date, interval="1d")
        stock_data['symbol'] = tick
        df = pd.concat([df, stock_data])

    # Clean
    df = df.reset_index()
    df = df.drop(['Dividends', 'Stock Splits', 'High', 'Low', ], axis=1)
    df['Returns'] = df['Close'] - df['Close'].shift(1)
    df = df.drop(index=0)

    print(df)
    return df