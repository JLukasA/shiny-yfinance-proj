import pandas as pd
import yfinance as yf

""" Fetches financial data from Yahoo using one or more tickers, puts it in a pandas DataFrame and returns it. """


def create_yf_df(tickers: list[str]) -> dict:
    from datetime import datetime
    import datetime
    import pandas as pd

    end_date = datetime.datetime.now() - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=365)
    dfs = {}

    for ticker in tickers:
        if not check_ticker_validity(ticker):
            print(f"Data for ticker '{ticker}' is not available. Skipping.")
        stock = yf.Ticker(ticker)
        stock_data = stock.history(start=start_date, end=end_date, interval="1d")
        stock_data = stock_data.reset_index()
        stock_data = stock_data.drop(['Dividends', 'Stock Splits', 'High', 'Low', ], axis=1)
        stock_data['Returns'] = stock_data['Close'] - stock_data['Close'].shift(1)
        stock_data = stock_data.dropna()
        stock_data = stock_data.rename(columns={'Date': 'date', 'Open': 'open', 'Close': 'close', 'Volume': 'volume', 'Returns': 'returns'})
        dfs[ticker] = stock_data

    return dfs


def check_ticker_validity(ticker: str) -> bool:
    info = yf.Ticker(ticker).history(period='7d', interval='1d')
    return len(info) > 0


def yf_df_to_db(engine, tickers, dfs):

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
