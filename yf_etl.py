import pandas as pd
import yfinance as yf
import sqlite3
from sqlalchemy import create_engine
import pandas as pd
pd.options.mode.chained_assignment = None


def create_yf_df(tickers: list[str]) -> dict:
    from datetime import datetime
    import datetime

    end_date = datetime.datetime.now() - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=3000)
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


def check_data_validity(df: pd.DataFrame) -> bool:

    # check if empty
    if df.empty:
        print("No data downloaded.")
        return False

    # check primary key
    if pd.Series(df['timestamp']).is_unique:
        pass
    else:
        raise Exception("Primary key is not unique.")

    # check for null values
    if df.isnull().values.any():
        raise Exception("Null values found in data.")

    return True


def run_yf_etl(tickers: list[str], database_location: str):
    # fetch data using yfinance
    data = create_yf_df(tickers)

    # validate fetched data
    for ticker in data:
        if check_data_validity(data[ticker]):
            print(f"{ticker} data valid.")

    # load data to SQL database
    engine = create_engine(database_location)
    s = database_location.replace('sqlite:///', '')
    conn = sqlite3.connect(s)
    print("Database connection initiated in preparation for upload.")
    cursor = conn.cursor()
    for ticker in data:

        # check database for ticker data, filter out if exists
        try:
            old_data = pd.read_sql(f"SELECT timestamp FROM {ticker}", conn)
            print(f"Data for {ticker} present in database, filtering out before uploading.")
        except (sqlite3.OperationalError, pd.io.sql.DatabaseError) as e:
            old_data = pd.DataFrame(columns=['timestamp'])  # empty DataFrame to skip filtering
            print(f"No data for {ticker} available in database, no filtering needed.")

        # ensure same type for filtering with isin()
        old_data['timestamp'] = pd.to_datetime(old_data['timestamp'])
        data[ticker]['timestamp'] = pd.to_datetime(data[ticker]['timestamp'])
        # generate filtered dataframe for upload
        new_data = data[ticker][~data[ticker]['timestamp'].isin(old_data['timestamp'])]

        try:
            new_data.to_sql(ticker, engine, index=False, if_exists='append')
            print(f"{ticker} data uploaded.")
        except:
            print("Failed uploading data to database.")

    conn.close()
    print("Database connection concluded.")


def fetch_data(ticker: str, database_location: str) -> pd.DataFrame:
    s = database_location.replace('sqlite:///', '')
    conn = sqlite3.connect(s)
    print("Database connection for fetching data initiated.")
    print(f"Fetching data for {ticker}")
    try:
        data = pd.read_sql(f"SELECT * FROM {ticker}", conn)
        print(f"Data for {ticker} fetched from database.")
    except (sqlite3.OperationalError, pd.io.sql.DatabaseError) as e:
        print(f"No data for {ticker} available in database.")
        data = pd.DataFrame()
    conn.close()
    print("Database connection concluded.")
    return data


def get_tickers_in_db(database_location: str) -> list:
    s = database_location.replace('sqlite:///', '')
    conn = sqlite3.connect(s)
    print("Database connection for fetching tickers initiated.")
    cursor = conn.cursor()
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    try:
        cursor.execute(query)
        tables = cursor.fetchall()  # tuples containing a single string, table name aka ticker
        tickers = [table[0] for table in tables]  # create list of strings
    except sqlite3.Error as e:
        print(f"Error fetching tickers: {e}")
        tickers = []
    conn.close()
    print("Database connection concluded.")
    return tickers
