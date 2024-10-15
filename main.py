import yf_etl
import yf_model_methods
import pandas as pd

tickers = ['NVDA']
dfs = yf_etl.create_yf_df(tickers)

print(dfs['NVDA'])
print(dfs['NVDA'].columns)
