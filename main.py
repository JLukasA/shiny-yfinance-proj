import yf_etl
from yf_model_methods import random_forest_model
import pandas as pd
import sqlite3
from sqlalchemy import create_engine


tickers = ['NVDA', 'MSFT']

yf_etl.run_yf_etl(tickers)
