import yf_etl
import yf_model_methods
import pandas as pd

database_location = 'sqlite:///yf_stock_data.sqlite'

available_tickers = yf_etl.get_tickers_in_db(database_location)

tickers = []
while True:
    print(f"Currently available tickers in database: {available_tickers}. ")

    while True:
        ans = input("Do you wish to add another ticker to the database or update one of the available tickers? Answer yes or no: ").upper()
        if ans in ["YES", "Y", "NO", "N"]:
            break
        else:
            print("Invalid input. Answer with yes/y or no/n.")

    if ans in ["YES", "Y"]:
        ticker = input("Please type in the ticker you want to add to the database or update : ").upper()
        tickers.append(ticker)
    if ans in ["NO", "N"]:
        break

if tickers:
    yf_etl.run_yf_etl(tickers, database_location)


# update available tickers after adding to db
available_tickers = yf_etl.get_tickers_in_db(database_location)

while True:
    ticker = input(f"Select data from the available tickers: {available_tickers}: ").upper()
    if ticker in available_tickers:
        break
    else:
        print("Ticker not available, try again.")


data = yf_etl.fetch_data(ticker, database_location)

# select model
available_models = ["RF", "ARMA"]
while True:
    chosen_model = input(f"Select a model : {available_models}").upper()
    if chosen_model in available_models:
        break
    else:
        print("Model not available, try again.")

if chosen_model == "RF":
    yf_model_methods.random_forest_model(ticker, data)
else:
    yf_model_methods.arma_model(ticker, data)
