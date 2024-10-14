

def random_forest_model(ticker, df):
    # import libs
    import math
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_score

    # create target
    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)

    predictors = ["Close", "Volume"]
    # create additional predictors based on horizons
    horizons = [2, 5, 25, 50, 250]
    for horizon in horizons:
        rolling_mean = df.rolling(horizon).mean()
        horizon_ratio = f"Close_ratio_{horizon}"
        df[horizon_ratio] = df["Close"] / rolling_mean["Close"]
        horizon_trend = f"Trend_{horizon}"
        df[horizon_trend] = df.shift(1).rolling(horizon).sum()["Target"]
        predictors += [horizon_ratio, horizon_trend]
    df = df.dropna()

    # generate model, train, predict, plot
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)  # OBS N ESTIMATORS LOW
    predictions, predictions_50 = backtest(df, model, predictors, confidence=55)
    PS = precision_score(predictions["Target"], predictions["Predictions"]).round(4)
    PS50 = precision_score(predictions_50["Target"], predictions_50["Predictions"]).round(4)
    rfplot(predictions, predictions_50, PS, PS50, ticker, df)

    def predict(train_data, test_data, predictors, model, confidence):
        model.fit(train_data[predictors], train_data["Target"])
        predictions = model.predict_proba(test_data[predictors])[:, 1]
        predictions_conf = predictions.copy()
        predictions_50 = predictions.copy()
        predictions_conf[predictions >= confidence / 100] = 1
        predictions_conf[predictions < confidence / 100] = 0
        predictions_50[predictions >= 50 / 100] = 1
        predictions_50[predictions < 50 / 100] = 0
        predictions_conf = pd.Series(predictions_conf, index=test_data.index, name="Predictions")
        predictions_50 = pd.Series(predictions_50, index=test_data.index, name="Predictions")
        combined = pd.concat([test_data["Target"], predictions_conf], axis=1)
        combined_50 = pd.concat([test_data["Target"], predictions_50], axis=1)
        return combined, combined_50

    def backtest(df, model, predictors, confidence):
        (rows, cols) = df.shape
        start = math.ceil(0.50 * rows)
        step = math.ceil(0.05 * rows)
        predictions_df = []
        predictions_df_50 = []
        for i in range(start, rows, step):
            train_data = df.iloc[0:i].copy()
            test_data = df.iloc[i: (i + step)].copy()
            prediction, prediction_50 = predict(train_data, test_data, predictors, model, confidence)
            predictions_df.append(prediction)
            predictions_df_50.append(prediction_50)
        return pd.concat(predictions_df), pd.concat(predictions_df_50)

    def rfplot(preds, preds_50, PS, PS50, ticker, df):
        # plot
        fig, axs = plt.subplots(2, 2, figsize=(18, 9))
        # Upper left
        axs[0, 0].plot(df["Close"])
        axs[0, 0].set_title(f"Historic Closing price {ticker}")
        # Upper right
        axs[0, 1].plot(preds["Target"][-150:], label="Target")
        axs[0, 1].plot(preds["Predictions"][-150:], label="Predictions (60%)")
        axs[0, 1].plot(preds_50["Predictions"][-150:], label="Predictions (50%)")
        axs[0, 1].set_title("Prediction target and predictions")
        axs[0, 1].legend(loc="upper right")
        # Lower left
        axs[1, 0].text(0.5, 0.7, f"Precision score (60% confidence) : {PS}", ha="center", va="center", fontsize=24)
        axs[1, 0].text(0.5, 0.3, f"Precision score (50% confidence): {PS50}", ha="center", va="center", fontsize=24)
        axs[1, 0].axis("off")
        axs[1, 0].set_title("Precision score")
        # Lower right
        axs[1, 1].axis("off")  # Turn off axis for the empty subplot
        axs[1, 1].text(0.5, 0.7, f"Precision score measures how many times ...",  ha="center", va="center", fontsize=20)
        # Adjust layout to prevent clipping of titles
        plt.tight_layout()
        plt.show()


def LSTM_model(ticker, df):
    # import
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # creates a spupervised frame
    def series_to_supervised(data, n_in, n_out, dropnan=True):
        # n_vars = 1 if type(data) is list else data.shape[1]
        n_vars = 1
        df = pd.DataFrame(data)
        # print("in series_to_supervised, df head {}".format(df.shape))
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    # Frame a time series as a supervised learning dataset for univariate multi-step forecasting.
    def differentiate(data, interval=1):
        diff = list()
        for i in range(interval, len(data)):
            value = data[i] - data[i - interval]
            diff.append(value)
        return pd.Series(diff)

    # perform stationarization/differentiation, rescaling, split into training/testing data and transform to supervised data set
    def prep_data(data, number_of_forecasts, lags, forecast_length):
        # differentiate data
        raw_data = data.values
        diff_data = differentiate(raw_data)
        # print("shape diff_data {}".format(diff_data.shape))
        diff_vals = diff_data.values
        diff_vals = diff_vals.reshape(len(diff_vals), 1)
        # print("shape diff_vals {}".format(diff_vals.shape))
        # scale data
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_vals = scaler.fit_transform(diff_vals)
        scaled_vals = scaled_vals.reshape(len(scaled_vals), 1)
        # create supervised learning frame
        supervised_data = series_to_supervised(scaled_vals, lags, forecast_length)
        supervised_vals = supervised_data.values
        # split into training and testing data and transform
        train_data, test_data = supervised_vals[0:-number_of_forecasts], supervised_vals[-number_of_forecasts:]

        return scaler, train_data, test_data

    # invert differenced value

    def inverse_diff(observation, forecast):
        # make list for storing inverted value, invert and append first forecast
        inv = list()
        inv.append(forecast[0] + observation)
        # invert and append other forecasts
        for i in range(1, len(forecast)):
            inv.append(forecast[i]+inv[i-1])
        return inv

    # invert transformed forecasted values
    # Kom på felet: Forecasts är 30 lång, borde inte vara det.
    # Ska matcha number_of_forecasts.

    def inverse_transform(data, forecasts, scaler, forecast_length):
        # make list for storing inverted values
        inv = list()
        # print(f"length forecasts : {len(forecasts)}")
        # print(f"length data : {len(data)}")

        for i in range(len(forecasts)):
            # array from forecast
            forecast = np.array(forecasts[i])
            # print(f"in inverse_transform, forecast shape {forecast.shape}")
            # print(f"len forecast : {len(forecast)}")
            forecast = forecast.reshape(1, forecast_length)
            # invert scale
            inv_sc = scaler.inverse_transform(forecast)
            inv_sc = inv_sc[0, :]
            # invert difference
            idx = len(data) - number_of_forecasts + i - 1
            # print(f"idx value : {idx}")
            obs = data.values[idx]
            inv_diff = inverse_diff(obs, inv_sc)
            # append to list
            inv.append(inv_diff)
        return inv

    # make a single forecast
    def lstm_forecast(model, data, batch_size):
        # reshape into [sample, timesteps, features]
        data = data.reshape(1, 1, len(data))
        # make forecast
        forecast = model.predict(data, batch_size=batch_size)
        return [x for x in forecast[0, :]]

    # perform the model forecasting
    def make_forecasts(model, test_data, batch_size, lags):
        forecasts = list()
        for i in range(len(test_data)):
            X, y = test_data[i, 0:lags], test_data[i, lags:]
            # make forecast and append
            forecast = lstm_forecast(model=model, data=X, batch_size=batch_size)
            forecasts.append(forecast)
        return forecasts

    # build and fit LSTM to training data
    def construct_lstm(data, lags, batch_size, epochs, neurons, dropout_rate):
        # prepare data
        train_data, test_data = data[:, 0:lags], data[:, lags:]
        train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1])
        # print(f"shape training data : {train_data.shape}")
        # construct model
        model = Sequential()
        model.add(LSTM(neurons, return_sequences=True, batch_input_shape=(batch_size, train_data.shape[1], train_data.shape[2]), stateful=True, dropout=dropout_rate))
        model.add(Dense(test_data.shape[1]))
        model.compile(optimizer="adam", loss="mean_squared_error")
        # fit model
        for i in range(epochs):
            model.fit(train_data, test_data, batch_size=batch_size, verbose=0, shuffle=False)
            # model.reset_states()
        return model

    # compute RMSE for output values
    def eval_rmse(data, forecasts, forecast_length):
        rmse = list()
        for i in range(forecast_length):
            true_val = [row[i] for row in data]
            forecasted_val = [row[i] for row in forecasts]
            rmse_val = np.sqrt(mean_squared_error(true_val, forecasted_val))
            rmse.append(rmse_val)
        return rmse

    def plot_lstm(rmse_vals, ticker, true_vals, forecasts, forecast_length):
        # plot
        fig, axs = plt.subplots(2, 2, figsize=(18, 9))
        # Upper left
        plot_forecasts(axs, 0, 0, ticker, true_vals, forecasts, forecast_length)
        # Upper right

        # Lower left
        plot_rmse(axs, 1, 0, ticker, rmse_vals)
        # Lower right
        plt.tight_layout()
        return fig

    def plot_rmse(axs, x, y, ticker, rmse_vals):
        vals = np.array(rmse_vals)
        axs[x, y].plot(range(1, len(vals)+1), vals)
        axs[x, y].set_xlabel("Forecast length")
        axs[x, y].set_ylabel("RMSE")
        axs[x, y].set_title(f"RMSE {ticker}")

    def plot_forecasts(axs, x, y, ticker, input_data, forecasts, forecast_length):
        # plot last 30 values
        last_30 = input_data[-30:]
        axs[x, y].plot(last_30.values, color="blue")
        # print(f"len true vals : {len(input_data)}, len forecasts : {forecast_length}")
        for i in range(len(forecasts)):
            l = len(last_30) - forecast_length + i - 1
            r = l + len(forecasts[i]) + 1
            xa = [z for z in range(l, r)]
            ya = [last_30[l]] + forecasts[i]
            axs[x, y].plot(xa, ya, color="red")
        axs[x, y].set_xlabel("")
        axs[x, y].set_ylabel("")
        axs[x, y].set_title(f"Forecasts {ticker}")

    def run(ticker, input_data, forecast_length, time_lags, n_epochs, batch_size, n_neurons, dropout, number_of_forecasts):
        # prepare data, construct and fit model, make forecasts
        input_data = input_data['Close']
        scaler, scaled_training_data, scaled_testing_data = prep_data(data=input_data, number_of_forecasts=number_of_forecasts, lags=time_lags, forecast_length=forecast_length)
        model = construct_lstm(data=scaled_training_data, lags=time_lags, batch_size=batch_size, epochs=n_epochs, neurons=n_neurons, dropout_rate=dropout)
        forecasts = make_forecasts(model=model, test_data=scaled_testing_data, batch_size=batch_size, lags=time_lags)

        # inverse transform data
        forecasts = inverse_transform(input_data, forecasts, scaler, (forecast_length))
        true_vals = [row[time_lags:] for row in scaled_testing_data]
        true_vals = inverse_transform(input_data, true_vals, scaler, (forecast_length))

        # evaluation and plot
        rmse_vals = eval_rmse(true_vals, forecasts, forecast_length)
        fig = plot_lstm(rmse_vals, ticker, input_data, forecasts, (number_of_forecasts + forecast_length - 1))
        return fig

    forecast_length = 3
    number_of_forecasts = 5
    time_lags = 15

    # DL model parameters
    n_epochs = 50
    batch_size = 3  # match forecast length
    n_neurons = 10
    dropout = 0.1
    # drop rows to match data shape to batch size
    rows_to_drop = len(df.index) % batch_size
    df = df.drop(df.index[:rows_to_drop])
    # run
    fig = run(ticker, df, forecast_length, time_lags, n_epochs, batch_size, n_neurons, dropout, number_of_forecasts)
    return fig
