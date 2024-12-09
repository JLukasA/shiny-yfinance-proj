def random_forest_model(ticker, df):
    # import libs
    import math
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_score

    def generate_additional_data(df: pd.DataFrame):
        # create target
        df["tomorrow"] = df["close"].shift(-1)
        df["target"] = (df["tomorrow"] > df["close"]).astype(int)
        predictors = ["close", "volume", "returns"]
        # create additional predictors based on horizons
        horizons = [2, 5, 10, 25, 50]
        for horizon in horizons:
            rolling_mean = df.select_dtypes('number').rolling(horizon).mean()
            horizon_ratio = f"close_ratio_{horizon}"
            df[horizon_ratio] = df["close"] / rolling_mean["close"]
            horizon_trend = f"trend_{horizon}"
            df[horizon_trend] = df["target"].shift(1).rolling(horizon).sum()
            predictors += [horizon_ratio, horizon_trend]
        df = df.dropna()
        return df, predictors

    def predict(train_data, test_data, predictors, model, confidence):
        model.fit(train_data[predictors], train_data["target"])
        predictions = model.predict_proba(test_data[predictors])[:, 1]
        predictions_conf = predictions.copy()
        predictions_50 = predictions.copy()
        predictions_conf[predictions >= confidence / 100] = 1
        predictions_conf[predictions < confidence / 100] = 0
        predictions_50[predictions >= 50 / 100] = 1
        predictions_50[predictions < 50 / 100] = 0
        predictions_conf = pd.Series(predictions_conf, index=test_data.index, name="predictions")
        predictions_50 = pd.Series(predictions_50, index=test_data.index, name="predictions")
        combined = pd.concat([test_data["target"], predictions_conf], axis=1)
        combined_50 = pd.concat([test_data["target"], predictions_50], axis=1)
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

    def plot_results(preds, preds_50, PS, PS50, ticker, df, condifence):
        # plot
        fig, axs = plt.subplots(2, 2, figsize=(18, 9))
        # Upper left
        axs[0, 0].plot(df["close"])
        axs[0, 0].set_title(f"Historic Closing price {ticker}")
        # Upper right
        axs[0, 1].plot(preds["target"][-150:], label="Target")
        axs[0, 1].plot(preds["predictions"][-150:], label="Predictions (60%)")
        axs[0, 1].plot(preds_50["predictions"][-150:], label="Predictions (50%)")
        axs[0, 1].set_title("Prediction target and predictions")
        axs[0, 1].legend(loc="upper right")
        # Lower left
        axs[1, 0].text(0.5, 0.7, f"Precision score ({confidence}% confidence) : {PS}", ha="center", va="center", fontsize=24)
        axs[1, 0].text(0.5, 0.3, f"Precision score (50% confidence): {PS50}", ha="center", va="center", fontsize=24)
        axs[1, 0].axis("off")
        axs[1, 0].set_title("Precision score")
        # Lower right
        axs[1, 1].axis("off")  # Turn off axis for the empty subplot
        axs[1, 1].text(0.5, 0.7, f"Precision score measures how many times ...",  ha="center", va="center", fontsize=20)
        # Adjust layout to prevent clipping of titles
        plt.tight_layout()
        plt.show()

    # run model
    df, predictors = generate_additional_data(df)
    # generate model, train, predict, plot
    confidence = 55
    model = RandomForestClassifier(n_estimators=500, min_samples_split=50, random_state=1)  # OBS N ESTIMATORS LOW
    predictions, predictions_50 = backtest(df, model, predictors, confidence)
    PS = precision_score(predictions["target"], predictions["predictions"]).round(4)
    PS50 = precision_score(predictions_50["target"], predictions_50["predictions"]).round(4)
    plot_results(predictions, predictions_50, PS, PS50, ticker, df, confidence)


def arma_model(ticker, df):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from sklearn.metrics import mean_absolute_error
    import pandas as pd
    import numpy as np
    from scipy import stats
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    def prepare_data(df: pd.DataFrame) -> pd.Series:
        data = pd.DataFrame()
        data['logreturns'] = np.log(df.close) - np.log(df.close.shift(1))
        data = data.dropna()
        idx = int(len(data) * 0.9)
        train_data = data['logreturns'][:idx]
        test_data = data['logreturns'][idx:]
        full_data = data['logreturns']
        return full_data, train_data, test_data

    def get_model_order(data: pd.Series):
        import itertools
        max_p = 8
        max_q = 8
        res_df = pd.DataFrame(columns=['order', 'AIC', 'BIC'])
        for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
            if p == 0 and q == 0:
                continue
            temp_model = sm.tsa.ARIMA(data, order=(p, 0, q), trend='n')
            res = temp_model.fit()
            res_df.loc[len(res_df.index)] = [f'({p}, 0, {q})', res.aic, res.bic]

        # sort based on BIC
        res_df = res_df.sort_values(by='BIC')

        # get top 5
        top_5 = res_df.head(5)

        # extract and return p, q for best model order, return with top 5
        order = res_df.iloc[0]['order']
        p, q = map(int, order.strip("()").split(", ")[::2])
        return p, q, top_5

    def test_adf(resid):
        adf_result = sm.tsa.adfuller(resid)
        adf_statistic = adf_result[0]
        adf_critical_value = adf_result[4]['5%']
        pval = adf_result[1]
        print("ADF statistic: ", adf_statistic)
        print("p-value: ", pval)
        adf_pass = adf_statistic < adf_critical_value
        pval_pass = pval < 0.05
        if adf_statistic < adf_critical_value:
            print(f"ADF test statistic less than 5% critical value{adf_critical_value}. Stationarity at 5% level.")
        else:
            print(f"ADF test statistic NOT less than 5% critical value {adf_critical_value}. NOT STATIONARY at 5% level.")
        if pval < 0.05:
            print("p-value is less than 0.05, null hypothesis can be rejected. Implies stationarity.")
        else:
            print("p-value is NOT less than 0.05, null hypothesis CANNOT be rejected. Implies NON-STATIONARITY.")
        passed_test = adf_pass and pval_pass
        return passed_test

    def test_autocorr(resid):
        r, q, p = sm.tsa.acf(resid, nlags=40, qstat=True)
        significant_lags = np.where(p < 0.01)[0]
        if len(significant_lags) > 0:
            print("Residual autocorrelation found at lags : ", significant_lags + 1)
            return False
        else:
            print("No significant autocorrelation found in the residuals.")
        return True

    def test_residuals(resid):
        print("Residual testing initiated.")
        print("First, Augmented Dickey-Fuller unit root test. Tests for presence of unit root and for stationarity.")
        adf_result = test_adf(resid)
        print("Autocorrelation testing initiated.")
        autocorr_result = test_autocorr(resid)
        passed_tests = adf_result and autocorr_result
        return passed_tests

    def plot_results(ticker, test_data, preds_df, orders_df, train_acc, test_acc):
        fig, axs = plt.subplots(3, 2, figsize=(18, 12))
        # Upper left
        axs[0, 0].plot(test_data[-100:], label='true values')
        axs[0, 0].plot(preds_df["preds"][-100:], label='true values')
        axs[0, 0].set_title(f"True vs predicted log-prices for  {ticker}")
        # Upper right
        axs[0, 1].axis("off")
        table_data = orders_df.values
        cols = orders_df.columns
        rows = [f"Model {i+1}" for i in range(len(orders_df))]
        table = axs[0, 1].table(cellText=table_data, colLabels=cols,  rowLabels=rows, loc="center", cellLoc="center", colLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.auto_set_column_width(col=list(range(len(cols))))  # Adjust column width
        axs[0, 1].set_title("Top 5 Models by BIC")
        # mid left
        axs[1, 0].text(0.5, 0.7, f"Accuracy on training data : {train_acc}", ha="center", va="center", fontsize=24)
        axs[1, 0].text(0.5, 0.3, f"Accuracy on testing data : {test_acc}", ha="center", va="center", fontsize=24)
        axs[1, 0].axis("off")
        axs[1, 0].set_title("Precision score")
        # mid right
        sm.graphics.tsa.plot_acf(train_data, lags=40, ax=axs[1, 1])
        axs[1, 1].set_title("Autocorrelation (ACF)")
        # lower left
        axs[2, 0].axis("off")
        # lower right
        sm.graphics.tsa.plot_pacf(train_data, lags=40, ax=axs[2, 1])
        axs[2, 1].set_title("Partial Autocorrelation (PACF)")

        # Adjust layout to prevent clipping of titles
        plt.tight_layout()
        plt.show()

    full_data, train_data, test_data = prepare_data(df)
    p, q, top_5 = get_model_order(train_data)
    train_model = sm.tsa.ARIMA(train_data, order=(p, 0, q), trend='n')
    res = train_model.fit()

    resid_test = test_residuals(res.resid)
    if not resid_test:
        print("Test of residuals did NOT pass.")
        # INSERT CODE HERE TO SHUT DOWN PROGRAM, TELL THE USER NOTHING WILL BE RETURNED ETC
    preds = res.predict()
    MSE = ((train_data - preds)**2).mean()
    train_acc = (np.sign(train_data) == np.sign(preds)).sum() / (len(preds))
    print(f"Mean squared error : {MSE}, accuracy: {train_acc}")

    true_model = sm.tsa.ARIMA(train_data, order=(p, 0, q), trend='n')
    true_res = true_model.fit()

    preds_df = pd.DataFrame(index=test_data.index, columns=['preds'])

    for t in tqdm(preds_df.index):
        # prep data
        temp_df = full_data.loc[:t].iloc[:-1]
        temp_model = true_model.clone(temp_df.values)
        with temp_model.fix_params(true_res.params):
            res = temp_model.fit()
        preds_df.loc[t] = res.predict(len(temp_df)-1, len(temp_df))[-1]

    preds_df['true vals'] = full_data
    test_acc = (np.sign(preds_df['preds']) == np.sign(preds_df['true vals'])).sum() / (len(preds_df))
    plot_results(ticker, test_data, preds_df, top_5, train_acc, test_acc)
