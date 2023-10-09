import joblib
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# Step 3: Seasonal Decomposition
def seasonal_decompose_series(series):
    decomposition = seasonal_decompose(series["Temperature"], model='additive')
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    return trend, seasonal, residual


# Step 4: Model Identification
def identify_model_order(series):
    # Plot ACF and PACF
    sm.graphics.tsa.plot_acf(series["Temperature"], lags=30)
    sm.graphics.tsa.plot_pacf(series["Temperature"], lags=30)
    # Perform model selection based on AIC
    model = sm.tsa.statespace.SARIMAX(series["Temperature"], order=(1, 0, 1), seasonal_order=(1, 0, 1, 7))
    results = model.fit()
    print(results.summary())


# Step 5: Model Estimation and Diagnostics
def estimate_sarima_model(series, order):
    model = sm.tsa.statespace.SARIMAX(series["Temperature"], order=order, seasonal_order=(1, 0, 1, 7))
    results = model.fit()
    print(results.summary())


# Step 6: Model Forecasting
def sarima_forecast(series, order, steps):
    model = sm.tsa.statespace.SARIMAX(series["Temperature"], order=order, seasonal_order=(1, 0, 1, 7))
    results = model.fit()
    forecast = results.get_forecast(steps=steps)
    forecast_mean = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()
    return forecast_mean, confidence_intervals


# Step 7: Model Evaluation and Refinement
def evaluate_forecast(actual, forecast):
    rmse = ((forecast - actual) ** 2).mean() ** 0.5
    mae = (forecast - actual).abs().mean()
    return rmse, mae



if __name__ == '__main__':
    trends = []
    seasonalities = []
    residuals = []

    clusters = [1, 2, 3, 4, 5]
    type_models = [1, 2]
    for cluster in clusters:
        key = f'cluster{cluster}'
        stations = joblib.load("joblib_objects/cluster_stations.joblib")[key]
        for station in stations:
            results_df = joblib.load('joblib_objects/results_df.joblib')
            data = results_df[station]

            trend, seasonal, residual = seasonal_decompose_series(data)
            trends.append(trend)
            seasonalities.append(seasonal)
            residuals.append(residual)
            print('IDENTIFICATION OF MODEL')
            identify_model_order(data)

            order = (1, 0, 1)  # Replace with the appropriate order identified in Step 4
            estimate_sarima_model(data, order)