#!/usr/bin/env python
from data_cleaning import DataCleaning

dc = DataCleaning(dataframe=df)

# Drop columns with excessive missing data
dc.drop_cols_missing_data(threshold=0.5)

# Replace negative values
dc.replace_negative_values()

# Impute missing values
for col in dc.dataframe.columns:
    if dc.dataframe[col].isnull().sum() > 0:
        if pd.api.types.is_numeric_dtype(dc.dataframe[col]):
            dc.imputing_vals_mean(col)
        else:
            dc.imputing_categorical_cols(col)

# Forward fill remaining missing values
dc.forward_fill()

# Final clean DataFrame
clean_df = dc.dataframe

import matplotlib.pyplot as plt

def plot_time_series(df, date_col, value_col):
    plt.figure(figsize=(14, 7))
    plt.plot(df[date_col], df[value_col], marker='o', linestyle='-')
    plt.title('Energy Consumption Over Time')
    plt.xlabel('Date')
    plt.ylabel('Energy Usage')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

plot_time_series(clean_df, 'date', 'energy_consumption')

# Model Implementation (ARIMA & SARIMA)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Split data into train and test
train = clean_df.iloc[:-12]
test = clean_df.iloc[-12:]

# ARIMA
arima_model = ARIMA(train_df['energy_consumption'], order=(1,1,1))
arima_fit = arima_model.fit()

# SARIMA (seasonal component)
sarima_model = SARIMAX(clean_df['energy_consumption'], order=(1,1,1), seasonal_order=(1,1,1,12))
sarima_fit = sarima_model.fit()


# MOdel evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Forecast
forecast_arima = arima_fit.get_forecast(steps=12)
forecast_arima = arima_forecast.predicted_mean

sarima_forecast = sarima_fit.get_forecast(steps=12)
sarima_forecast_series = sarima_forecast.predicted_mean

# Actual data for last 12 months
actual = clean_df['energy_consumption'][-12:]

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ARIMA metrics
arima_mae = mean_absolute_error(actual, forecast_series)
arima_rmse = np.sqrt(mean_squared_error(actual, arima_forecast_series))

# SARIMA metrics
sarima_mae = mean_absolute_error(actual, sarima_forecast_series)
sarima_rmse = np.sqrt(mean_squared_error(actual, sarima_forecast_series))

print(f"ARIMA MAE: {arima_mae}, RMSE: {arima_rmse}")
print(f"SARIMA MAE: {sarima_mae}, RMSE: {sarima_rmse}")


# Visualization & Interactive Dashboard
import dash
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(figure=fig),
    html.P("Select model:"),
    dcc.Dropdown(['ARIMA', 'SARIMA'], 'ARIMA', id='model-selector')
])

@app.callback(
    Output('forecast-graph', 'figure'),
    Input('model-selector', 'value')
)
def update_graph(selected_model):
    # Logic to dynamically switch between ARIMA and SARIMA forecasts
    pass

app.run_server(debug=True)


