import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet
from datetime import datetime, timedelta
from pmdarima.arima import auto_arima
from darts import TimeSeries
from darts.models import ExponentialSmoothing
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller

data = pd.read_excel(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\Prediction_product\test_data.xlsx")

number_of_step_ahead = 30    #Number of periods that user want to predict
seasonality = True           #If user select seasonality this parameter would be True
weekly_seasonality = False   #If user select weakly seasonality this parameter would be True

def fbprophet_predictor(data, number_of_step_ahead, seasonality, weekly_seasonality):

    #Changing column's name to fbprophet format
    data.columns = ['ds', 'y']

    #
    for i in range(0, len(data["ds"])):
        year = (datetime.today() - timedelta(days=i)).year
        month = (datetime.today() - timedelta(days=i)).month
        day = (datetime.today() - timedelta(days=i)).day
        data["ds"].iloc[len(data["ds"])-i-1] = str(year) + "-" + str(month) + "-" + str(day)


    m = Prophet( weekly_seasonality=weekly_seasonality)
    if (seasonality):
        m.add_seasonality(name='monthly', period=12, fourier_order=5)
    m.fit(data)
    future = m.make_future_dataframe(periods=number_of_step_ahead)
    forecast = m.predict(future)
    forecast["ds"] = [x for x in range(1, 1 + len(forecast["ds"]))]
    forecast = forecast[["ds", "yhat"]]
    forecast.columns = ["date", "prediction"]
    forecast = forecast[len(forecast["date"]) - number_of_step_ahead:]

    return forecast



results = fbprophet_predictor(data, number_of_step_ahead, seasonality, weekly_seasonality)


def SARIMA_predictor(data, number_of_step_ahead, seasonality):
    arima_model = auto_arima(data["value"], start_p=0, start_q=0, max_p=2, max_d=2, max_q=2, start_P=0,
                             start_Q=0, max_P=1,
                             max_D=1, max_Q=1, m=12, seasonal=seasonality, error_action='ignore', trace=True,
                             suppress_warnings=True, stepwise=True, random_state=20, n_fits=50)

    arima_model.fit(data["value"])
    future = arima_model.predict(n_periods = number_of_step_ahead)
    forecast = pd.DataFrame(np.zeros((number_of_step_ahead,2)))
    forecast[0] = [x + len(data["value"]) for x in range(1,1+number_of_step_ahead)]
    forecast[1] = future
    forecast.columns = ["date", "prediction"]

    return forecast

results = SARIMA_predictor(data, number_of_step_ahead, seasonality)

def ExponentialSmoothing_predictor(data, number_of_step_ahead):

    series = TimeSeries.from_dataframe(data, 'date', 'value')
    model = ExponentialSmoothing()
    model.fit(series)
    prediction = model.predict(number_of_step_ahead, num_samples=1)
    forecast = pd.DataFrame(np.zeros((number_of_step_ahead, 2)))
    forecast[0] = [x + len(data["value"]) for x in range(1, 1 + number_of_step_ahead)]
    forecast[1] = pd.DataFrame(prediction.values())
    forecast.columns = ["date", "prediction"]

    return forecast

results = ExponentialSmoothing_predictor(data, number_of_step_ahead)


def NBEATSModel_predictor(data, number_of_step_ahead):
    series = TimeSeries.from_dataframe(data, 'date', 'value').astype(np.float32)

    model_nbeats = NBEATSModel(input_chunk_length=30,
                               output_chunk_length=7,
                               generic_architecture=True,
                               num_stacks=10,
                               num_blocks=1,
                               num_layers=4,
                               layer_widths=512,
                               n_epochs=100,
                               nr_epochs_val_period=1,
                               batch_size=800,
                               model_name="nbeats_run")

    model_nbeats.fit(series)
    prediction = model_nbeats.predict(number_of_step_ahead)
    forecast = pd.DataFrame(np.zeros((number_of_step_ahead, 2)))
    forecast[0] = [x + len(data["value"]) for x in range(1, 1 + number_of_step_ahead)]
    forecast[1] = pd.DataFrame(prediction.values())
    forecast.columns = ["date", "prediction"]

    return forecast

results = NBEATSModel_predictor(data, number_of_step_ahead)

