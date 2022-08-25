import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet
from datetime import datetime, timedelta
from pmdarima.arima import auto_arima


data = pd.read_excel(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\Prediction_product\test_data.xlsx")

number_of_step_ahead = 30    #Number of periods that user want to predict
seasonality = True           #If user select seasonality this parameter would be True
weekly_seasonality = False   #If user select weakly seasonality this parameter would be True

def fbprophet_predictor(data, number_of_step_ahead, seasonality, weekly_seasonality):
    data.columns = ['ds', 'y']

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

    return forecast[["ds","yhat"]]

def reverse_date_to_series(data):
    data["ds"] = [x for x in range(1,1+len(data["ds"]))]
    return data

results = fbprophet_predictor(data, number_of_step_ahead, seasonality, weekly_seasonality)
results = results[len(results["ds"]) - number_of_step_ahead :]
results = reverse_date_to_series(results)




