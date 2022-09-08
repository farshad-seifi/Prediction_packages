import numpy as np
import pandas as pd
from fbprophet import Prophet



data = pd.read_excel(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\Prediction_product\test_data.xlsx")

## Models parameters that give from user
number_of_step_ahead = 30     # Number of periods that user want to predict
Monthly_seasonality = True    # If user sets Monthly_seasonality True then model fits Monthly seasonality to data
Weekly_seasonality = False    # If user sets Weekly_seasonality True then model fits Weekly seasonality to data
Confidence_limit = True       # Is this parameter be Ture model generates LCL and UCL in addition to prediction
frequency = "Monthly"         # Data level of prediction -- Allowed values : Monthly & Daily
RNN_Type = "LSTM"             # If user selects RNN-Based model as a main model, he could select Model type between RNN, LSTM, and GRU
Direction = "postdict"        # If user selects postdict model removes number of selected step from data and then predict without them.
                              # After that user can see the accuarcy of prediction. Another choice for this variable is predict which doesn't have accuracy option.

## Removing last #number_of_step_ahead from data if Direction be postdict
if Direction == "postdict":
    train_data = data.iloc[:len(data["date"]) - number_of_step_ahead]
    test_data = data.iloc[len(data["date"]) - number_of_step_ahead:]
else:
    train_data = data


def fbprophet_predictor(data, number_of_step_ahead, Monthly_seasonality, Weekly_seasonality, frequency, Confidence_limit):
    """
    The main function for Fbprophet prediction
    :param data: The main data for prediction
    :param number_of_step_ahead: Number of step ahead for prediction
    :param Monthly_seasonality: Status of Monthly seasonality
    :param Weekly_seasonality: Status of weekly seasonality
    :param frequency: Type of date. Monthly or Daily
    :param Confidence_limit: If be True model returns LCL and UCL
    :return: A 4*number_of_step_ahead dimension dataframe
    """

    # Changing column's name to fbprophet format
    data.columns = ['ds', 'y']

    # This loop creates desire date format ("YYYY-MM-DD") for fbprophet
    # It should be noted that for any value of frequency (Monthly or daily), we should create this format.
    for i in range(0, len(data["ds"])):
        year = (datetime.today() - timedelta(days=i)).year
        month = (datetime.today() - timedelta(days=i)).month
        day = (datetime.today() - timedelta(days=i)).day
        data["ds"].iloc[len(data["ds"]) - i - 1] = str(year) + "-" + str(month) + "-" + str(day)

    # Craeting model object. By selecting weekly_seasonality equals to True, the model add weekly seasonality
    # In addition to that if the seasonality sets True, Monthly seasonlity adds to model.
    m = Prophet(weekly_seasonality= Weekly_seasonality)
    if (Monthly_seasonality and frequency == 'Daily'):
        m.add_seasonality(name='monthly', period= 365, fourier_order=5)
    elif (Monthly_seasonality and frequency == 'Monthly'):
        m.add_seasonality(name='monthly', period= 12, fourier_order=5)

    m.fit(data)
    future = m.make_future_dataframe(periods=number_of_step_ahead)
    forecast = m.predict(future)

    # Creating final alignment for forecasting
    forecast["ds"] = [x for x in range(1, 1 + len(forecast["ds"]))]
    if Confidence_limit:
        forecast = forecast[["ds", "yhat","yhat_lower", "yhat_upper"]]
    else:
        forecast = forecast[["ds", "yhat"]]
        forecast["UCL"] = 0
        forecast["LCL"] = 0

    forecast.columns = ["date", "prediction", "LCL", "UCL"]
    forecast = forecast[len(forecast["date"]) - number_of_step_ahead:]

    return forecast
