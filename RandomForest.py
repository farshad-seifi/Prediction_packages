import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import GaussianLikelihood
from darts.models import RandomForest


data = pd.read_excel(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\Prediction_product\test_data.xlsx")

## Models parameters that give from user
number_of_step_ahead = 30       # Number of periods that user want to predict
Monthly_seasonality = True      # If user sets Monthly_seasonality True then model fits Monthly seasonality to data
Weekly_seasonality = False      # If user sets Weekly_seasonality True then model fits Weekly seasonality to data
Confidence_limit = True         # Is this parameter be Ture model generates LCL and UCL in addition to prediction
frequency = "Monthly"           # Data level of prediction -- Allowed values : Monthly & Daily
RNN_Type = "LSTM"               # If user selects RNN-Based model as a main model, he could select Model type between RNN, LSTM, and GRU
Direction = "postdict"          # If user selects postdict model removes number of selected step from data and then predict without them.
                                # After that user can see the accuarcy of prediction. Another choice for this variable is predict which doesn't have accuracy option.


## Removing last #number_of_step_ahead from data if Direction be postdict
if Direction == "postdict":
    train_data = data.iloc[:len(data["date"]) - number_of_step_ahead]
    test_data = data.iloc[len(data["date"]) - number_of_step_ahead:]
else:
    train_data = data


def monthdelta(date, delta):
    # This function calculates the difference between to date based on month
    m, y = (date.month + delta) % 12, date.year + ((date.month) + delta - 1) // 12
    if not m: m = 12
    d = min(date.day, [31,
                       29 if y % 4 == 0 and (not y % 100 == 0 or y % 400 == 0) else 28,
                       31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])

    return date.replace(day=d, month=m, year=y)

  def RandomForest_Model(data, number_of_step_ahead, frequency):

    """
    The main function for RandomForest prediction
    :param data: The main data for prediction
    :param number_of_step_ahead: Number of step ahead for prediction
    :param frequency: Type of date. Monthly or Daily
    :return: A 4*number_of_step_ahead dimension dataframe
    """

    # Handling dates format of darts
    for i in range(0, len(data["date"])):
        if (frequency == "Monthly"):
            year = (monthdelta(datetime.today(), -i)).year
            month = (monthdelta(datetime.today(), -i)).month
            day = (monthdelta(datetime.today(), -i)).day
            data["date"].iloc[len(data["date"]) - i - 1] = str(year) + "-" + str(month)
        elif (frequency == "Daily"):
            year = (datetime.today() - timedelta(days=i)).year
            month = (datetime.today() - timedelta(days=i)).month
            day = (datetime.today() - timedelta(days=i)).day
            data["date"].iloc[len(data["date"]) - i - 1] = str(year) + "-" + str(month) + "-" + str(day)

    series = TimeSeries.from_dataframe(data, 'date', 'value').astype(np.float32)
    transformer = Scaler()
    series = transformer.fit_transform(series)

    date = pd.DataFrame(np.zeros((len(data["date"]) + number_of_step_ahead, 1)))

    for i in range(0, len(data["date"]) + number_of_step_ahead):
        if (frequency == "Monthly"):
            year = (monthdelta(datetime.today(), number_of_step_ahead - i)).year
            month = (monthdelta(datetime.today(), number_of_step_ahead - i)).month
            day = (monthdelta(datetime.today(), number_of_step_ahead - i)).day
            date[0].iloc[len(date[0]) - i - 1] = str(year) + "-" + str(month)

        elif (frequency == "Daily"):
            year = (datetime.today() - timedelta(days=-number_of_step_ahead + i)).year
            month = (datetime.today() - timedelta(days=-number_of_step_ahead + i)).month
            day = (datetime.today() - timedelta(days=-number_of_step_ahead + i)).day
            data["date"].iloc[len(data["date"]) - i - 1] = str(year) + "-" + str(month) + "-" + str(day)
            date[0].iloc[len(date[0]) - i - 1] = str(year) + "-" + str(month) + "-" + str(day)

    date.columns = ["date"]
    date["value"] = 0
    series_date = TimeSeries.from_dataframe(date, 'date', 'value').astype(np.float32)

    #HPO part for reaching better response time
    # parameters = {"lags": [int(np.floor(len(data) / 10)), int(np.floor(len(data) / 5)), int(np.floor(len(data) / 3))]}
    #
    # best_model = RandomForest.gridsearch(parameters=parameters,
    #                                           series=series,
    #                                           forecast_horizon=number_of_step_ahead)


    # my_model = RandomForest(lags=best_model[1]["lags"], n_estimators= 30)
    my_model = RandomForest(lags= int(np.floor(len(data) / 3)) , n_estimators= 100)

    my_model.fit(series)
    prediction = my_model.predict(number_of_step_ahead, num_samples = 1)

    forecast = pd.DataFrame(np.zeros((number_of_step_ahead, 4)))
    forecast[0] = [x + len(data["value"]) for x in range(1, 1 + number_of_step_ahead)]
    forecast[1] = pd.DataFrame(transformer.inverse_transform(prediction).values())
    forecast.columns = ["date", "prediction", "LCL", "UCL"]

    return forecast
