import numpy as np
import pandas as pd
from darts import TimeSeries
import jdatetime
from darts.models import LinearRegressionModel
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from datetime import datetime, timedelta
from darts import metrics


data = pd.read_excel(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\Prediction_product\test_data_month.xlsx")


data.columns = ["date", "value"]

## Models parameters that give from user
number_of_step_ahead = 30   # Number of periods that user want to predict
Confidence_limit = True     # Is this parameter be Ture model generates LCL and UCL in addition to prediction
Direction = "postdict"      # If user selects postdict model removes number of selected step from data and then predict without them.
                            # After that user can see the accuarcy of prediction. Another choice for this variable is predict which doesn't have accuracy option.


## Removing last #number_of_step_ahead from data if Direction be postdict
if Direction == "postdict":
    train_data = data.iloc[:len(data["date"]) - number_of_step_ahead]
    test_data = data.iloc[len(data["date"]) - number_of_step_ahead:]
else:
    train_data = data

def date_handler(input):
    date = input
    try:
        if len(str(date)) < 8:
            if "/" in str(date):
                date = str(date) + "/01"
            elif "-" in str(date):
                date = str(date) + "-01"
            else:
                date = str(date) + "01"

        if "/" in str(date) or "-" in str(date):
            date = str(date).replace("-" , "").replace("/" , "")
        if int(str(date)[0:4]) < 1500:
            eng_date = pd.to_datetime(jdatetime.date(year = int(str(date)[0:4]), month = int(str(date)[4:6]) , day = int(str(date)[6:8])).togregorian())
        if int(str(date)[0:4]) > 1500:
            eng_date = pd.to_datetime(str(date))
        return eng_date
    except:
        return input


def frequency_finder(data):
    if(len(data) >= 2):
        if ((data["date"].iloc[0].hour + data["date"].iloc[1].hour) > 0):
            frequency = "Hourly"
        elif ((data["date"].iloc[1] - data["date"].iloc[0]).days == 1):
            frequency = "Daily"
        elif (((data["date"].iloc[1] - data["date"].iloc[0]).days > 27) and (
                (data["date"].iloc[1] - data["date"].iloc[0]).days < 32)):
            frequency = "Monthly"
        elif (((data["date"].iloc[1] - data["date"].iloc[0]).days > 363) and (
                (data["date"].iloc[1] - data["date"].iloc[0]).days < 366)):
            frequency = "Yearly"
        else:
            frequency = "None"
    else:
        frequency = "None"


    return frequency


def monthdelta(date, delta):
    # This function calculates the difference between to date based on month
    m, y = (date.month + delta) % 12, date.year + ((date.month) + delta - 1) // 12
    if not m: m = 12
    d = min(date.day, [31,
                       29 if y % 4 == 0 and (not y % 100 == 0 or y % 400 == 0) else 28,
                       31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])

    return date.replace(day=d, month=m, year=y)


def LinearRegression(data, number_of_step_ahead):

    """
    The main function for NBEATS prediction
    :param data: The main data for prediction
    :param number_of_step_ahead: Number of step ahead for prediction
    :return: A 4*number_of_step_ahead dimension dataframe
    """
    # Creating the well fitted formated based on the type of frequency
    # Handling dates format of darts

    frequency = frequency_finder(data)

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
        elif (frequency == "Hourly"):
            year = (datetime.today() - timedelta(hours=i)).year
            month = (datetime.today() - timedelta(hours=i)).month
            day = (datetime.today() - timedelta(hours=i)).day
            hour = (datetime.today() - timedelta(hours=i)).hour
            data["date"].iloc[len(data["date"]) - i - 1] = str(year) + "-" + str(month) + "-" + str(day) + " " + str(hour) + ":00:00"

    series = TimeSeries.from_dataframe(data, 'date', 'value').astype(np.float32)
    transformer = Scaler()
    series = transformer.fit_transform(series)

    if len(data) > 20:
        parameters = {"lags" : [int(np.floor(len(data)/20)) , int(np.floor(len(data)/10)),
                            int(np.floor(len(data)/5)), int(np.floor(len(data)/3)) ] }

        best_model = LinearRegressionModel.gridsearch(parameters=parameters,
                                                      series=series,
                                                      forecast_horizon=number_of_step_ahead,
                                                      metric=metrics.mse)

        my_model = LinearRegressionModel(lags=best_model[1]["lags"])

    elif len(data) >= 6:
        my_model = LinearRegressionModel(lags=int(len(data)/2))

    if len(data) >= 6:

        my_model.fit(series)
        prediction = my_model.predict(number_of_step_ahead, num_samples=1)
        forecast = pd.DataFrame(np.zeros((number_of_step_ahead, 4)))
        forecast[0] = [x + len(data["value"]) for x in range(1, 1 + number_of_step_ahead)]
        forecast[1] = pd.DataFrame(transformer.inverse_transform(prediction).values())
        forecast.columns = ["date", "prediction", "LCL", "UCL"]

    else:
        prediction = np.zeros(number_of_step_ahead)
        for i in range(0, number_of_step_ahead):
            prediction[i] = data["value"].mean()
        # Creating final alignment for forecasting
        forecast = pd.DataFrame(np.zeros((number_of_step_ahead, 4)))
        forecast[0] = [x + len(data["value"]) for x in range(1, 1 + number_of_step_ahead)]
        forecast[1] = prediction
        forecast.columns = ["date", "prediction", "LCL", "UCL"]


    return forecast


train_data['date'] = train_data['date'].apply(date_handler)
results = LinearRegression(train_data, number_of_step_ahead)
