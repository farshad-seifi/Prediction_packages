import numpy as np
import pandas as pd
from fbprophet import Prophet
from datetime import datetime, timedelta
from darts import TimeSeries
from darts.utils.statistics import check_seasonality
import jdatetime


# data = pd.read_excel(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\Prediction_product\test_data.xlsx")
data = pd.read_excel(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\Prediction_product\test_data - Copy.xlsx")

## Models parameters that give from user
number_of_step_ahead = 30       # Number of periods that user want to predict
Confidence_limit = True         # Is this parameter be Ture model generates LCL and UCL in addition to prediction
Direction = "postdict"           # If user selects postdict model removes number of selected step from data and then predict without them.
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


def fbprophet_predictor(data, number_of_step_ahead, Confidence_limit):
    """
    The main function for Fbprophet prediction
    :param data: The main data for prediction
    :param number_of_step_ahead: Number of step ahead for prediction
    :param Confidence_limit: If be True model returns LCL and UCL
    :return: A 4*number_of_step_ahead dimension dataframe
    """

    frequency = frequency_finder(data)

    if len(data) >=6 and frequency != "None":
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
                data["date"].iloc[len(data["date"]) - i - 1] = str(year) + "-" + str(month) + "-" + str(
                    day) + " " + str(hour) + ":00:00"

        series = TimeSeries.from_dataframe(data, 'date', 'value').astype(np.float32)

        # Adding monthly seasonality to data
        m = Prophet()

        if (frequency == "Hourly" and (len(data["date"]) > 24) and
                (check_seasonality(series, m=24, max_lag=len(data["date"]))[0])):

            m.add_seasonality(name='Daily', period=24, fourier_order=5)

        elif (frequency == "Hourly" and (len(data["date"]) > 7 * 24) and
              (check_seasonality(series, m=7 * 24, max_lag=len(data["date"]))[0])):

            m.add_seasonality(name='Weekly', period=7 * 24, fourier_order=5)

        elif (frequency == "Hourly" and (len(data["date"]) > 30 * 24) and
              (check_seasonality(series, m=30 * 24, max_lag=len(data["date"]))[0])):

            m.add_seasonality(name='Monthly', period=30 * 24, fourier_order=5)

        elif (frequency == "Daily" and (len(data["date"]) > 7) and
              (check_seasonality(series, m=7, max_lag=len(data["date"]))[0])):

            m.add_seasonality(name='Weekly', period=7, fourier_order=5)

        elif (frequency == "Daily" and (len(data["date"]) > 30) and
              (check_seasonality(series, m=30, max_lag=len(data["date"]))[0])):

            m.add_seasonality(name='Monthly', period=30, fourier_order=5)

        elif (frequency == "Monthly" and (len(data["date"]) > 12) and
              (check_seasonality(series, m=12, max_lag=len(data["date"]))[0])):

            m.add_seasonality(name='Yearly', period=12, fourier_order=5)

    else:
        m = Prophet()


    # Changing column's name to fbprophet format
    data.columns = ['ds', 'y']

    # This loop creates desire date format ("YYYY-MM-DD") for fbprophet
    # It should be noted that for any value of frequency (Monthly or daily), we should create this format.
    for i in range(0, len(data["ds"])):
        year = (datetime.today() - timedelta(days=i)).year
        month = (datetime.today() - timedelta(days=i)).month
        day = (datetime.today() - timedelta(days=i)).day
        data["ds"].iloc[len(data["ds"]) - i - 1] = str(year) + "-" + str(month) + "-" + str(day)

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



train_data['date'] = train_data['date'].apply(date_handler)
results = fbprophet_predictor(train_data, number_of_step_ahead, Confidence_limit)
