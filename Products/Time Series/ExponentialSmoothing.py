import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import ExponentialSmoothing
from darts.utils.statistics import check_seasonality
import jdatetime

data = pd.read_excel(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\Prediction_product\test_data - Copy.xlsx")
data = pd.read_excel(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\Prediction_product\test_data_month.xlsx")

# data = pd.read_csv(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\Prediction_product\Test data\Time series\AEP_hourly.csv")
# data = pd.read_csv(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\Prediction_product\Test data\Time series\EKPC_hourly.csv")

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


def ExponentialSmoothing_predictor(data, number_of_step_ahead, Confidence_limit):
    """
    The main function for ExponentialSmoothing prediction
    :param data: The main data for prediction
    :param number_of_step_ahead: Number of step ahead for prediction
    :param Monthly_seasonality: Status of Monthly seasonality
    :param Weekly_seasonality: Status of weekly seasonality
    :param frequency: Type of date. Monthly or Daily
    :param Confidence_limit: If be True model returns LCL and UCL
    :return: A 4*number_of_step_ahead dimension dataframe
    """
    # Converting dataframe format to timeseries for using darts package

    frequency = frequency_finder(data)

    data.reset_index(inplace= True, drop= True)
    data.reset_index(inplace=True)
    data["index"] += 1
    data = data[["index","value"]]
    data.columns = ["date", "value"]

    series = TimeSeries.from_dataframe(fill_missing_dates=True, freq=None, df= data, time_col= 'date', value_cols='value')

    # Creating model object
    if (frequency == "Hourly" and (len(data["date"]) >= 2*24) and
            (check_seasonality(series, m = 24, max_lag = len(data["date"]))[0])):
        model = ExponentialSmoothing(seasonal_periods= 24)

    elif (frequency == "Hourly" and (len(data["date"]) >= 2*7*24) and
          (check_seasonality(series, m = 7*24, max_lag = len(data["date"]))[0])):
        model = ExponentialSmoothing(seasonal_periods= 7 * 24)

    elif (frequency == "Hourly" and (len(data["date"]) >= 2*30 * 24) and
          (check_seasonality(series, m=30 * 24, max_lag=len(data["date"]))[0])):
        model = ExponentialSmoothing(seasonal_periods=30 * 24)

    elif (frequency == "Daily" and (len(data["date"]) >= 14) and
          (check_seasonality(series, m=7, max_lag=len(data["date"]))[0])):
        model = ExponentialSmoothing(seasonal_periods=7)

    elif (frequency == "Daily" and (len(data["date"]) >= 60) and
          (check_seasonality(series, m=30, max_lag=len(data["date"]))[0])):
        model = ExponentialSmoothing(seasonal_periods=30)

    elif (frequency == "Monthly" and (len(data["date"]) >= 24) and
          (check_seasonality(series, m=12, max_lag=len(data["date"]))[0])):
        model = ExponentialSmoothing(seasonal_periods=12)
    else:
        model = ExponentialSmoothing(seasonal= None)


    if len(data) >= 3 :
        model.fit(series)
        prediction = model.predict(number_of_step_ahead, num_samples=1)
        if Confidence_limit:
            prediction_for_std = model.predict(number_of_step_ahead, num_samples=1000)
            prediction_for_std = prediction_for_std.std().values()

        # Creating final alignment for forecasting
        forecast = pd.DataFrame(np.zeros((number_of_step_ahead, 4)))
        forecast[0] = [x + len(data["value"]) for x in range(1, 1 + number_of_step_ahead)]
        forecast[1] = pd.DataFrame(prediction.values())
        if Confidence_limit:
            forecast[2] = - 1.96 * pd.DataFrame(prediction_for_std)[0] + pd.DataFrame(forecast[1])[1]
            forecast[3] = 1.96 * pd.DataFrame(prediction_for_std)[0] + pd.DataFrame(forecast[1])[1]
    else:
        prediction = np.zeros(number_of_step_ahead)
        for i in range(0, number_of_step_ahead):
            prediction[i] = data["value"].mean()
        # Creating final alignment for forecasting
        forecast = pd.DataFrame(np.zeros((number_of_step_ahead, 4)))
        forecast[0] = [x + len(data["value"]) for x in range(1, 1 + number_of_step_ahead)]
        forecast[1] = prediction
        forecast[2] = prediction
        forecast[3] = prediction



    forecast.columns = ["date", "prediction", "LCL", "UCL"]

    return forecast


train_data['date'] = train_data['date'].apply(date_handler)
results = ExponentialSmoothing_predictor(train_data, number_of_step_ahead, Confidence_limit)
