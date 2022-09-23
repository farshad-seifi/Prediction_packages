import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pmdarima.arima import auto_arima
from darts import TimeSeries
from darts.utils.statistics import check_seasonality
import jdatetime


data = pd.read_excel(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\Prediction_product\test_data_month.xlsx")

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
    if((data["date"].iloc[0].hour + data["date"].iloc[1].hour) > 0 ):
        frequency = "Hourly"
    elif((data["date"].iloc[1]-data["date"].iloc[0]).days == 1):
        frequency = "Daily"
    elif(((data["date"].iloc[1]-data["date"].iloc[0]).days > 27) and ((data["date"].iloc[1]-data["date"].iloc[0]).days < 32)):
        frequency = "Monthly"
    elif(((data["date"].iloc[1]-data["date"].iloc[0]).days > 363) and ((data["date"].iloc[1]-data["date"].iloc[0]).days < 366)):
        frequency = "Yearly"

    return frequency


def monthdelta(date, delta):
    # This function calculates the difference between to date based on month
    m, y = (date.month + delta) % 12, date.year + ((date.month) + delta - 1) // 12
    if not m: m = 12
    d = min(date.day, [31,
                       29 if y % 4 == 0 and (not y % 100 == 0 or y % 400 == 0) else 28,
                       31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])

    return date.replace(day=d, month=m, year=y)


def SARIMA_predictor(data, number_of_step_ahead, Confidence_limit):
    """
    The main function for SARIMA prediction
    :param data: The main data for prediction
    :param number_of_step_ahead: Number of step ahead for prediction
    :param Confidence_limit: If be True model returns LCL and UCL
    :return: A 4*number_of_step_ahead dimension dataframe
    """
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

    # Creating model object. by this block engine search till SARIMA(2,2,2)(1,1,1)
    # and selects the best value for (p,d,q)(P,D,Q) between these values.
    # changing the range of this hyperparameter may produce better results but reduce the speed of fitting.
    if (frequency == "Hourly" and (len(data["date"]) > 24) and
            (check_seasonality(series, m = 24, max_lag = len(data["date"]))[0])):

        arima_model = auto_arima(data["value"], start_p=0, start_q=0, max_p=2, max_d=2, max_q=2, start_P=0,
                                 start_Q=0, max_P=1,
                                 max_D=1, max_Q=1, m=24, seasonal=True, error_action='ignore',
                                 trace=True,
                                 suppress_warnings=True, stepwise=True, random_state=20, n_fits=50)

    elif (frequency == "Hourly" and (len(data["date"]) > 7*24) and
          (check_seasonality(series, m = 7*24, max_lag = len(data["date"]))[0])):

        arima_model = auto_arima(data["value"], start_p=0, start_q=0, max_p=2, max_d=2, max_q=2, start_P=0,
                                 start_Q=0, max_P=1,
                                 max_D=1, max_Q=1, m=7*24, seasonal=True, error_action='ignore',
                                 trace=True,
                                 suppress_warnings=True, stepwise=True, random_state=20, n_fits=50)

    elif (frequency == "Hourly" and (len(data["date"]) > 30 * 24) and
          (check_seasonality(series, m=30 * 24, max_lag=len(data["date"]))[0])):

        arima_model = auto_arima(data["value"], start_p=0, start_q=0, max_p=2, max_d=2, max_q=2, start_P=0,
                                 start_Q=0, max_P=1,
                                 max_D=1, max_Q=1, m=30*24, seasonal=True, error_action='ignore',
                                 trace=True,
                                 suppress_warnings=True, stepwise=True, random_state=20, n_fits=50)

    elif (frequency == "Daily" and (len(data["date"]) > 7) and
          (check_seasonality(series, m=7, max_lag=len(data["date"]))[0])):

        arima_model = auto_arima(data["value"], start_p=0, start_q=0, max_p=2, max_d=2, max_q=2, start_P=0,
                                 start_Q=0, max_P=1,
                                 max_D=1, max_Q=1, m=7, seasonal=True, error_action='ignore',
                                 trace=True,
                                 suppress_warnings=True, stepwise=True, random_state=20, n_fits=50)

    elif (frequency == "Daily" and (len(data["date"]) > 30) and
          (check_seasonality(series, m=30, max_lag=len(data["date"]))[0])):

        arima_model = auto_arima(data["value"], start_p=0, start_q=0, max_p=2, max_d=2, max_q=2, start_P=0,
                                 start_Q=0, max_P=1,
                                 max_D=1, max_Q=1, m=30, seasonal=True, error_action='ignore',
                                 trace=True,
                                 suppress_warnings=True, stepwise=True, random_state=20, n_fits=50)

    elif (frequency == "Monthly" and (len(data["date"]) > 12) and
          (check_seasonality(series, m=12, max_lag=len(data["date"]))[0])):

        arima_model = auto_arima(data["value"], start_p=0, start_q=0, max_p=2, max_d=2, max_q=2, start_P=0,
                                 start_Q=0, max_P=1,
                                 max_D=1, max_Q=1, m=12, seasonal=True, error_action='ignore',
                                 trace=True,
                                 suppress_warnings=True, stepwise=True, random_state=20, n_fits=50)


    arima_model.fit(data["value"])

    future = arima_model.predict(n_periods=number_of_step_ahead, return_conf_int=Confidence_limit)

    if Confidence_limit:
        confident_limit = future[1]
        future = future[0]

    # Creating final alignment for forecasting
    forecast = pd.DataFrame(np.zeros((number_of_step_ahead, 4)))
    forecast[0] = [x + len(data["value"]) for x in range(1, 1 + number_of_step_ahead)]
    forecast[1] = future

    if Confidence_limit:
        for i in range(0, len(confident_limit)):
            forecast[2].iloc[i] = confident_limit[i][0]
            forecast[3].iloc[i] = confident_limit[i][1]

    forecast.columns = ["date", "prediction", "LCL", "UCL"]

    return forecast


train_data['date'] = train_data['date'].apply(date_handler)
results = SARIMA_predictor(train_data, number_of_step_ahead, Confidence_limit)
