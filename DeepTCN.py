import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.utils.statistics import check_seasonality
import jdatetime
from darts.models import TCNModel
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from datetime import datetime, timedelta
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import GaussianLikelihood


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



def DeepTCN_predictor(data, number_of_step_ahead, Confidence_limit):

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

        elif (frequency == "Hourly"):
            year = (datetime.today() - timedelta(hours=-number_of_step_ahead + i)).year
            month = (datetime.today() - timedelta(hours=-number_of_step_ahead + i)).month
            day = (datetime.today() - timedelta(hours=-number_of_step_ahead + i)).day
            hour = (datetime.today() - timedelta(hours=-number_of_step_ahead + i)).hour

            data["date"].iloc[len(data["date"]) - i - 1] = str(year) + "-" + str(month) + "-" + str(day) + " " + str(hour) + ":00:00"
            date[0].iloc[len(date[0]) - i - 1] = str(year) + "-" + str(month) + "-" + str(day) + " " + str(hour) + ":00:00"


    date.columns = ["date"]
    date["value"] = 0
    series_date = TimeSeries.from_dataframe(date, 'date', 'value').astype(np.float32)
    status = False

    # Adding monthly seasonality to data
    # We should add weekly too
    if (frequency == "Hourly" and (len(data["date"]) > 24) and
            (check_seasonality(series, m = 24, max_lag = len(data["date"]))[0])):

        year_data = datetime_attribute_timeseries(series_date, attribute="year")
        year_series = Scaler().fit_transform(year_data)
        hour_series = datetime_attribute_timeseries(year_series, attribute="hour", one_hot=True)
        covariates = year_series.stack(hour_series)
        status = True

    elif (frequency == "Hourly" and (len(data["date"]) > 7*24) and
          (check_seasonality(series, m = 7*24, max_lag = len(data["date"]))[0])):

        year_data = datetime_attribute_timeseries(series_date, attribute="year")
        year_series = Scaler().fit_transform(year_data)
        week_series = datetime_attribute_timeseries(year_series, attribute="weekday", one_hot=True)
        covariates = year_series.stack(week_series)
        status = True

    elif (frequency == "Hourly" and (len(data["date"]) > 30 * 24) and
          (check_seasonality(series, m=30 * 24, max_lag=len(data["date"]))[0])):

        year_data = datetime_attribute_timeseries(series_date, attribute="year")
        year_series = Scaler().fit_transform(year_data)
        month_series = datetime_attribute_timeseries(year_series, attribute="month", one_hot=True)
        covariates = year_series.stack(month_series)
        status = True

    elif (frequency == "Daily" and (len(data["date"]) > 7) and
          (check_seasonality(series, m=7, max_lag=len(data["date"]))[0])):

        year_data = datetime_attribute_timeseries(series_date, attribute="year")
        year_series = Scaler().fit_transform(year_data)
        week_series = datetime_attribute_timeseries(year_series, attribute="weekday", one_hot=True)
        covariates = year_series.stack(week_series)
        status = True

    elif (frequency == "Daily" and (len(data["date"]) > 30) and
          (check_seasonality(series, m=30, max_lag=len(data["date"]))[0])):

        year_data = datetime_attribute_timeseries(series_date, attribute="year")
        year_series = Scaler().fit_transform(year_data)
        day_series = datetime_attribute_timeseries(year_series, attribute="day", one_hot=True)
        covariates = year_series.stack(day_series)
        status = True

    elif (frequency == "Monthly" and (len(data["date"]) > 12) and
          (check_seasonality(series, m=12, max_lag=len(data["date"]))[0])):

        year_data = datetime_attribute_timeseries(series_date, attribute="year")
        year_series = Scaler().fit_transform(year_data)
        month_series = datetime_attribute_timeseries(year_series, attribute="month", one_hot=True)
        covariates = year_series.stack(month_series)
        status = True


    # Creating model object
    # Hyperparameters should be optimized.
    my_model = TCNModel(
                        input_chunk_length=30,
                        output_chunk_length=20,
                        kernel_size=2,
                        num_filters=4,
                        dilation_base=2,
                        dropout=0,
                        random_state=0,
                        likelihood=GaussianLikelihood())


    if (status):
        my_model.fit(series, past_covariates=covariates.astype(np.float32))
    else:
        my_model.fit(series)

    # Creating final alignment for forecasting

    prediction = my_model.predict(number_of_step_ahead, num_samples=1)
    if Confidence_limit:
        prediction_for_std = my_model.predict(number_of_step_ahead, num_samples=1000)
        prediction_for_std = transformer.inverse_transform(prediction_for_std).std().values()

    #Creating final alignment for forecasting
    forecast = pd.DataFrame(np.zeros((number_of_step_ahead, 4)))
    forecast[0] = [x + len(data["value"]) for x in range(1, 1 + number_of_step_ahead)]
    forecast[1] = pd.DataFrame(transformer.inverse_transform(prediction).values())
    if Confidence_limit:
        forecast[2] = - 1.96 * pd.DataFrame(prediction_for_std)[0] + pd.DataFrame(forecast[1])[1]
        forecast[3] = 1.96 * pd.DataFrame(prediction_for_std)[0] + pd.DataFrame(forecast[1])[1]

    return forecast


train_data['date'] = train_data['date'].apply(date_handler)
results = DeepTCN_predictor(train_data, number_of_step_ahead, Confidence_limit)
