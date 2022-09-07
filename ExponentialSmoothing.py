import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import ExponentialSmoothing

data = pd.read_excel(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\Prediction_product\test_data.xlsx")

## Models parameters that give from user
number_of_step_ahead = 30    #Number of periods that user want to predict
Monthly_seasonality = True   #If user sets Monthly_seasonality True then model fits Monthly seasonality to data
Weekly_seasonality = False   #If user sets Weekly_seasonality True then model fits Weekly seasonality to data
Confidence_limit = True      #Is this parameter be Ture model generates LCL and UCL in addition to prediction
frequency = "Monthly"        #Data level of prediction -- Allowed values : Monthly & Daily
RNN_Type = "LSTM"            #If user selects RNN-Based model as a main model, he could select Model type between RNN, LSTM, and GRU
Direction = "postdict"       #If user selects postdict model removes number of selected step from data and then predict without them.
                             #After that user can see the accuarcy of prediction. Another choice for this variable is predict which doesn't have accuracy option.

  ## Removing last #number_of_step_ahead from data if Direction be postdict
if Direction == "postdict":
    train_data = data.iloc[:len(data["date"])-number_of_step_ahead]
    test_data = data.iloc[len(data["date"])-number_of_step_ahead:]
else:
    train_data = data
    
def ExponentialSmoothing_predictor(data, number_of_step_ahead, Monthly_seasonality, Weekly_seasonality, frequency, Confidence_limit):

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
    #Converting dataframe format to timeseries for using darts package
    series = TimeSeries.from_dataframe(data, 'date', 'value')

    #Creating model object
    if Weekly_seasonality:
        model = ExponentialSmoothing(seasonal_periods = 7)
    elif (Monthly_seasonality and frequency == 'Monthly'):
        model = ExponentialSmoothing(seasonal_periods = 12)
    elif (Monthly_seasonality and frequency == 'Daily'):
        model = ExponentialSmoothing(seasonal_periods = 364)
    else:
        model = ExponentialSmoothing()


    model.fit(series)
    prediction = model.predict(number_of_step_ahead, num_samples=1)
    if Confidence_limit:
        prediction_for_std = model.predict(number_of_step_ahead, num_samples=1000)
        prediction_for_std = prediction_for_std.std().values()

    #Creating final alignment for forecasting
    forecast = pd.DataFrame(np.zeros((number_of_step_ahead, 4)))
    forecast[0] = [x + len(data["value"]) for x in range(1, 1 + number_of_step_ahead)]
    forecast[1] = pd.DataFrame(prediction.values())
    if Confidence_limit:
        forecast[2] = - 2.64 * pd.DataFrame(prediction_for_std)[0] + pd.DataFrame(forecast[1])[1]
        forecast[3] = 2.64 * pd.DataFrame(prediction_for_std)[0] + pd.DataFrame(forecast[1])[1]

    forecast.columns = ["date", "prediction", "LCL", "UCL"]

    return forecast

