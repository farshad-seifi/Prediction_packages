import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima

data = pd.read_excel(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\Prediction_product\test_data.xlsx")

## Models parameters that give from user
number_of_step_ahead = 30    #Number of periods that user want to predict
Monthly_seasonality = True   #If user sets Monthly_seasonality True then model fits Monthly seasonality to data
Weekly_seasonality = False   #If user sets Weekly_seasonality True then model fits Weekly seasonality to data
Confidence_limit = True      #Is this parameter be Ture model generates LCL and UCL in addition to prediction
frequency = "Monthly"        #Data level of prediction -- Allowed values : Monthly & Daily
Direction = "postdict"       #If user selects postdict model removes number of selected step from data and then predict without them.
                             #After that user can see the accuarcy of prediction. Another choice for this variable is predict which doesn't have accuracy option.


## Removing last #number_of_step_ahead from data if Direction be postdict
if Direction == "postdict":
    train_data = data.iloc[:len(data["date"])-number_of_step_ahead]
    test_data = data.iloc[len(data["date"])-number_of_step_ahead:]
else:
    train_data = data


def SARIMA_predictor(data, number_of_step_ahead, Monthly_seasonality, Weekly_seasonality, frequency, Confidence_limit):

    """
    The main function for SARIMA prediction
    :param data:                  The main data for prediction
    :param number_of_step_ahead:  Number of step ahead for prediction
    :param Monthly_seasonality:   Status of Monthly seasonality
    :param Weekly_seasonality:    Status of weekly seasonality
    :param frequency:             Type of date. Monthly or Daily
    :param Confidence_limit:      If be True model returns LCL and UCL in addition to forecasts
    :return:                      A 4*number_of_step_ahead dimension dataframe
    """

    #Creating model object. by this block engine search till SARIMA(2,2,2)(1,1,1)
    #and selects the best value for (p,d,q)(P,D,Q) between these values.
    #changing the range of this hyperparameter may produce better results but reduce the speed of fitting.

    if frequency == "Monthly":
        arima_model = auto_arima(data["value"], start_p=0, start_q=0, max_p=2, max_d=2, max_q=2, start_P=0,
                                start_Q=0, max_P=1,
                                max_D=1, max_Q=1, m=12, seasonal=Monthly_seasonality, error_action='ignore', trace=True,
                                suppress_warnings=True, stepwise=True, random_state=20, n_fits=50)
    elif frequency == "Daily":
        if Weekly_seasonality:
            arima_model = auto_arima(data["value"], start_p=0, start_q=0, max_p=2, max_d=2, max_q=2, start_P=0,
                                     start_Q=0, max_P=1,
                                     max_D=1, max_Q=1, m=7, seasonal=Weekly_seasonality, error_action='ignore',
                                     trace=True,
                                     suppress_warnings=True, stepwise=True, random_state=20, n_fits=50)
        elif Monthly_seasonality:
            arima_model = auto_arima(data["value"], start_p=0, start_q=0, max_p=2, max_d=2, max_q=2, start_P=0,
                                     start_Q=0, max_P=1,
                                     max_D=1, max_Q=1, m=365, seasonal=Monthly_seasonality, error_action='ignore',
                                     trace=True,
                                     suppress_warnings=True, stepwise=True, random_state=20, n_fits=50)
        else:
            arima_model = auto_arima(data["value"], start_p=0, start_q=0, max_p=2, max_d=2, max_q=2, start_P=0,
                                     start_Q=0, max_P=1,
                                     max_D=1, max_Q=1, m=0, seasonal=Monthly_seasonality, error_action='ignore',
                                     trace=True,
                                     suppress_warnings=True, stepwise=True, random_state=20, n_fits=50)


    arima_model.fit(data["value"])


    future = arima_model.predict(n_periods = number_of_step_ahead , return_conf_int= Confidence_limit)

    if Confidence_limit:
        confident_limit = future[1]
        future = future[0]


    #Creating final alignment for forecasting
    forecast = pd.DataFrame(np.zeros((number_of_step_ahead,4)))
    forecast[0] = [x + len(data["value"]) for x in range(1,1+number_of_step_ahead)]
    forecast[1] = future

    if Confidence_limit:
        for i in range(0, len(confident_limit)):
            forecast[2].iloc[i] = confident_limit[i][0]
            forecast[3].iloc[i] = confident_limit[i][1]

    forecast.columns = ["date", "prediction", "LCL", "UCL"]

    return forecast
