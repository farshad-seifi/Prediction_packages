import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet
from datetime import datetime, timedelta
from pmdarima.arima import auto_arima
from darts import TimeSeries
from darts.models import ExponentialSmoothing
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.models import RNNModel, BlockRNNModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import GaussianLikelihood
from darts.models import TCNModel

data = pd.read_excel(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\Prediction_product\test_data.xlsx")

## Models parameters that give from user
number_of_step_ahead = 30    #Number of periods that user want to predict
seasonality = True           #If user selects fbprophet as a main model and then selects seasonality this parameter would be True
weekly_seasonality = False   #If user selects fbprophet as a main model and then selects weakly seasonality this parameter would be True
frequency = "Monthly"        #Data level of prediction -- Allowed values : Monthly & Daily
RNN_Type = "LSTM"            #If user selects RNN-Based model as a main model, he could select Model type between RNN, LSTM, and GRU
Direction = "postdict"       #If user selects postdict model removes number of selected step from data and then predict without them.
                             #After that user can see the accuarcy of prediction. Another choice for this variable is predict which doesn't have accuracy option.
Metric_name = "MAPE"         #If user selects postdict for Direction parameter, he would select accuracy metric from a valid list.


## Removing last #number_of_step_ahead from data if Direction be postdict
if Direction == "postdict":
    train_data = data.iloc[:len(data["date"])-number_of_step_ahead]
    test_data = data.iloc[len(data["date"])-number_of_step_ahead:]
else:
    train_data = data


def monthdelta(date, delta):
    #This function calculates the difference between to date based on month
    m, y = (date.month+delta) % 12, date.year + ((date.month)+delta-1) // 12
    if not m: m = 12
    d = min(date.day, [31,
            29 if y%4==0 and (not y%100==0 or y%400 == 0) else 28,
            31,30,31,30,31,31,30,31,30,31][m-1])

    return date.replace(day=d,month=m, year=y)

def fbprophet_predictor(data, number_of_step_ahead, seasonality, weekly_seasonality):
    #The main function for Fbprophet prediction

    #Changing column's name to fbprophet format
    data.columns = ['ds', 'y']

    #This loop creates desire date format ("YYYY-MM-DD") for fbprophet
    #It should be noted that for any value of frequency (Monthly or daily), we should create this format.
    for i in range(0, len(data["ds"])):
        year = (datetime.today() - timedelta(days=i)).year
        month = (datetime.today() - timedelta(days=i)).month
        day = (datetime.today() - timedelta(days=i)).day
        data["ds"].iloc[len(data["ds"])-i-1] = str(year) + "-" + str(month) + "-" + str(day)


    #Craeting model object. By selecting weekly_seasonality equals to True, the model add weekly seasonality
    #In addition to that if the seasonality sets True, Monthly seasonlity adds to model.
    m = Prophet( weekly_seasonality=weekly_seasonality)
    if (seasonality):
        m.add_seasonality(name='monthly', period=12, fourier_order=5)
    m.fit(data)
    future = m.make_future_dataframe(periods=number_of_step_ahead)
    forecast = m.predict(future)

    #Creating final alignment for forecasting
    forecast["ds"] = [x for x in range(1, 1 + len(forecast["ds"]))]
    forecast = forecast[["ds", "yhat"]]
    forecast.columns = ["date", "prediction"]
    forecast = forecast[len(forecast["date"]) - number_of_step_ahead:]

    return forecast


def SARIMA_predictor(data, number_of_step_ahead, seasonality):
    #The main function for SARIMA prediction

    #Creating model object. by this block engine search till SARIMA(2,2,2)(1,1,1)
    #and selects the best value for (p,d,q)(P,D,Q) between these values.
    #changing the range of this hyperparameter may produce better results but reduce the speed of fitting.
    arima_model = auto_arima(data["value"], start_p=0, start_q=0, max_p=2, max_d=2, max_q=2, start_P=0,
                             start_Q=0, max_P=1,
                             max_D=1, max_Q=1, m=12, seasonal=seasonality, error_action='ignore', trace=True,
                             suppress_warnings=True, stepwise=True, random_state=20, n_fits=50)

    arima_model.fit(data["value"])
    future = arima_model.predict(n_periods = number_of_step_ahead)

    #Creating final alignment for forecasting
    forecast = pd.DataFrame(np.zeros((number_of_step_ahead,2)))
    forecast[0] = [x + len(data["value"]) for x in range(1,1+number_of_step_ahead)]
    forecast[1] = future
    forecast.columns = ["date", "prediction"]

    return forecast


def ExponentialSmoothing_predictor(data, number_of_step_ahead):
    #The main function for ExponentialSmoothing prediction

    #Converting dataframe format to timeseries for using darts package
    series = TimeSeries.from_dataframe(data, 'date', 'value')
    #Creating model object
    model = ExponentialSmoothing()
    model.fit(series)
    prediction = model.predict(number_of_step_ahead, num_samples=1)

    #Creating final alignment for forecasting
    forecast = pd.DataFrame(np.zeros((number_of_step_ahead, 2)))
    forecast[0] = [x + len(data["value"]) for x in range(1, 1 + number_of_step_ahead)]
    forecast[1] = pd.DataFrame(prediction.values())
    forecast.columns = ["date", "prediction"]

    return forecast



def NBEATSModel_predictor(data, number_of_step_ahead):
    #The main function for NBEATS prediction

    #Converting dataframe format to timeseries for using darts package
    series = TimeSeries.from_dataframe(data, 'date', 'value').astype(np.float32)

    #Creating model object
    #Hyperparameters should be optimized.
    model_nbeats = NBEATSModel(input_chunk_length=30,
                               output_chunk_length=7,
                               generic_architecture=True,
                               num_stacks=10,
                               num_blocks=1,
                               num_layers=4,
                               layer_widths=512,
                               n_epochs=100,
                               nr_epochs_val_period=1,
                               batch_size=800,
                               model_name="nbeats_run")

    model_nbeats.fit(series)
    prediction = model_nbeats.predict(number_of_step_ahead)

    #Creating final alignment for forecasting
    forecast = pd.DataFrame(np.zeros((number_of_step_ahead, 2)))
    forecast[0] = [x + len(data["value"]) for x in range(1, 1 + number_of_step_ahead)]
    forecast[1] = pd.DataFrame(prediction.values())
    forecast.columns = ["date", "prediction"]

    return forecast


def RNNModel_predictor(data, number_of_step_ahead,RNN_Type, frequency, seasonality):
    #The main function for RNN prediction

    #Creating the well fitted formated based on the type of frequency
    for i in range(0, len(data["date"])):
        if(frequency == "Monthly"):
            year = (monthdelta(datetime.today(), -i)).year
            month = (monthdelta(datetime.today(), -i)).month
            day = (monthdelta(datetime.today(), -i)).day
            data["date"].iloc[len(data["date"]) - i - 1] = str(year) + "-" + str(month)
        elif(frequency == "Daily"):
            year = (datetime.today() - timedelta(days=i)).year
            month = (datetime.today() - timedelta(days=i)).month
            day = (datetime.today() - timedelta(days=i)).day
            data["date"].iloc[len(data["date"])-i-1] = str(year) + "-" + str(month) + "-" + str(day)

    #Converting dataframe format to timeseries for using darts package
    #Then scaling the value to Z form for getting better prediction
    series = TimeSeries.from_dataframe(data, 'date', 'value').astype(np.float32)
    transformer = Scaler()
    series = transformer.fit_transform(series)

    #Handling dates format of darts
    date = pd.DataFrame(np.zeros((len(data["date"])+number_of_step_ahead, 1)))

    for i in range(0, len(data["date"])+number_of_step_ahead):
        if(frequency == "Monthly"):
            year = (monthdelta(datetime.today(), number_of_step_ahead-i)).year
            month = (monthdelta(datetime.today(), number_of_step_ahead-i)).month
            day = (monthdelta(datetime.today(), number_of_step_ahead-i)).day
            date[0].iloc[len(date[0]) - i - 1] = str(year) + "-" + str(month)

        elif(frequency == "Daily"):
            year = (datetime.today() - timedelta(days=-number_of_step_ahead+i)).year
            month = (datetime.today() - timedelta(days=-number_of_step_ahead+i)).month
            day = (datetime.today() - timedelta(days=-number_of_step_ahead+i)).day
            data["date"].iloc[len(data["date"])-i-1] = str(year) + "-" + str(month) + "-" + str(day)
            date[0].iloc[len(date[0]) - i - 1] = str(year) + "-" + str(month) + "-" + str(day)


    date.columns = ["date"]
    date["value"] = 0
    series_date = TimeSeries.from_dataframe(date, 'date', 'value').astype(np.float32)


    #Adding monthly seasonality to data
    #We should add weekly too
    if(seasonality):
        year_data = datetime_attribute_timeseries(series_date, attribute="year")
        year_series = Scaler().fit_transform(year_data)
        month_series = datetime_attribute_timeseries(year_series, attribute="month", one_hot=True)

        covariates = year_series.stack(month_series)

    #Creating model object
    my_model = RNNModel(
                    model= RNN_Type,
                    hidden_dim=20,
                    dropout=0,
                    batch_size=16,
                    n_epochs=100,
                    optimizer_kwargs={"lr": 1e-3},
                    model_name="data_RNN",
                    log_tensorboard=True,
                    random_state=42,
                    training_length=20,
                    input_chunk_length=14,
                    force_reset=True,
                    save_checkpoints=True)

    if(seasonality):
        my_model.fit(series,future_covariates=covariates.astype(np.float32))
    else:
        my_model.fit(series)

    #Creating final alignment for forecasting

    prediction = my_model.predict(number_of_step_ahead)
    forecast = pd.DataFrame(np.zeros((number_of_step_ahead, 2)))
    forecast[0] = [x + len(data["value"]) for x in range(1, 1 + number_of_step_ahead)]
    forecast[1] = pd.DataFrame(transformer.inverse_transform(prediction).values())
    forecast.columns = ["date", "prediction"]

    return forecast


def DeepTCN_predictor(data, number_of_step_ahead, frequency):

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

    year_data = datetime_attribute_timeseries(series_date, attribute="year")
    year_series = Scaler().fit_transform(year_data)
    month_series = datetime_attribute_timeseries(
        year_series, attribute="month", one_hot=True)

    covariates = year_series.stack(month_series)

    deeptcn = TCNModel(
        input_chunk_length=30,
        output_chunk_length=20,
        kernel_size=2,
        num_filters=4,
        dilation_base=2,
        dropout=0,
        random_state=0,
        likelihood=GaussianLikelihood())

    deeptcn.fit(series, past_covariates=covariates.astype(np.float32))

    prediction = deeptcn.predict(number_of_step_ahead)
    forecast = pd.DataFrame(np.zeros((number_of_step_ahead, 2)))
    forecast[0] = [x + len(data["value"]) for x in range(1, 1 + number_of_step_ahead)]
    forecast[1] = pd.DataFrame(transformer.inverse_transform(prediction).values())
    forecast.columns = ["date", "prediction"]

    return forecast

results = DeepTCN_predictor(train_data, number_of_step_ahead, frequency)

def Accuracy_metric_calculator(test_data, results, Metric_name):

    test_data = test_data.reset_index(drop=True)
    results = results.reset_index(drop=True)

    if (Metric_name == "MAPE"):
        Metric_value = 100 * np.abs((results["prediction"] - test_data["value"]) / (test_data["value"])).mean()
    elif (Metric_name == "MSE"):
        Metric_value = ((results["prediction"] - test_data["value"])**2).mean()
    elif (Metric_name == "MAE"):
        Metric_value = (np.abs(results["prediction"] - test_data["value"])).mean()
    elif (Metric_name == "RMSE"):
        Metric_value = np.sqrt(((results["prediction"] - test_data["value"])**2).mean())
    elif (Metric_name == "SMAPE"):
        Metric_value = 100 * (np.abs(results["prediction"] - test_data["value"])/(np.abs(results["prediction"]) + np.abs(test_data["value"]))).mean()

    return  Metric_value

results = fbprophet_predictor(train_data, number_of_step_ahead, seasonality, weekly_seasonality)
results = SARIMA_predictor(train_data, number_of_step_ahead, seasonality)
results = ExponentialSmoothing_predictor(train_data, number_of_step_ahead)
results = NBEATSModel_predictor(train_data, number_of_step_ahead)
results = RNNModel_predictor(train_data, number_of_step_ahead, RNN_Type, frequency, seasonality)

Acc = Accuracy_metric_calculator(test_data, results, Metric_name)
