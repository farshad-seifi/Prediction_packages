import numpy as np
import pandas as pd

Bonus_data = pd.read_excel(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\Prediction_product\Bonus_table.xlsx")
data = pd.read_excel(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\Prediction_product\test_data_month.xlsx")

# The desire format of columns for targeting
data.columns = ["date", "value", "segment"]

# The column header for Bonus part
Bonus_data.columns = ["percentage", "Bonus"]

# The base price for each products selling
Base_price = 10
# The desire point of realization "The initial value should be set as 1
Economy_coefficient = 0.95
# If a total given target exists this value would be grater than 0
Given_target = 20000
# The base model for targeting
model = "SARIMA"


def Target_Calculator(data, Bonus_data, Base_price, Economy_coefficient, Given_target, model):

    """
    :param data: The historical data
    :param Bonus_data: The bonus data which
    :param Base_price: The base price for each products selling
    :param Economy_coefficient: The desire point of realization
    :param Given_target: If a total given target exists this value would be grater than 0
    :param model: The base model for targeting
    :return: Model returns Total_Target, Total_Budget, actual, targets
    """

    targets = np.zeros((len(data["segment"].unique()), 2))
    targets = pd.DataFrame(targets)
    targets.columns = ["segment", "target"]
    forecast = np.zeros((len(data["segment"].unique()), 2))
    forecast = pd.DataFrame(forecast)
    forecast.columns = ["segment", "forecast"]

    for i in range(0, len(data["segment"].unique())):

        # Creating targets table for each segment
        target_data = data[data["segment"] == data["segment"].unique()[i]]
        target_data = target_data[["date", "value"]]
        target_data["date"] = target_data["date"].apply(date_handler)


        if model == "SARIMA":
            results = SARIMA_predictor(target_data, 1, True)
        elif model == "ExponentialSmoothing":
            results = ExponentialSmoothing_predictor(target_data, 1, True)
        elif model == "RandomForest":
            results = RandomForest_Model(target_data, 1)
        elif model == "LinearRegression":
            results = LinearRegression(target_data, 1)
        elif model == "DeepTCN":
            results = DeepTCN_predictor(target_data, 1, True)
        elif model == "RNN":
            results = RNNModel_predictor(target_data, 1, "RNN")
        elif model == "LSTM":
            results = RNNModel_predictor(target_data, 1, "LSTM")
        elif model == "GRU":
            results = RNNModel_predictor(target_data, 1, "GRU")
        elif model == "NBEATS":
            results = NBEATSModel_predictor(target_data, 1)
        elif model == "fbprophet":
            results = fbprophet_predictor(target_data, 1, True)

        forecast["segment"].iloc[i] = data["segment"].unique()[i]
        forecast["forecast"].iloc[i] = results["prediction"].iloc[0]
        targets["segment"].iloc[i] = data["segment"].unique()[i]
        targets["target"].iloc[i] = results["prediction"].iloc[0]

        # This part checks that the calculated target doesn't be grater than forecasted UCL
        if (results["UCL"].iloc[0] > 0):
            targets["target"].iloc[i] = min(results["UCL"].iloc[0], (targets["target"].iloc[i]) / Economy_coefficient)
        else:
            targets["target"].iloc[i] = (targets["target"].iloc[i]) / Economy_coefficient

    # This part cascades the given target for each segment
    if Given_target > 0:
        correction_coefficient = Given_target / targets["target"].sum()
        targets["target"] = targets["target"] * correction_coefficient

    # If the type of historical data be integer this part rounds the target to integer number
    if (data.dtypes["value"] == 'int64'):
        targets["target"] = targets["target"].round()

    # Calculating percentage of realization based on target and forecast
    actual = np.zeros((len(data["segment"].unique()), 2))
    actual = pd.DataFrame(actual)
    actual.columns = ["segment", "percentage"]
    actual["percentage"] = forecast["forecast"] / targets["target"]
    actual["segment"] = targets["segment"]

    # Determining that each segment locate in what step of bonus
    coef = np.zeros((len(actual["percentage"]), 2))
    coef = pd.DataFrame(coef)
    coef.columns = ["segment", "step"]

    for i in range(0, len(coef["step"])):
        status = False
        for j in range(0, len(Bonus_data["percentage"]) - 1):
            if (actual["percentage"].iloc[i] * 100 <= Bonus_data["percentage"].iloc[j]):
                coef["step"].iloc[i] = Bonus_data["Bonus"].iloc[j]
                status = True
                break

        if not status:
            coef["step"].iloc[i] = Bonus_data["Bonus"].iloc[len(Bonus_data["percentage"]) - 1]

        coef["segment"].iloc[i] = forecast["segment"].iloc[i]

    # Calculating the total budget
    Total_Budget = ((forecast["forecast"] * (1 + coef["step"]) * Base_price).sum()).round()
    Total_Target = targets["target"].sum()

    return Total_Target, Total_Budget, actual, targets


Total_Target, Total_Budget, actual, targets = Target_Calculator(data, Bonus_data, Base_price, Economy_coefficient, Given_target, model)
